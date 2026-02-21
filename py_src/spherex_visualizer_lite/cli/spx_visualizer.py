import numpy as np
import healpy as hl
import matplotlib.pyplot as pl
from matplotlib import cm
from astropy import units as u
import pandas as pd
import time

# get and parse command line arguments
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
req = parser.add_argument_group('required arguments')
req.add_argument('-f','--filename',required=True,\
    help='Filename to read the survey plan file')
req.add_argument('-s','--survey-number',required=True,type=int,\
    help='Number of the survey: [1,2,3,4] for main surveys, or [8888,9999] for deep north and deep south respectively, or [13,24] for Nyquist-sampled composites of two al-sky surveys (i.e. 13 = surveys 1 and 3, 24 = surveys 2 and 4)')
req.add_argument('-N','--number-of-cores',required=False,default=4,type=int,\
    help='Number of CPU cores used for processing. Default is 4')
args = parser.parse_args()

survey = args.survey_number

Ncores = args.number_of_cores

import os
os.system('mkdir maps'+str(survey))
os.system('rm maps'+str(survey)+'/*')

NSIDE = 4096 
NPIX = hl.nside2npix(NSIDE)

# SPHEREx Geometry
CHANNEL_WIDTH  = 3.5 * u.deg
CHANNEL_HEIGHT = 11.85 * u.arcmin
SMILE_SAG      = 8.0 * u.arcmin

OFFSET_X       = np.concatenate((-1.06*CHANNEL_WIDTH*np.ones(17),     np.zeros(17)*u.deg,  1.06*CHANNEL_WIDTH*np.ones(17)))
OFFSET_Y       = np.concatenate((          np.arange(-8,9)*11.80,  np.arange(-8,9)*11.80,  np.arange(-8,9)*11.80))         * u.arcmin

if survey==13 or survey==24:
    # we must be in Nyquist sampling evaluation mode, cut the width in half artificially
    CHANNEL_HEIGHT = CHANNEL_HEIGHT / 2.0
    # and double the channel count
    OFFSET_X       = np.concatenate((-1.06*CHANNEL_WIDTH*np.ones(17*2),     np.zeros(17*2)*u.deg,  1.06*CHANNEL_WIDTH*np.ones(17*2)))
    OFFSET_Y       = np.concatenate((          np.arange(-8,9,0.5)*11.80,  np.arange(-8,9,0.5)*11.80,  np.arange(-8,9,0.5)*11.80))         * u.arcmin
    
# note, this function AI-written
# the idea is to create a polygon that approximates the smile shape of a SPHEREx spectral channel
def get_smile_polygon(width, height, sag, offset_x, offset_y, steps=30):
    w = width.to(u.deg).value
    h = height.to(u.deg).value
    s = sag.to(u.deg).value
    ox = offset_x.to(u.deg).value
    oy = offset_y.to(u.deg).value
    
    if s == 0: R = np.inf
    else:      R = (w**2) / (8 * s)
    
    x = np.linspace(-w/2, w/2, steps)
    y_curve = R - np.sqrt(R**2 - x**2)
    y_curve -= np.mean(y_curve)
    
    x_bot, y_bot = x, y_curve - h/2
    x_top, y_top = x[::-1], (y_curve + h/2)[::-1]
    
    return np.concatenate([x_bot, x_top]) + ox, np.concatenate([y_bot, y_top]) + oy

def get_rotation_matrix(lon_deg, lat_deg, PA_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    PA  = np.radians(PA_deg)

    # precompute trig
    clat, slat = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    cpa, spa   = np.cos(PA),  np.sin(PA)

    M = np.array([[clat*clon, -cpa*slon - spa*clon*slat,  spa*slon - cpa*clon*slat],
                [clat*slon,  cpa*clon - spa*slat*slon, -spa*clon - cpa*slat*slon],
                [slat,       spa*clat,                  cpa*clat]])
    return M

# note, this function is AI-written
# this is a function to call the healpy query_polygon function
# but with the non-concave shape of a SPHEREx spectral channel outline
def query_concave_strip_optimized(nside, v_poly_sky):
    n_points = v_poly_sky.shape[0]
    steps = n_points // 2
    
    # preallocate list for pixel arrays
    pixel_batches = []
    
    for i in range(steps - 1):
        # vertex indices for the current trapezoid segment
        # bottom: i, i+1  |  top: corresponding reverse indices
        idx_bot_1 = i
        idx_bot_2 = i + 1
        idx_top_2 = n_points - 1 - (i + 1)
        idx_top_1 = n_points - 1 - i
        
        # extract 4 vertices
        quad_verts = v_poly_sky[[idx_bot_1, idx_bot_2, idx_top_2, idx_top_1]]
        
        # query and append
        pixel_batches.append(hl.query_polygon(nside, quad_verts, inclusive=True))

    # fast merge of all pixels
    if not pixel_batches:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(pixel_batches))

# load the survey plan
df = pd.read_csv(args.filename)

# pick which survey
if survey==1:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    thisdf = df[(df['PositionAngle']==0) & (df['Day']<380.5)]
elif survey==2:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    thisdf = df[(df['PositionAngle']==180) & (df['Day']<380.5)]
elif survey==3:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    thisdf = df[(df['PositionAngle']==0) & (df['Day']>380.5)]
elif survey==4:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    thisdf = df[(df['PositionAngle']==180) & (df['Day']>380.5)]
elif survey==13:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    # nyquist half-step survey
    thisdf = df[(df['PositionAngle']==0)]
elif survey==24:
    # just do all sky
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='all_sky')]
    # nyquist half-step survey
    thisdf = df[(df['PositionAngle']==180)]
elif survey==8888:
    # deep north
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='deep_north')]
elif survey==9999:
    # deep south
    df = df[(df['EngFlag']=='arrive_to_sci') & (df['SciFlag']=='deep_south')]
else:
    print('no survey specified')

##################################################################################################################
##################################################################################################################
### # DELETE THIS
### # do a test where we only analyze survey 1, but do the halfsize channels anyway
### thisdf = df[(df['PositionAngle']==0) & (df['Day']<380.5)]

# get out the lons, lats and PA angles
lons = np.array(df['TargetLon'])
lats = np.array(df['TargetLat'])
pas  = np.array(df['PositionAngle'])

def calc_hitmap(chnum):
    # generate spectral channel outline
    px, py = get_smile_polygon(CHANNEL_WIDTH, CHANNEL_HEIGHT, SMILE_SAG, OFFSET_X[chnum], OFFSET_Y[chnum])

    # convert to 3D unit vectors
    v_inst_ref = np.array(hl.ang2vec(px, py, lonlat=True)).T

    # accumulate hits for a full map simulation
    hit_map = np.zeros(NPIX).astype('uint8')

    for i in range(len(lons)):
        if not np.mod(i,1000):
            print('chnum = '+str(chnum).zfill(3)+': i = '+str(i)+' of '+str(len(lons)))
        # calculate rotation matrix
        R = get_rotation_matrix(lons[i], lats[i], pas[i])
        
        # rotate spectral channel outline here
        v_sky = (R @ v_inst_ref).T
        
        # find which healpix pixels are in the outline shape
        pix_indices = query_concave_strip_optimized(NSIDE, v_sky)
        
        # accumulate
        hit_map[pix_indices] += 1

    # save the result to a file
    np.savez_compressed('maps'+str(survey)+'/hitmap_ch'+str(chnum).zfill(3)+'.npz',hit_map=hit_map)

    return

def main():
    # parallelize processing spectral channels
    import multiprocessing
    with multiprocessing.Pool(processes=Ncores) as p:
        p.map(calc_hitmap,np.arange(len(OFFSET_X)))

    if survey not in [8888,9999]:
        # all sky
        # make plots
        from matplotlib import cm
            
        # load up the maps to calculate the number of observations per pixel
        num_spec_obs = np.zeros(NPIX).astype('uint16')
        for chnum in range(len(OFFSET_X)):
            h = np.load('maps'+str(survey)+'/hitmap_ch'+str(chnum).zfill(3)+'.npz')
            num_spec_obs[h['hit_map']>=1] += 1

        vox_comp = np.sum(num_spec_obs)/(len(OFFSET_X)*len(num_spec_obs))

        from healpy.newvisufunc import projview

        projview(num_spec_obs*2,min=0,max=len(OFFSET_X)*2,cbar=True,cmap=cm.YlGnBu,
                title='Voxel Completeness in All-Sky Survey #'+str(survey)+': '+str(100*vox_comp)[0:7]+'%',
                graticule=True, graticule_labels=True,xtick_label_color='white',
                latitude_grid_spacing=45,longitude_grid_spacing=45,
                cb_orientation='vertical',unit='# of Spectral Channels')

        pl.gcf().set_size_inches(10.5, 4.5)

        pl.text(7.20,-1.45,"Ecliptic Coordinates\nMollweide Projection",style='italic')

        pl.tight_layout()

        pl.savefig('spectral_coverage_survey'+str(survey)+'.png',dpi=800)

    if survey==8888 or survey==9999:
        # deep survey
        num_complete_obs = np.zeros(NPIX).astype('uint16') + np.iinfo(np.uint16).max
        # for each channel
        for chnum in range(len(OFFSET_X)):
            # load this hitmap
            hitmap = np.array(np.load('maps'+str(survey)+'/hitmap_ch'+str(chnum).zfill(3)+'.npz')['hit_map'])
            # each pixel can only have as many complete spectra as the channel with the lowest number of hits
            # for each part of the hitmap for which this channel has fewer than the previous "candidate" number of hits
            # set those pixels equal to these pixels
            i_where_less = np.where(hitmap < num_complete_obs)[0]
            num_complete_obs[i_where_less] = hitmap[i_where_less]

    # from https://github.com/zonca/paperplots/blob/master/python/scripts/PlanckFig_map.py
    from matplotlib.projections.geo import GeoAxes

    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        """Shifts labelling by pi
        Shifts labelling from -180,180 to 0-360"""
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2*np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    def lon_to_plot(lon,lat):
        if len(lon)==0:
            # special case for matplotlib projection
            return [100],[100]
        else:
            return -(np.mod(lon+180,360)-180)*np.pi/180, lat*np.pi/180

    # get utility function from healpy for coordinate conversion
    from healpy import Rotator
    r = Rotator(coord=['G','E'])
    def gal_to_eclip(gal_lon,gal_lat):
        return r(gal_lon,gal_lat,lonlat=True)

    if survey==8888:  
        #A = none
        #B = Greys
        #C = hot
        #D = YlGnBu

        f = pl.figure(figsize=(6.0,4.5))
        hl.visufunc.gnomview(map=num_complete_obs,rot=(0,90,0),xsize=550,fig=f,cbar=False,title=r'Northern Deep Field',badcolor='white',bgcolor='white',notext=True,cmap=cm.YlGnBu)
        hl.graticule(local=False,dmer=22.5)
        lons = np.arange(-180,181,1)
        hl.projplot(lons,85+np.zeros_like(lons),':k',lonlat=True)
        pl.legend(loc=2)
        #hl.visufunc.projtext(-151,84.9,'85 deg\neclip. lat',lonlat=True)
        #hl.visufunc.projtext(-1,84.2,'0 deg\neclip. lon',lonlat=True,color='grey')
        #hl.visufunc.projtext(20.2,83.9,'22.5 deg\neclip. lon',lonlat=True,color='grey')

        ax = pl.gca()
        im = ax.get_images()[0]
        cmap = f.colorbar(im,ax=ax)

        pl.savefig('spectral_coverage_deep_north.png',dpi=900)

    if survey==9999:
        f = pl.figure(figsize=(6.0,4.5))
        hl.visufunc.gnomview(map=num_complete_obs,rot=(44.8,-82,0),xsize=550,fig=f,cbar=False,title=r'Southern Deep Field',badcolor='white',bgcolor='white',notext=True,cmap=cm.YlGnBu)
        hl.graticule(local=False,dmer=22.5)
        pl.legend(loc=2)
        #hl.visufunc.projtext(102,-84.2,'90 deg\neclip. lon',lonlat=True,color='grey')
        #hl.visufunc.projtext(90,-87.5,'67.5 deg\neclip. lon',lonlat=True,color='grey')

        def generate_great_circle_points(center_lon, center_lat, radius_deg, step_deg=1):
            # convert to radians
            lat0 = np.radians(center_lat)
            lon0 = np.radians(center_lon)
            radius_rad = np.radians(radius_deg)

            # bearings from 0 to 360 degrees
            bearings_deg = np.arange(0, 361, step_deg)
            bearings_rad = np.radians(bearings_deg)

            # compute points using spherical formulas
            # NOTE these formulas are from ChatGPT
            # TODO verify these formulas by hand or replace with human-derived formula
            lat_points = np.arcsin(np.sin(lat0)*np.cos(radius_rad) + np.cos(lat0)*np.sin(radius_rad)*np.cos(bearings_rad))
            lon_points = lon0 + np.arctan2(np.sin(bearings_rad)*np.sin(radius_rad)*np.cos(lat0), np.cos(radius_rad) - np.sin(lat0)*np.sin(lat_points))

            # convert back to degrees
            lat_points_deg = np.degrees(lat_points)
            lon_points_deg = np.degrees(lon_points)

            # normalize longitude to [-180, 180]
            lon_points_deg = np.mod(lon_points_deg + 180,360) - 180

            return lon_points_deg,lat_points_deg

        lons_to_plot,lats_to_plot = generate_great_circle_points(44.8, -82.0, 5, step_deg=1)
        #hl.projplot(lons_to_plot,lats_to_plot,':k',lonlat=True)
        hl.projplot(np.arange(-180,180),np.zeros_like(np.arange(-180,180))+-85,':k',lonlat=True)
        hl.projplot(np.arange(-180,180),np.zeros_like(np.arange(-180,180))+-80,':k',lonlat=True)
        #hl.visufunc.projtext(39,-77.3,'5 deg radius around\neclip. lon=44.8\n          lat=-82.0',lonlat=True)

        ax = pl.gca()
        im = ax.get_images()[0]
        cmap = f.colorbar(im,ax=ax)

        pl.savefig('spectral_coverage_deep_south.png',dpi=900)

if __name__ == "__main__":
    main()
