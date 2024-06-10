# Program to register individual OMEGA observations into HEALPix maps using healpy module;
# 13/05/24
# Yann Leseigneur
# Last modified by Yann Leseigneur : 13/05/24

##### Modules importation 
import numpy as np
import healpy as hp
from copy import copy

##### Functions
def px_omega2healpix_func_v2(nside, lon_values, lat_values, lonlat_shape=None, reshape=True): #Function from global_enregistrement_donnees_new_etalonnageOMEGAetMER_T.py
    """Search corresponding initial pixels to HEALPix ones.

    Parameters
    ==========
    nside: int
        The nside parameter of healpy.
    lon_values: array (floats)
        The longitudes values.
    lat_values: array (floats)
        The latitudes values.
    lonlat_shape: tuple (optional, None)
        The shape of the longitude and latitude grid.
    reshape: bool (optional, True)
        The exit format of the pixel indexes (1D-array or 2D-array of the shape of the latitude grid).
    
    Returns
    =======
    px_ini2healpix: 1D-array (int)
        The HEALPix pixel indexes that correspond to the initial pixels.                       
    """
    if len(np.shape(lon_values)) == 1:
        lon_values_grid = copy.deepcopy(lon_values)
        lon_values_grid = lon_values_grid[np.newaxis,:].repeat(lonlat_shape[0], axis=0)
    else: 
        lon_values_grid = copy.deepcopy(lon_values)
    if len(np.shape(lat_values)) == 1:
        lat_values_grid = copy.deepcopy(lat_values)
        lat_values_grid = lat_values_grid[:,np.newaxis].repeat(lonlat_shape[1], axis=1)
    else:
        lat_values_grid = copy.deepcopy(lat_values)
    
    lon_values_1D = lon_values_grid.ravel()
    lat_values_1D = lat_values_grid.ravel()
    
    #Montabone longitudes [-180, 180]°E, HEALPix longitudes [0, 360]°E, => Montabone longitude conversion
    if np.sum(lon_values_1D < 0.)>0:
        cond_lon_neg = lon_values_1D < 0.
        lon_values_1D[cond_lon_neg] = lon_values_1D[cond_lon_neg] + 360.

    # Search the 4 nearest neighbours (and their coordinates)
    ind_4px_obs = hp.pixelfunc.get_interp_weights(nside, lon_values_1D, lat_values_1D, lonlat=True)[0]
    lon_4px_obs, lat_4px_obs = hp.pixelfunc.pix2ang(nside, ind_4px_obs, lonlat=True)
    # Compute the norm of a vector (difference between 4 HEALPix neighboors pixels and initial pixels) and select the smallest one
    r_4px_obs = np.sqrt((lon_4px_obs-lon_values_1D)**2 + (lat_4px_obs-lat_values_1D)**2)
    r_4px_obs_argmin = np.argmin(r_4px_obs, axis=0)
    # Compute the corresponding between inital pixels to HEALPix ones
    px_ini2healpix = []
    for i in range(len(r_4px_obs_argmin)):
        px_ini2healpix.append(ind_4px_obs[r_4px_obs_argmin[i],i])
    px_ini2healpix = np.array(px_ini2healpix)
    if reshape is True:
        px_ini2healpix = np.reshape(px_ini2healpix, np.shape(lat_values_grid))
     #for each initial pixel, the corresponding HEALPix pixel index
    return px_ini2healpix

def register_1OMEGAobs_to_1HEALPix_map(nside, OMEGA_obs_grid, OMEGA_obs_lon_grid, OMEGA_obs_lat_grid):
    """Register individual OMEGA observations to an HEALPix by averaging.

    Parameters
    ==========
    nside: int
        The nside parameter of healpy.
    OMEGA_obs_grid: 2D-array (float)
        The Montabone map for one Ls.
    OMEGA_obs_lon_grid: array (floats)
        The longitudes values.
    OMEGA_obs_lat_grid: array (floats)
        The latitudes values.

    =======
    hp_grid: 1D-array (float)
        The HEALPix grid of the Montabone map.                       
    """
    lonlat_shape = np.shape(OMEGA_obs_grid)
    hp_grid = np.ones(hp.pixelfunc.nside2npix(nside), dtype=float)*np.nan #Initialization of a HEALPix grid
    index_px_Montabone_toHEALPix = px_omega2healpix_func_v2(nside, OMEGA_obs_lon_grid, OMEGA_obs_lat_grid, lonlat_shape=lonlat_shape, reshape=True)
    for i in range(len(hp_grid)):
        cond_HEALPix_px = index_px_Montabone_toHEALPix == i
        hp_grid[i] = np.mean(OMEGA_obs_grid[cond_HEALPix_px])
    return hp_grid

##### Main program 

NSIDE = 16 # The parameter that fixed the sampling to discretise the Martian sphere
NPIX = hp.nside2npix(NSIDE)
# An HEALPix map is a 1D-array (3072 elements for NSIDE=16)
pixel_indexes = np.arange(0, NPIX, 1, dtype=int) # Each pixel of the HEALPix map is defined by it index (from 0 to 3071 for NSIDE=16)
# You can acess the longitude/latitude coordinate of each HEALPix pixel center using hp.pixelfunc.pix2ang (such as below)
longitudes_all_HEALPix_pixels, latitudes_all_HEALPix_pixels = hp.pixelfunc.pix2ang(NSIDE, pixel_indexes, lonlat=True)

# Your dust optical depth, latitude and longitude grids of your individual OMEGA observation:
tau_dust_one_OMEGA_obs = np.zeros((2000, 32))
latitude_one_OMEGA_obs = np.zeros((2000, 32))
longitude_one_OMEGA_obs = np.zeros((2000, 32))

# The corresponding HEALPix map of the individual OMEGA observation
HEALPix_map_tau_dust_one_OMEGA_obs = register_1OMEGAobs_to_1HEALPix_map(NSIDE, tau_dust_one_OMEGA_obs,
                                            longitude_one_OMEGA_obs, latitude_one_OMEGA_obs)

# If you do this for all your observations, you can average the HEALPix maps regarding the Ls (up to 5° width) 
# and then average the longitude to make the Lat/Ls diagram. In my code, I accumulate all the OMEGA observations
# in a 5° Ls wide and then made an average on the longitude to create the Lat/Ls diagram.