import matplotlib.pylab as plt
import numpy as np
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname_obs = '/Users/nyonn/IDLWorkspace/Default/savfile/Table_SP_obs_calc_orb0920_3.sav'
sav_data_obs = readsav(sav_fname_obs)

sav_fname_lut = '/Users/nyonn/IDLWorkspace/Default/savfile/Table_SP_Trans_calc.sav'
sav_data_lut = readsav(sav_fname_lut)

pressure15 = sav_data_lut['table_equivalent_pressure15']
observation = sav_data_obs['obs_spec']
