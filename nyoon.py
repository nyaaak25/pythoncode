# %%
# LUTと観測データの吸収量を確認できるプログラム
import matplotlib.pylab as plt
import numpy as np
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

"""
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname_obs = '/Users/nyonn/IDLWorkspace/Default/savfile/Table_SP_obs_calc_orb0931_3.sav'
sav_data_obs = readsav(sav_fname_obs)

sav_fname_lut = '/Users/nyonn/IDLWorkspace/Default/savfile/Table_SP_Trans_calc.sav'
sav_data_lut = readsav(sav_fname_lut)
print(sav_data_lut.keys())

pressure15 = sav_data_lut['table_equivalent_pressure15']
pressure13 = sav_data_lut['table_equivalent_pressure13']
pressure1 = sav_data_lut['table_equivalent_pressure1']
observation = sav_data_obs['obs_spec']

"""
# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/Desktop/pythoncode/work file/work3_ORB0920_3.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())

lati = sav_data['lati']
longi = sav_data['longi']
ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))

# ORB0030_1
# ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))
# ORB0920_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))
# ORB0931_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 272) & (longi < 277))

lati = lati[ind]
longi = longi[ind]

pressure3 = sav_data['pressure']
pressure3 = np.exp(pressure3)

pressure3[pressure3 <= 0] = np.nan
pressure3 = pressure3[ind]

fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='ORB0313_4')
im = ax.scatter(longi, lati, c=pressure3, s=2)
fig.colorbar(im, orientation='horizontal')

# もっと広い範囲のスペクトルをみたい！
# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0313_4.sav'
sav_data = readsav(sav_fname)
wvl = sav_data['wvl']

specmars = np.loadtxt(
    '/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat')
dmars = sav_data['dmars']
specmars = specmars/dmars/dmars

CO2 = np.where((wvl > 1.0) & (wvl < 2.6))
specmars = specmars[CO2]
wvl = wvl[CO2]

jdat = sav_data['jdat']

nwvl = len(wvl)
io = len(jdat[1, 1, :])
ip = len(jdat[:, 1, 1])
flux = np.zeros((io, nwvl, ip))

for i in range(io):
    for o in range(ip):
        flux[i, :, o] = jdat[o, CO2, i]/specmars

radiance = jdat[0, CO2, 0]
radiance[radiance <= 0.01] = np.nan
radiance[radiance >= 100] = np.nan

flux[flux <= 0.0001] = np.nan
flux[flux >= 10] = np.nan

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.plot(wvl, flux[0, :, 0], color='red', zorder=2)
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
ax.set_ylabel('Radiance', fontsize=14)
# %%
