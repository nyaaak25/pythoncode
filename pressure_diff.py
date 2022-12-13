# %%
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/pressuremap_ORB0931_1.sav'
sav_data1 = readsav(sav_fname1)

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname2 = '/Users/nyonn/Desktop/pythoncode/pressuremap_ORB0920_3.sav'
sav_data2 = readsav(sav_fname2)

lati1 = sav_data1['lati']
longi1 = sav_data1['longi']
pressure1_b = sav_data1['pressure']

lati2 = sav_data2['lati']
longi2 = sav_data2['longi']
pressure2_b = sav_data2['pressure']

pressure1 = np.zeros((596, 128))
pressure1 = pressure1_b + 0

pressure2 = np.zeros((596, 128))
pressure2 = pressure2_b + 0

ind1 = np.where((lati1 > 54.1) & (lati1 < 56) &
                (longi1 > 274) & (longi1 < 276.9))
ind2 = np.where((lati2 > 54.1) & (lati2 < 56) &
                (longi2 > 274) & (longi2 < 276.9))

pressure_1 = pressure1[ind1]
pressure_2 = pressure2[ind2]

longi_1 = longi1[ind1]
lati_1 = lati1[ind1]

longi_2 = longi2[ind2]
lati_2 = lati2[ind2]

# %%
longi_dif2 = longi_2[0:3520]
lati_dif2 = lati_2[0:3520]

pressure_dif = pressure_1[0:3520] - pressure_2[0:3520]
#pressure_dif[pressure_dif <= -100] = np.nan
#pressure_dif[pressure_dif >= 100] = np.nan

longi_dif = longi_1[0:3520]
lati_dif = lati_1[0:3520]

longi_dif2 = longi_2[0:3520]
lati_dif2 = lati_2[0:3520]

# %%
fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='Pressure diffrence')
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi_dif, lati_dif, c=pressure_dif, s=2, cmap='jet')
fig.colorbar(im, orientation='horizontal')

# %%
lati_diff = lati_dif - lati_dif2
longi_diff = longi_dif - longi_dif2
plt.plot(lati_diff)

# %%
# interpol

x = longi_1
y = lati_1
z = pressure_1

f = interpolate.interp2d(x, y, z, kind='linear')

xnew = np.arange(54.1, 56, 0.001)
ynew = np.arange(274, 276.9, 0.001)
xnew_, ynew_ = np.meshgrid(xnew, ynew)
znew = f(xnew, ynew)

x2 = longi_2
y2 = lati_2
z2 = pressure_2

f2 = interpolate.interp2d(x2, y2, z2, kind='linear')

xnew = np.arange(54.1, 56, 0.001)
ynew = np.arange(274, 276.9, 0.001)
z2new = f2(xnew, ynew)

# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/pressuremap_ORB0920_3_albedo'
sav_data1 = readsav(sav_fname1)
print(sav_data1.keys())

lati1 = sav_data1['lati']
longi1 = sav_data1['longi']
albedo1 = sav_data1['altitude']

pressure1 = np.zeros((596, 128))
pressure1 = albedo1 + 0

pressure1[pressure1 == 0] = np.nan

fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='MOLA map orb0931_3')
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi1, lati1, c=pressure1, s=2, cmap='jet')
fig.colorbar(im, orientation='horizontal')
