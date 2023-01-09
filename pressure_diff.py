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
# vmin, vmaxでcolorbarの範囲を設定することができる
im = ax.scatter(longi1, lati1, c=pressure1, s=2, cmap='jet')
fig.colorbar(im, orientation='horizontal')


# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/SPmap_ORB0920_3_B1.sav'
sav_data1 = readsav(sav_fname1)
print(sav_data1.keys())

lati1 = sav_data1['lati']
longi1 = sav_data1['longi']
albedo1 = sav_data1['dustmap']

ip = len(lati1[:, 0])
io = len(lati1[0, :])

pressure1 = np.zeros((ip, io))
pressure1 = albedo1 + 0

pressure1[pressure1 == 0] = np.nan


fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='dust map orb0920_3')
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi1, lati1, c=pressure1, s=2,
                cmap='jet', vmin=0.07, vmax=0.082)
fig.colorbar(im, orientation='horizontal')

# %%
# 複数の図をプリントする
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/sav file/mcddata_ORB7750_2.sav'
sav_data1 = readsav(sav_fname1)
print(sav_data1.keys())

lati = sav_data1['lati']
longi = sav_data1['longi']
albedo = sav_data1['albedo']
dust = sav_data1['dustmap']
altitude = sav_data1['altitude']
ta = sav_data1['tamap']
SP = sav_data1['mcdpressure']

ip = len(lati[:, 0])
io = len(lati[0, :])

albedo1 = np.zeros((ip, io))
albedo1 = albedo + 0
albedo1[albedo1 == 0] = np.nan

dust1 = np.zeros((ip, io))
dust1 = dust + 0
dust1[dust1 == 0] = np.nan

altitude1 = np.zeros((ip, io))
altitude1 = altitude + 0
altitude1[altitude1 == 0] = np.nan

ta1 = np.zeros((ip, io))
ta1 = ta + 0
ta1[ta1 == 0] = np.nan

SP1 = np.zeros((ip, io))
SP1 = SP + 0
SP1[SP1 == 0] = np.nan

fig = plt.figure(figsize=(15, 7), dpi=400)
ax1 = fig.add_subplot(151, title='Albedo')
ax2 = fig.add_subplot(152, title='Dust')
ax3 = fig.add_subplot(153, title='MOLA altitude')
ax4 = fig.add_subplot(154, title='Temperature')
ax5 = fig.add_subplot(155, title='Surface pressure')

im1 = ax1.scatter(longi, lati, c=albedo1, s=2, cmap='jet')
fig.colorbar(im1, ax=ax1, pad=0.08, shrink=0.9,
             aspect=50, orientation='horizontal')

im2 = ax2.scatter(longi, lati, c=dust1, s=2, cmap='jet')
fig.colorbar(im2, ax=ax2, pad=0.08, shrink=0.9,
             aspect=50, orientation='horizontal')

im3 = ax3.scatter(longi, lati, c=altitude1, s=2, cmap='jet')
fig.colorbar(im3, ax=ax3, pad=0.08, shrink=0.9,
             aspect=50, orientation='horizontal')

im4 = ax4.scatter(longi, lati, c=ta1, s=2, cmap='jet')
fig.colorbar(im4, ax=ax4, pad=0.08, shrink=0.9,
             aspect=50, orientation='horizontal')

im5 = ax5.scatter(longi, lati, c=SP1, s=2, cmap='jet')
fig.colorbar(im5, ax=ax5, pad=0.08, shrink=0.9,
             aspect=50, orientation='horizontal')

# %%
# 3つのプロットを並べて表示
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0931_3.sav'
sav_data = readsav(sav_fname)

specmars = np.loadtxt(
    '/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat')
dmars = sav_data['dmars']
specmars = specmars/dmars/dmars
jdat = sav_data['jdat']
wvl = sav_data['wvl']

wvl1 = wvl[256:352]
specmars1 = specmars[256:352]

# 0:127がSWIR
# 128:255がLWIR
# 256:352がVNIR

nwvl1 = len(wvl1)
io1 = len(jdat[1, 1, :])
ip1 = len(jdat[:, 1, 1])

jdat = sav_data['jdat']
flux1 = np.zeros((io1, nwvl1, ip1))

for i in range(io1):
    for o in range(ip1):
        flux1[i, :, o] = jdat[o, 256:352, i]/specmars1

flux1[flux1 <= 0.0001] = np.nan
flux1[flux1 >= 100] = np.nan


wvl2 = wvl[0:127]
specmars2 = specmars[0:127]

# 0:127がSWIR
# 128:255がLWIR
# 256:352がVNIR

nwvl2 = len(wvl2)
jdat = sav_data['jdat']
flux2 = np.zeros((io, nwvl2, ip))

for i in range(io):
    for o in range(ip):
        flux2[i, :, o] = jdat[o, 0:127, i]/specmars2

flux2[flux2 <= 0.0001] = np.nan
flux2[flux2 >= 100] = np.nan

wvl3 = wvl[128:255]
specmars3 = specmars[128:255]

# 0:127がSWIR
# 128:255がLWIR
# 256:352がVNIR

nwvl3 = len(wvl3)
jdat = sav_data['jdat']
flux3 = np.zeros((io, nwvl3, ip))

for i in range(io):
    for o in range(ip):
        flux3[i, :, o] = jdat[o, 128:255, i]/specmars3

flux3[flux3 <= 0.0001] = np.nan
flux3[flux3 >= 100] = np.nan


fig = plt.figure(figsize=(5, 18), dpi=200)
ax1 = fig.add_subplot(311, title='OMEGA VIS channel')
ax1.grid(c='lightgray', zorder=1)
ax1.set_xlabel('Wavenumber [μm]', fontsize=10)
ax1.set_ylabel('I/F', fontsize=10)
ax1.plot(wvl1, flux1[0, :, 10], label="OMEGA raw data", lw=1.5)
h1, l1 = ax1.get_legend_handles_labels()

ax2 = fig.add_subplot(312, title='OMEGA SWIR channel')
ax2.grid(c='lightgray', zorder=1)
ax2.set_xlabel('Wavenumber [μm]', fontsize=10)
ax2.set_ylabel('I/F', fontsize=10)
ax2.plot(wvl2, flux2[0, :, 10], label="OMEGA raw data", lw=1.5)
h1, l1 = ax2.get_legend_handles_labels()

ax3 = fig.add_subplot(313, title='OMEGA LWIR channel')
ax3.grid(c='lightgray', zorder=1)
ax3.set_xlabel('Wavenumber [μm]', fontsize=10)
ax3.set_ylabel('I/F', fontsize=10)
ax3.plot(wvl3, flux3[0, :, 10], label="OMEGA raw data", lw=1.5)
h1, l1 = ax3.get_legend_handles_labels()

# %%
# 高度補正
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/SPmap_ORB0313_4.sav'
sav_data1 = readsav(sav_fname1)

lati1 = sav_data1['lati']
longi1 = sav_data1['longi']
pressure = sav_data1['pressure']
tamap = sav_data1['tamap']
MOLA = sav_data1['altitude']

ip = len(lati1[:, 0])
io = len(lati1[0, :])

pressure1 = np.zeros((ip, io))
pressure1 = pressure + 0
pressure1[pressure1 == 0] = np.nan

tamap1 = np.zeros((ip, io))
tamap1 = tamap + 0
tamap1[tamap1 == 0] = np.nan

MOLA1 = np.zeros((ip, io))
MOLA1 = MOLA + 0
MOLA1[MOLA1 == 0] = np.nan

# 高度補正を行う
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname2 = '/Users/nyonn/Desktop/pythoncode/ORB0313_4.sav'
sav_data2 = readsav(sav_fname2)

alt1 = sav_data2['alt']
R = 192
g = 3.72
Gconst = 6.67430e-11
MMars = 6.42e23
RMars = 3.4e6
# g = -Gconst*MMars/(-RMars*RMars)

# pre_sev = pressure1 * np.exp((alt1*1e3)/((R*tamap1)/g))
pre_sev = pressure1 * \
    np.exp((alt1*1e3)/((R*tamap1)/(-Gconst * MMars /
           (-1*(RMars+alt1*1e3)*(RMars+alt1*1e3)))))


fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='SP map orb0313_4')
ax.set_ylim(36.7, 41)
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi1, lati1, c=MOLA1, s=2,
                cmap='jet')
fig.colorbar(im, orientation='horizontal')
