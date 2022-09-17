# %%
# LUTと観測データの吸収量を確認できるプログラム
from numba import jit, f8
import time
from memory_profiler import profile
import pandas as pd
import matplotlib.pyplot as plt
from pyrsistent import v
import matplotlib.pylab as plt
import numpy as np
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

# %%
# OMEGA dataとARS dataのfittingをみる

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0931_3.sav'
sav_data = readsav(sav_fname)

wvl = sav_data['wvl']
CO2 = np.where((wvl > 1.81) & (wvl < 2.19))
wvl = wvl[CO2]

jdat = sav_data['jdat']

flux = jdat[0, CO2, 0]
flux[flux <= 0.01] = np.nan
flux[flux >= 100] = np.nan

x = [wvl[0], wvl[1], wvl[2], wvl[23], wvl[24], wvl[25]]
y = [flux[:, 0], flux[:, 1], flux[:, 2],
     flux[:, 23], flux[:, 24], flux[:, 25]]
c, d = np.polyfit(x, y, 1)
cont = d + c*wvl

OMEGA_calc = flux/cont

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)

Pa_list = ['Pa=50', 'Pa=150', 'Pa=180', 'Pa=215', 'Pa=257', 'Pa=308', 'Pa=369',
           'Pa=442', 'Pa=529', 'Pa=633', 'Pa=758', 'Pa=907', 'Pa=1096', 'Pa=1300', 'Pa=1500']
TA_list = ['TA=135', 'TA=160', 'TA=213', 'TA=260', 'TA=285']
TB_list = ['TB=80', 'TB=146', 'TB=200']
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60', 'SZA=75']
EA_list = ['EA=0', 'EA=5', 'EA=10']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180']
Dust_list = ['Dust=0', 'Dust=0.3', 'Dust=0.6',
             'Dust=0.9', 'Dust=1.2', 'Dust=1.5']
WaterI_list = ['WaterI=0', 'WaterI=0.5', 'WaterI=1.0']
Albedo_list = ['Albedo=0.05', 'Albedo=0.1', 'ALbedo=0.2',
               'Albedo=0.3', 'Albedo=0.4', 'Albedo=0.5']

for i in range(1, 15, 1):
    ARS = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                     str(i)+'_TA3_TB2_SZA4_EA2_PA2_Dust1_WaterI1_SurfaceA3_rad.dat')
    ARS_x = ARS[:, 0]
    ARS_x = (1/ARS_x)*10000
    ARS_x = ARS_x[::-1]
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]

    ARS1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                      str(i+1)+'_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
    ARS_x1 = ARS1[:, 0]
    ARS_x1 = (1/ARS_x)*10000
    ARS_x1 = ARS_x1[::-1]
    ARS_y1 = ARS1[:, 1]
    ARS_y1 = ARS_y1[::-1]

    POLY_x = [ARS_x[0], ARS_x[3], ARS_x[5], ARS_x[23], ARS_x[24], ARS_x[25]]
    POLY_y = [ARS_y[0], ARS_y[3], ARS_y[5], ARS_y[23], ARS_y[24], ARS_y[25]]
    a, b = np.polyfit(POLY_x, POLY_y, 1)

    cont0 = b + a*ARS_x
    y_calc = ARS_y/cont0

    ax.plot(ARS_x, y_calc, label=Pa_list[i-1], zorder=i, lw=0.5)

ax.plot(wvl, OMEGA_calc[0, :], label="OMEGA raw data", lw=1.5, color='red')
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)


# %%
# リトリーバルテスト用
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('Diference', fontsize=10)

Pa_list = ['Pa=50', 'Pa=150', 'Pa=180', 'Pa=215', 'Pa=257', 'Pa=308', 'Pa=369',
           'Pa=442', 'Pa=529', 'Pa=633', 'Pa=758', 'Pa=907', 'Pa=1096', 'Pa=1300', 'Pa=1500']
TA_list = ['TA=135', 'TA=160', 'TA=213', 'TA=260', 'TA=285']
TB_list = ['TB=80', 'TB=146', 'TB=200']
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60', 'SZA=75']
EA_list = ['EA=0', 'EA=5', 'EA=10']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180']
Dust_list = ['Dust=0', 'Dust=0.3', 'Dust=0.6',
             'Dust=0.9', 'Dust=1.2', 'Dust=1.5']
WaterI_list = ['WaterI=0', 'WaterI=0.5', 'WaterI=1.0']
Albedo_list = ['Albedo=0.05', 'Albedo=0.1', 'ALbedo=0.2',
               'Albedo=0.3', 'Albedo=0.4', 'Albedo=0.5']

for i in range(1, 15, 1):
    ARS = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                     str(i)+'_TA3_TB2_SZA4_EA2_PA2_Dust1_WaterI1_SurfaceA3_rad.dat')
    ARS_x = ARS[:, 0]
    ARS_x = (1/ARS_x)*10000
    ARS_x = ARS_x[::-1]
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]

    ARS1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                      str(i+1)+'_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
    ARS_x1 = ARS1[:, 0]
    ARS_x1 = (1/ARS_x)*10000
    ARS_x1 = ARS_x1[::-1]
    ARS_y1 = ARS1[:, 1]
    ARS_y1 = ARS_y1[::-1]

    POLY_x = [ARS_x[0], ARS_x[3], ARS_x[5], ARS_x[23], ARS_x[24], ARS_x[25]]
    POLY_y = [ARS_y[0], ARS_y[3], ARS_y[5], ARS_y[23], ARS_y[24], ARS_y[25]]
    a, b = np.polyfit(POLY_x, POLY_y, 1)

    cont0 = b + a*ARS_x
    y_calc = ARS_y/cont0

    ax.plot(ARS_x, y_calc, label=Pa_list[i-1], zorder=i)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)


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
# インポート

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0931_3.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())
wvl = sav_data['wvl']

specmars = np.loadtxt(
    '/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat')
dmars = sav_data['dmars']
specmars = specmars/dmars/dmars

CO2 = np.where((wvl > 1.81) & (wvl < 2.19))
wvl = wvl[CO2]
specmars = specmars[CO2]

jdat = sav_data['jdat']
nwvl = len(wvl)
io = len(jdat[1, 1, :])
ip = len(jdat[:, 1, 1])
flux = np.zeros((io, nwvl, ip))

for i in range(io):
    for o in range(ip):
        flux[i, :, o] = jdat[o, CO2, i]/specmars


fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('Diference', fontsize=10)

Pa_list = ['Pa=50', 'Pa=150', 'Pa=180', 'Pa=215', 'Pa=257', 'Pa=308', 'Pa=369',
           'Pa=442', 'Pa=529', 'Pa=633', 'Pa=758', 'Pa=907', 'Pa=1096', 'Pa=1300', 'Pa=1500']
TA_list = ['TA=135', 'TA=160', 'TA=213', 'TA=260', 'TA=285']
TB_list = ['TB=80', 'TB=146', 'TB=200']
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60', 'SZA=75']
EA_list = ['EA=0', 'EA=5', 'EA=10']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180']
Dust_list = ['Dust=0', 'Dust=0.3', 'Dust=0.6',
             'Dust=0.9', 'Dust=1.2', 'Dust=1.5']
WaterI_list = ['WaterI=0', 'WaterI=0.5', 'WaterI=1.0']
Albedo_list = ['Albedo=0.05', 'Albedo=0.1', 'ALbedo=0.2',
               'Albedo=0.3', 'Albedo=0.4', 'Albedo=0.5']

for i in range(1, 15, 1):
    ARS = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                     str(i)+'_TA3_TB2_SZA4_EA2_PA2_Dust1_WaterI1_SurfaceA3_rad.dat')
    ARS_x = ARS[:, 0]
    ARS_x = (1/ARS_x)*10000
    ARS_x = ARS_x[::-1]
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]

    ARS1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                      str(i+1)+'_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
    ARS_x1 = ARS1[:, 0]
    ARS_x1 = (1/ARS_x)*10000
    ARS_x1 = ARS_x1[::-1]
    ARS_y1 = ARS1[:, 1]
    ARS_y1 = ARS_y1[::-1]

    error1 = (ARS_y1 - ARS_y) * 100 / ARS_y

    ax.plot(ARS_x, ARS_y, label=Pa_list[i]+"-"+Pa_list[i-1])
    #ax.set_xlim(1.81, 1.825)
    # ax.set_ylim(-1e-15, 1e-15)
ax.plot(wvl, flux[0, :], label="OMEGA data")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)


# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/Desktop/pythoncode/work file/work3_ORB0920_3.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())

lati = sav_data['lati']
longi = sav_data['longi']
ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))

# ORB0030_1
# ind = np.where((lati > -50) & (lati < -47) & (longi > 60) & (longi < 62)
# ORB0920_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))
# ORB0931_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 272) & (longi < 277))
# ORB0313_4
ind = np.where((lati > 36) & (lati < 41) & (longi > 95) & (longi < 98))

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
# アルベドのバイアス計算
P_txt = np.loadtxt('albedo_idl.txt')
albedo_idl = P_txt[:, 3]

T_txt = np.loadtxt('loop2_albedo.txt')
albedo_loop = T_txt[:, 2]

dif_albedo = albedo_idl - albedo_loop

# P_txt[:,2] これはlatitude側に回している (j)
# P_txt[:,1] これはlongitude側に回している (i)
plt.scatter(P_txt[:, 2], dif_albedo)

# %%
#  波長が格納されているところが間違っていないかの確認
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/Desktop//Table_calc_wave2.sav'
sav_data = readsav(sav_fname)

pressure13 = sav_data['table_equivalent_pressure15']
print(pressure13[3, 0, 0, 0, 3, 2, 1, 2])

ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP15_TA3_TB2_SZA3_EA4_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = (1/ARS_x)*10000
ARS_x = ARS_x[::-1]
ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]

POLY_x = [ARS_x[0], ARS_x[1], ARS_x[2], ARS_x[23], ARS_x[24], ARS_x[25]]
POLY_y = [ARS_y[0], ARS_y[1], ARS_y[2], ARS_y[23], ARS_y[24], ARS_y[25]]
a, b = np.polyfit(POLY_x, POLY_y, 1)

cont0 = b + a*ARS_x
y_calc = ARS_y/cont0
y_calc[16] = np.nan

print(1-y_calc)

# %%
# Forgetの1036 Paと比較
# OMEGA dataとARS dataのfittingをみる

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0030_1.sav'
sav_data = readsav(sav_fname)

wvl = sav_data['wvl']
CO2 = np.where((wvl > 1.81) & (wvl < 2.19))
wvl = wvl[CO2]

jdat = sav_data['jdat']

flux = jdat[1000, CO2, 15]
flux[flux <= 0.01] = np.nan
flux[flux >= 100] = np.nan

x = [wvl[0], wvl[1], wvl[2], wvl[23], wvl[24], wvl[25]]
y = [flux[:, 0], flux[:, 1], flux[:, 2],
     flux[:, 23], flux[:, 24], flux[:, 25]]
c, d = np.polyfit(x, y, 1)
cont = d + c*wvl

OMEGA_calc = flux/cont
OMEGA_calc = 1 - flux/cont
OMEGA_calc = OMEGA_calc[0, :]
spec = np.nansum(OMEGA_calc[band])

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)

Pa_list = ['Pa=50', 'Pa=150', 'Pa=180', 'Pa=215', 'Pa=257', 'Pa=308', 'Pa=369',
           'Pa=442', 'Pa=529', 'Pa=633', 'Pa=758', 'Pa=907', 'Pa=1096', 'Pa=1300', 'Pa=1500']
TA_list = ['TA=135', 'TA=160', 'TA=213', 'TA=260', 'TA=285']
TB_list = ['TB=80', 'TB=146', 'TB=200']
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60', 'SZA=75']
EA_list = ['EA=0', 'EA=5', 'EA=10']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180']
Dust_list = ['Dust=0', 'Dust=0.3', 'Dust=0.6',
             'Dust=0.9', 'Dust=1.2', 'Dust=1.5']
WaterI_list = ['WaterI=0', 'WaterI=0.5', 'WaterI=1.0']
Albedo_list = ['Albedo=0.05', 'Albedo=0.1', 'ALbedo=0.2',
               'Albedo=0.3', 'Albedo=0.4', 'Albedo=0.5']

ARS = np.loadtxt(
    '/Users/nyonn/Desktop/SP15_TA3_TB2_SZA3_EA4_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = (1/ARS_x)*10000
ARS_x = ARS_x[::-1]
ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]

POLY_x = [ARS_x[0], ARS_x[1], ARS_x[2], ARS_x[23], ARS_x[24], ARS_x[25]]
POLY_y = [ARS_y[0], ARS_y[1], ARS_y[2], ARS_y[23], ARS_y[24], ARS_y[25]]
a, b = np.polyfit(POLY_x, POLY_y, 1)

cont0 = b + a*ARS_x
y_calc = ARS_y/cont0
y_calc[16] = np.nan
y_calc = 1 - ARS_y/cont0
obs_spec = np.nansum(y_calc[band])

ARS1 = np.loadtxt(
    '/Users/nyonn/Desktop/SP13_TA3_TB2_SZA3_EA4_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x1 = ARS1[:, 0]
ARS_x1 = (1/ARS_x)*10000
ARS_x1 = ARS_x1[::-1]
ARS_y1 = ARS1[:, 1]
ARS_y1 = ARS_y1[::-1]

POLY_x1 = [ARS_x1[0], ARS_x1[1], ARS_x1[2], ARS_x1[23], ARS_x1[24], ARS_x1[25]]
POLY_y1 = [ARS_y1[0], ARS_y1[1], ARS_y1[2], ARS_y1[23], ARS_y1[24], ARS_y1[25]]
c, d = np.polyfit(POLY_x1, POLY_y1, 1)

cont01 = d + c*ARS_x1
y_calc1 = ARS_y1/cont01
y_calc1[16] = np.nan
y_calc1 = 1 - ARS_y1/cont01
obs_spec1 = np.nansum(y_calc1[band])

ax.scatter(ARS_x, y_calc, label="retrival result", s=8)
ax.scatter(ARS_x, y_calc1, label="Forget result", s=8)
ax.scatter(wvl, OMEGA_calc[0, :], label="OMEGA raw data", s=13)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)

print("OMEGA raw data：", spec)
print("Retrieval Result：", obs_spec)
print("Forget Result：", obs_spec1)
print("Forget - Retrieval：", abs(obs_spec1 - obs_spec))
print("Forget - OMEGA：", abs(obs_spec1 - spec))
print("Retrieval - OMEGA：", abs(obs_spec - spec))

# %%
# 上のprogramのupdate
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0363_3.sav'
sav_data = readsav(sav_fname)

wvl = sav_data['wvl']
CO2 = np.where((wvl > 1.81) & (wvl < 2.19))
wvl = wvl[CO2]

band = np.where((wvl > 1.93) & (wvl < 2.04))

jdat = sav_data['jdat']

xind = 62
yind = 594

flux = jdat[yind, CO2, xind]
flux[flux <= 0.01] = np.nan
flux[flux >= 100] = np.nan

x = [wvl[0], wvl[1], wvl[2], wvl[23], wvl[24], wvl[25]]
y = [flux[:, 0], flux[:, 1], flux[:, 2],
     flux[:, 23], flux[:, 24], flux[:, 25]]
c, d = np.polyfit(x, y, 1)
cont = d + c*wvl

OMEGA_calc = flux/cont
OMEGA_calc = OMEGA_calc[0, :]
OMEGA_calc_rate = 1 - OMEGA_calc
spec = np.nansum(OMEGA_calc_rate[band])

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)

ARS = np.loadtxt(
    '/Users/nyonn/Desktop//pythoncode/ARS_calc/SP14_TA3_TB2_SZA2_EA1_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = (1/ARS_x)*10000
ARS_x = ARS_x[::-1]
ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]

POLY_x = [ARS_x[0], ARS_x[1], ARS_x[2], ARS_x[23], ARS_x[24], ARS_x[25]]
POLY_y = [ARS_y[0], ARS_y[1], ARS_y[2], ARS_y[23], ARS_y[24], ARS_y[25]]
a, b = np.polyfit(POLY_x, POLY_y, 1)

cont0 = b + a*ARS_x
y_calc = ARS_y/cont0
y_calc[16] = np.nan
y_calc = ARS_y/cont0
y_calc_rate = 1 - y_calc
obs_spec = np.nansum(y_calc_rate[band])

ARS1 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP12_TA3_TB2_SZA2_EA1_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x1 = ARS1[:, 0]
ARS_x1 = (1/ARS_x)*10000
ARS_x1 = ARS_x1[::-1]
ARS_y1 = ARS1[:, 1]
ARS_y1 = ARS_y1[::-1]

POLY_x1 = [ARS_x1[0], ARS_x1[1], ARS_x1[2], ARS_x1[23], ARS_x1[24], ARS_x1[25]]
POLY_y1 = [ARS_y1[0], ARS_y1[1], ARS_y1[2], ARS_y1[23], ARS_y1[24], ARS_y1[25]]
c, d = np.polyfit(POLY_x1, POLY_y1, 1)

cont01 = d + c*ARS_x1
y_calc1 = ARS_y1/cont01
y_calc1[16] = np.nan
y_calc1 = ARS_y1/cont01
y_calc_rate1 = 1 - y_calc1
obs_spec1 = np.nansum(y_calc_rate1[band])

ax.plot(ARS_x, y_calc, label="retrival result")
ax.plot(ARS_x, y_calc1, label="Forget result")
ax.scatter(wvl, OMEGA_calc, label="OMEGA raw data", s=13)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)

print("OMEGA raw data：", spec)
print("Retrieval Result：", obs_spec)
print("Forget Result：", obs_spec1)
print("Forget - Retrieval：", abs(obs_spec1 - obs_spec))
print("Forget - OMEGA：", abs(obs_spec1 - spec))
print("Retrieval - OMEGA：", abs(obs_spec - spec))
