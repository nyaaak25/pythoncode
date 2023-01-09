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

# %%
# 可視化マップ作成のためのプログラム
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/Table_SP_calc_ver4_add_albedo.sav'
sav_data = readsav(sav_fname)

pressure1 = sav_data['table_equivalent_pressure1']
pressure2 = sav_data['table_equivalent_pressure2']
pressure3 = sav_data['table_equivalent_pressure3']
pressure4 = sav_data['table_equivalent_pressure4']
pressure5 = sav_data['table_equivalent_pressure5']
pressure6 = sav_data['table_equivalent_pressure6']
pressure7 = sav_data['table_equivalent_pressure7']
pressure8 = sav_data['table_equivalent_pressure8']
pressure9 = sav_data['table_equivalent_pressure9']
pressure10 = sav_data['table_equivalent_pressure10']
pressure11 = sav_data['table_equivalent_pressure11']
pressure12 = sav_data['table_equivalent_pressure12']
pressure13 = sav_data['table_equivalent_pressure13']
pressure14 = sav_data['table_equivalent_pressure14']
pressure15 = sav_data['table_equivalent_pressure15']

pressure1 = pressure1[1, 0, :, 0, 0, 0, 0, 1]
pressure2 = pressure2[1, 0, :, 0, 0, 0, 0, 1]
pressure3 = pressure3[1, 0, :, 0, 0, 0, 0, 1]
pressure4 = pressure4[1, 0, :, 0, 0, 0, 0, 1]
pressure5 = pressure5[1, 0, :, 0, 0, 0, 0, 1]
pressure6 = pressure6[1, 0, :, 0, 0, 0, 0, 1]
pressure7 = pressure7[1, 0, :, 0, 0, 0, 0, 1]
pressure8 = pressure8[1, 0, :, 0, 0, 0, 0, 1]
pressure9 = pressure9[1, 0, :, 0, 0, 0, 0, 1]
pressure10 = pressure10[1, 0, :, 0, 0, 0, 0, 1]
pressure11 = pressure11[1, 0, :, 0, 0, 0, 0, 1]
pressure12 = pressure12[1, 0, :, 0, 0, 0, 0, 1]
pressure13 = pressure13[1, 0, :, 0, 0, 0, 0, 1]
pressure14 = pressure14[1, 0, :, 0, 0, 0, 0, 1]
pressure15 = pressure15[1, 0, :, 0, 0, 0, 0, 1]

Pa_list = [50, 150, 180, 215, 257, 308, 369,
           442, 529, 633, 758, 907, 1096, 1300, 1500]

Pa = np.repeat(Pa_list, 6)
Dust = [0, 0.3, 0.6, 0.9, 1.2, 1.5]*15

Pa_array = [pressure1, pressure2, pressure3, pressure4, pressure5, pressure6, pressure7,
            pressure8, pressure9, pressure10, pressure11, pressure12, pressure13, pressure14, pressure15]
pressure_array = np.ravel(Pa_array)

Dust_pressure = np.array([Pa, Dust, pressure_array])

fig = plt.figure(figsize=(2, 7), dpi=200)
ax = fig.add_subplot(111, title='Dust - pressure')
im = ax.scatter(Pa, Dust, c=pressure_array, s=30)
fig.colorbar(im, orientation='horizontal')
ax.set_xlabel('Pressure [Pa]')
ax.set_ylabel('Dust Opacity')

# %%
# OMEGAのスペクトル

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0931_3.sav'
sav_data = readsav(sav_fname)

specmars = np.loadtxt(
    '/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat')
dmars = sav_data['dmars']
specmars = specmars/dmars/dmars

jdat = sav_data['jdat']

wvl = sav_data['wvl']
wvl = wvl[0:127]
specmars = specmars[0:127]

# 0:127がSWIR
# 128:255がLWIR
# 256:352がVNIR

nwvl = len(wvl)
io = len(jdat[1, 1, :])
ip = len(jdat[:, 1, 1])

jdat = sav_data['jdat']
flux = np.zeros((io, nwvl, ip))

for i in range(io):
    for o in range(ip):
        flux[i, :, o] = jdat[o, 0:127, i]/specmars

flux[flux <= 0.01] = np.nan
flux[flux >= 100] = np.nan

fig = plt.figure(figsize=(4, 2), dpi=200)
ax = fig.add_subplot(111, title='OMEGA SWIR channel')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('I/F', fontsize=10)
ax.plot(wvl, flux[0, :, 10], label="OMEGA raw data", lw=1.5)
h1, l1 = ax.get_legend_handles_labels()
print(wvl)

# %%
# ガウス-ニュートン法の実装まで python version
# モジュールインポート

# 前提条件の入力
T = np.array([64.51, 66.14, 67.83, 69.58, 71.29, 73.16,
             75.36, 77.9, 81.48, 84.01, 87.53, 92.39, 100.])
P = np.array([101.325, 107.9911184, 115.283852, 123.2431974, 131.4691875, 140.9750724,
             152.8807599, 167.6395461, 190.4243388, 208.0095592, 234.6607007, 276.0039671, 352.4910099])

# Antoine式


def theoreticalValue(beta):
    Pcal = 10**(beta[0]+beta[1]/(T+beta[2]))
    return Pcal

# 残差


def objectiveFunction(beta):
    r = P - theoreticalValue(beta)
    return r

# Gauss-Newton法


def gaussNewton(function, beta, tolerance, epsilon):
    delta = 2*tolerance
    alpha = 1
    while np.linalg.norm(delta) > tolerance:
        F = function(beta)
        J = np.zeros((len(F), len(beta)))  # 有限差分ヤコビアン
        for jj in range(0, len(beta)):
            dBeta = np.zeros(beta.shape)
            dBeta[jj] = epsilon
            J[:, jj] = (function(beta+dBeta)-F)/epsilon
        delta = -np.linalg.pinv(J).dot(F)
        print("before:", delta)
        abcde = -np.linalg.pinv(J)
        efg = abcde.dot(F)
        print("after:", efg)  # 探索方向
        beta = beta + alpha*delta
    return beta


# Gauss-Newton法の実行
#initialValue = np.array([1,1,1])
#initialValue = np.array([10,-100,100])
initialValue = np.array([10, -1500, 200])
betaID = gaussNewton(objectiveFunction, initialValue, 1e-4, 1e-4)

# グラフの出力
# print(betaID)
plt.figure()
plt.plot(T, P, 'o')
plt.plot(T, theoreticalValue(betaID), '-')
plt.xlabel('T [℃]')
plt.ylabel('P [kPa]')
plt.legend(['Row data', 'Pcal'])
plt.show()

# %%
# スペクトルを表示させる

ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP15_TA3_TB2_SZA3_EA4_PA1_Dust1_WaterI1_SurfaceA4_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = ARS_x[::-1]
ARS_x = (1/ARS_x)  # cm-1 → cm
ARSx = ARS_x*1e4

ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]
ARS_y = (ARS_y/(ARS_x)**2)*1e-7


fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='ARS Calc Spectrum')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('Radiance', fontsize=10)
ax.plot(ARSx, ARS_y, label="retrival result")
# ax.axvline(x=ARS_x[9])
# ax.axvline(x=ARS_x[15])
print(ARS_x)

# %%
# EW法のband幅を表示させる用
ARS = np.loadtxt(
    '/Users/nyonn/Desktop/SP6_TA1_TB1_SZA1_EA1_PA1_Dust5_WaterI2_SurfaceA7_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = ARS_x[::-1]
ARS_x = (1/ARS_x)  # cm-1 → cm
ARSx = ARS_x*1e4

ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]
ARS_y = (ARS_y/(ARS_x)**2)*1e-7

ARS_y[8] = np.nan
ARS_y[17] = np.nan
ARS_y[24] = np.nan
ARS_y[27] = np.nan

POLY_x = [ARSx[0], ARSx[1], ARSx[2], ARSx[23], ARSx[25], ARSx[26]]
POLY_y = [ARS_y[0], ARS_y[1], ARS_y[2], ARS_y[23], ARS_y[25], ARS_y[26]]
a, b = np.polyfit(POLY_x, POLY_y, 1)

cont0 = b + a*ARSx
y_calc = ARS_y/cont0

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='Using spectrum')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavelength [μm]', fontsize=10)
ax.set_ylabel('Radiance', fontsize=10)
ax.scatter(ARSx, ARS_y, label="retrival result")
ax.scatter(POLY_x, POLY_y, label="retrival result", color='red')
ax.plot(ARSx, ARS_y, label="OMEGA raw data", lw=1.5, color='blue')
ax.plot(ARSx, cont0, label="OMEGA raw data", lw=1.5, color='red')

# 1.85-2.10um[band1]
ax.axvline(x=ARSx[4], color="green")
ax.axvline(x=ARSx[21], color="green")

# 1.94~2.09um [band2]
ax.axvline(x=ARSx[10], color="green")
ax.axvline(x=ARSx[20], color="green")

# 1.94~1.99um [band3]
ax.axvline(x=ARSx[10], color="green")
ax.axvline(x=ARSx[13], color="green")

# 1.94-2.04um [band4]
ax.axvline(x=ARSx[10], color="green")
ax.axvline(x=ARSx[17], color="green")

# %%
# 使用している波長をみてる

ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/SP5_TA1_TB1_SZA1_EA1_PA1_Dust5_WaterI2_SurfaceA7_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = ARS_x[::-1]
ARS_x = (1/ARS_x)  # cm-1 → cm
ARSx = ARS_x*1e4

ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]
ARS_y = (ARS_y/(ARS_x)**2)*1e-7


fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='ARS Calc Spectrum')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('Radiance', fontsize=10)
ax.plot(ARSx, ARS_y, label="retrival result")

# not use spectrum
ax.axvline(x=ARSx[8], color="red")
ax.axvline(x=ARSx[17], color="red")
ax.axvline(x=ARSx[24], color="red")
ax.axvline(x=ARSx[27], color="red")

ARSx[8] = np.nan
ARSx[17] = np.nan
ARSx[24] = np.nan
ARSx[27] = np.nan

# continuumを引く場所
ax.axvline(x=ARSx[0], color="green")
ax.axvline(x=ARSx[1], color="green")
ax.axvline(x=ARSx[2], color="green")
ax.axvline(x=ARSx[23], color="green")
ax.axvline(x=ARSx[25], color="green")
ax.axvline(x=ARSx[26], color="green")

print(ARSx[8], ARSx[17], ARSx[24], ARSx[27])

# %%
# 観測も一緒に表示
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0931_3.sav'
sav_data = readsav(sav_fname)

wvl = sav_data['wvl']
CO2 = np.where((wvl > 1.8) & (wvl < 2.2))
wvl = wvl[CO2]

jdat = sav_data['jdat']

flux = jdat[0, CO2, 0]
aaa = flux[0, :]
aaa[8] = np.nan
aaa[17] = np.nan
aaa[24] = np.nan
aaa[27] = np.nan


ARS = np.loadtxt(
    '/Users/nyonn/Desktop/SP6_TA1_TB1_SZA1_EA1_PA1_Dust5_WaterI2_SurfaceA7_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = ARS_x[::-1]
ARS_x = (1/ARS_x)  # cm-1 → cm
ARSx = ARS_x*1e4

ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]
ARS_y = (ARS_y/(ARS_x)**2)*1e-7

ARS_y[8] = np.nan
ARS_y[17] = np.nan
ARS_y[24] = np.nan
ARS_y[27] = np.nan

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='ARS Calc Spectrum')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('Radiance', fontsize=10)
ax.plot(ARSx, ARS_y, label="retrival result")
ax.plot(wvl, aaa, label="OMEGA raw data", lw=1.5, color='red')

# ax.axvline(x=ARSx[8],color="red")
# ax.axvline(x=ARSx[17],color="red")
# ax.axvline(x=ARSx[24],color="red")
# ax.axvline(x=ARSx[27],color="red")

ax.axvline(x=ARSx[0], color="green")
ax.axvline(x=ARSx[1], color="green")
ax.axvline(x=ARSx[2], color="green")
ax.axvline(x=ARSx[23], color="green")
ax.axvline(x=ARSx[25], color="green")
ax.axvline(x=ARSx[26], color="green")

print(ARSx[8], ARSx[17], ARSx[24], ARSx[27])

# %%
# 351のORBでの気圧変動を確認
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/Desktop/pythoncode/work file/work_CO2_ORB0351_3.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())

lati = sav_data['lati']
longi = sav_data['longi']
#ind = np.where((lati > 18) & (lati < 27) & (longi > 200) & (longi < 205))

# ORB0030_1
# ind = np.where((lati > -50) & (lati < -47) & (longi > 60) & (longi < 62)
# ORB0920_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 271) & (longi < 278))
# ORB0931_3
# ind = np.where((lati > 50) & (lati < 61) & (longi > 272) & (longi < 277))
# ORB0313_4
# ind = np.where((lati > 36) & (lati < 41) & (longi > 95) & (longi < 98))

#lati = lati[ind]
#longi = longi[ind]

pressure3 = sav_data['pressure']
pressure3 = np.exp(pressure3)

pressure3[pressure3 <= 0] = np.nan
#pressure3 = pressure3[ind]

fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='ORB0313_4')
# cmapを指定することでカラーマップの様子を変更するnpことができる
im = ax.scatter(longi, lati, c=pressure3, s=2, cmap='jet')
fig.colorbar(im, orientation='horizontal')

# %%
# dustの図を作成
# ダストとアルベドの評価図を作成
tau = [0.04, 0.14, 0.24, 0.34, 0.44]
A_15 = [-149.585, -75.1, 0, 74.619, 138.569]
A_20 = [-116.32, -57.554, 0, 60.816, 121.276]
A_25 = [-93.965, -47.304, 0, 50.443, 102.065]
A_29 = [-81.146, -41.302, 0, 44.177, 90.215]
A_33 = [-72.376, -38.045, 0, 37.042, 78.215]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(
    111, title='Influence of the dust opacity unsertainties on the pressure retrieval')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Assumed dust opacity', fontsize=10)
ax.set_ylabel('Surface pressure deviation (Pa)', fontsize=10)
ax.plot(tau, A_15, linestyle="dotted", label="A=0.15", color="black")
ax.plot(tau, A_20, linestyle="dashdot", label="A=0.20", color="black")
ax.plot(tau, A_25, linestyle="dashed", label="A=0.25", color="black")
ax.plot(tau, A_29, linestyle=(0, (1, 1)), label="A=0.29", color="black")
ax.plot(tau, A_33, label="A=0.33", color="black")

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=10)

# %%
# pressure fileをカラー表示させる
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

ind = np.where(pressure1 == 0)
pressure1[ind] = np.nan

ind = np.where(pressure2 == 0)
pressure2[ind] = np.nan

fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='ORB0920_3')
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi2, lati2, c=pressure2, s=2, cmap='plasma')
fig.colorbar(im, orientation='horizontal')

# %%
# TBD
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

ind1 = np.where((lati1 > 54) & (lati1 < 56) & (longi1 > 274) & (longi1 < 277))
ind2 = np.where((lati2 > 54) & (lati2 < 56) & (longi2 > 274) & (longi2 < 277))

pressure_1 = pressure1[ind1]
pressure_2 = pressure2[ind2]

longi_1 = longi1[ind1]
lati_1 = lati1[ind1]

pressure_dif = pressure_1[0:3520] - pressure_2[0:3520]
#pressure_dif[pressure_dif <= -100] = np.nan
#pressure_dif[pressure_dif >= 100] = np.nan

longi_dif = longi_1[0:3520]
lati_dif = lati_1[0:3520]

fig = plt.figure(figsize=(2, 5), dpi=200)
ax = fig.add_subplot(111, title='Pressure diffrence')
# cmapを指定することでカラーマップの様子を変更することができる
im = ax.scatter(longi_dif, lati_dif, c=pressure_dif, s=2, cmap='jet')
fig.colorbar(im, orientation='horizontal')

# %%
# multi and single scattering
fig = plt.figure(dpi=200)
ax = fig.add_subplot(
    111, title='Difference between Multi-scattering and Single-scattering')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavelength [μm]', fontsize=10)
ax.set_ylabel('Radiance', fontsize=10)

ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP11_TA1_TB1_SZA1_EA1_PA1_Dust2_WaterI2_SurfaceA1_rad_test.dat')
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


ARS1 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP11_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI2_SurfaceA1_rad_d0.dat')
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


ax.plot(ARS_x, y_calc, label="d=0.3", color="red")
ax.plot(ARS_x, y_calc1, label="d=0", color="blue")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=7)

# %%
# 高度補正
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname1 = '/Users/nyonn/Desktop/pythoncode/SPmap_ORB0313_4.sav'
sav_data1 = readsav(sav_fname1)

lati1 = sav_data1['lati']
longi1 = sav_data1['longi']
pressure = sav_data1['pressure']
tamap = sav_data1['tamap']

ip = len(lati1[:, 0])
io = len(lati1[0, :])

pressure1 = np.zeros((ip, io))
pressure1 = pressure + 0
pressure1[pressure1 == 0] = np.nan

tamap1 = np.zeros((ip, io))
tamap1 = tamap + 0
tamap1[tamap1 == 0] = np.nan

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
im = ax.scatter(longi1, lati1, c=pre_sev, s=2,
                cmap='jet', vmin=618, vmax=640)
fig.colorbar(im, orientation='horizontal')
