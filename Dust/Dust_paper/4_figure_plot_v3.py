# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
# 4章の図を作成する

import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
import glob
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# %%
# ------------------------------------- Figure 4.1a -------------------------------------
# datac coverageをplotした図を作成する
# MY27のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 0.2
max_dust = 0.4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY27_all_hist = Ls_good[derease_indices_all[0]:derease_indices_all[1]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY27", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY27_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[0], derease_indices_all[1], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=28)
ax2.set_ylabel("Latitude [Deg]", fontsize=20)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)

for i in range(0,82,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)

# %%
# ------------------------------------- Figure 4.1b -------------------------------------
# datac coverageをplotした図を作成する
# MY27のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 0.2
max_dust = 0.4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY28_all_hist = Ls_good[derease_indices_all[1]:derease_indices_all[2]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY28", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY28_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[1], derease_indices_all[2], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=28)
ax2.set_ylabel("Latitude [Deg]", fontsize=20)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)

for i in range(81,118,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)
# %%
# ------------------------------------- Figure 4.1c -------------------------------------
# datac coverageをplotした図を作成する
# MY29のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 0.2
max_dust = 0.4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY29_all_hist = Ls_good[derease_indices_all[2]:derease_indices_all[3]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY29", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY29_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[2], derease_indices_all[3], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=28)
ax2.set_ylabel("Latitude [Deg]", fontsize=20)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)

for i in range(118, count, 1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)

# %%
# ------------------------------------- Figure 4.2 -------------------------------------
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']

data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/coverage.sav")
# データを取り出す
ALL_ls = data['Ls_ind']
ALL_lt = data['LT_ind']
ALL_lat = data['lat_ind']

"""
# Ls 0 -180°
all_spring_summer = np.where((ALL_ls >= 0) & (ALL_ls < 180))
obs_spring_summer = np.where((OBS_ls >= 0) & (OBS_ls < 180))

ALL_ls_ss = ALL_ls[all_spring_summer]
ALL_lt_ss = ALL_lt[all_spring_summer]
OBS_ls_ss = OBS_ls[obs_spring_summer]
OBS_lt_ss = OBS_lt[obs_spring_summer]

# Ls 180 - 360°
all_autumn_winter = np.where((ALL_ls >= 180) & (ALL_ls < 360))
obs_autumn_winter = np.where((OBS_ls >= 180) & (OBS_ls < 360))

ALL_ls_aw = ALL_ls[all_autumn_winter]
ALL_lt_aw = ALL_lt[all_autumn_winter]
OBS_ls_aw = OBS_ls[obs_autumn_winter]
OBS_lt_aw = OBS_lt[obs_autumn_winter]

# Spring and Summer
ls_obs_ss_hist = OBS_ls_ss.flatten()
ls_all_ss_hist = ALL_ls_ss.flatten()
lt_obs_ss_hist = OBS_lt_ss.flatten()
lt_all_ss_hist = ALL_lt_ss.flatten()

# Autumn and Winter
ls_obs_aw_hist = OBS_ls_aw.flatten()
ls_all_aw_hist = ALL_ls_aw.flatten()
lt_obs_aw_hist = OBS_lt_aw.flatten()
lt_all_aw_hist = ALL_lt_aw.flatten()


# ヒストグラムを作成
# Spring and Summer
# からの配列を作成d
obs_ss_hist = np.zeros(24)
all_ss_hist = np.zeros(24)
obs_aw_hist = np.zeros(24)
all_aw_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    ind_ss_obs = np.where((OBS_lt_ss >= loop) & (OBS_lt_ss < loop+1))
    obs_ss_hist[loop] = len(OBS_lt_ss[ind_ss_obs])

    ind_ss_all = np.where((ALL_lt_ss >= loop) & (ALL_lt_ss < loop+1))
    all_ss_hist[loop] = len(ALL_lt_ss[ind_ss_all])

    ind_aw_obs = np.where((OBS_lt_aw >= loop) & (OBS_lt_aw < loop+1))
    obs_aw_hist[loop] = len(OBS_lt_aw[ind_aw_obs])

    ind_aw_all = np.where((ALL_lt_aw >= loop) & (ALL_lt_aw < loop+1))
    all_aw_hist[loop] = len(ALL_lt_aw[ind_aw_all])

# 確率を計算する
pro_ss = (obs_ss_hist/all_ss_hist) *100
pro_aw = (obs_aw_hist/all_aw_hist) *100

ind1 = np.where(obs_ss_hist > 0)
print('detection;', obs_ss_hist[ind1])
print('Observation', all_ss_hist[ind1])

ind2 = np.where(obs_aw_hist > 0)
print('detection;', obs_aw_hist[ind2])
print('Observation', all_aw_hist[ind2])

# ヒストグラムの作成
fig, axs = plt.subplots(2,2,figsize=(10,10),dpi=800)
axs[0,0].set_title("Spring and Summer", fontsize=20)
axs[0,0].set_ylabel("Number of all orbits", fontsize=14)
axs[0,0].set_xlim(6, 18)
axs[0,0].bar(range(0,24,1),all_ss_hist, color='red', edgecolor='black')
axs[0,0].set_ylim(0, 550)

axs[0,1].set_title("Autumn and Winter", fontsize=20)
axs[0,1].set_xlim(6, 18)
axs[0,1].set_ylim(0, 550)
axs[0,1].bar(range(0,24,1),all_aw_hist, color='blue', edgecolor='black')

axs[1,0].set_xlabel("Local time [h]", fontsize=14)
axs[1,0].set_ylabel("Probability [%]", fontsize=14)
axs[1,0].set_xlim(6, 18)
axs[1,0].bar(range(0,24,1),pro_ss, color='red', edgecolor='black')
axs[1,0].set_ylim(0, 10)

axs[1,1].set_xlabel("Local time [h]", fontsize=14)
axs[1,1].set_xlim(6, 18)
axs[1,1].set_ylim(0,10)
axs[1,1].bar(range(0,24,1),pro_aw ,color='blue', edgecolor='black')
plt.show()
"""

# LAT + LS
# SS
# Lat 0-30度
all_lat_0_30ss = np.where((ALL_lat >= 0) & (ALL_lat < 30) & (ALL_ls >= 0) & (ALL_ls < 180))
obs_lat_0_30ss = np.where((OBS_lat >= 0) & (OBS_lat < 30) & (OBS_ls >= 0) & (OBS_ls < 180))

# Lat 30-60度
all_lat_30_60ss = np.where((ALL_lat >= 30) & (ALL_lat < 60) & (ALL_ls >= 0) & (ALL_ls < 180))
obs_lat_30_60ss = np.where((OBS_lat >= 30) & (OBS_lat < 60) & (OBS_ls >= 0) & (OBS_ls < 180))

# lat -30-0度
all_lat_m30_0ss = np.where((ALL_lat >= -30) & (ALL_lat < 0) & (ALL_ls >= 0) & (ALL_ls < 180))
obs_lat_m30_0ss = np.where((OBS_lat >= -30) & (OBS_lat < 0) & (OBS_ls >= 0) & (OBS_ls < 180))

# lat -60-30度
all_lat_m60_m30ss = np.where((ALL_lat >= -60) & (ALL_lat < -30) & (ALL_ls >= 0) & (ALL_ls < 180))
obs_lat_m60_m30ss = np.where((OBS_lat >= -60) & (OBS_lat < -30) & (OBS_ls >= 0) & (OBS_ls < 180))

# AW
# Lat 0-30度
all_lat_0_30aw = np.where((ALL_lat >= 0) & (ALL_lat < 30) & (ALL_ls >= 180) & (ALL_ls < 360))
obs_lat_0_30aw = np.where((OBS_lat >= 0) & (OBS_lat < 30) & (OBS_ls >= 180) & (OBS_ls < 360))

# Lat 30-60度
all_lat_30_60aw = np.where((ALL_lat >= 30) & (ALL_lat < 60) & (ALL_ls >= 180) & (ALL_ls < 360))
obs_lat_30_60aw = np.where((OBS_lat >= 30) & (OBS_lat < 60) & (OBS_ls >= 180) & (OBS_ls < 360))

# lat -30-0度
all_lat_m30_0aw = np.where((ALL_lat >= -30) & (ALL_lat < 0) & (ALL_ls >= 180) & (ALL_ls < 360))
obs_lat_m30_0aw = np.where((OBS_lat >= -30) & (OBS_lat < 0) & (OBS_ls >= 180) & (OBS_ls < 360))

# lat -60-30度
all_lat_m60_m30aw = np.where((ALL_lat >= -60) & (ALL_lat < -30) & (ALL_ls >= 180) & (ALL_ls < 360))
obs_lat_m60_m30aw = np.where((OBS_lat >= -60) & (OBS_lat < -30) & (OBS_ls >= 180) & (OBS_ls < 360))

# Local timeに格納する
all_ss_NH_low_lt_hist = ALL_lt[all_lat_0_30ss]
obs_ss_NH_low_lt_hist = OBS_lt[obs_lat_0_30ss]
all_ss_NH_mid_lt_hist = ALL_lt[all_lat_30_60ss]
obs_ss_NH_mid_lt_hist = OBS_lt[obs_lat_30_60ss]
all_ss_SH_low_lt_hist = ALL_lt[all_lat_m30_0ss]
obs_ss_SH_low_lt_hist = OBS_lt[obs_lat_m30_0ss]
all_ss_SH_mid_lt_hist = ALL_lt[all_lat_m60_m30ss]
obs_ss_SH_mid_lt_hist = OBS_lt[obs_lat_m60_m30ss]

all_aw_NH_low_lt_hist = ALL_lt[all_lat_0_30aw]
obs_aw_NH_low_lt_hist = OBS_lt[obs_lat_0_30aw]
all_aw_NH_mid_lt_hist = ALL_lt[all_lat_30_60aw]
obs_aw_NH_mid_lt_hist = OBS_lt[obs_lat_30_60aw]
all_aw_SH_low_lt_hist = ALL_lt[all_lat_m30_0aw]
obs_aw_SH_low_lt_hist = OBS_lt[obs_lat_m30_0aw]
all_aw_SH_mid_lt_hist = ALL_lt[all_lat_m60_m30aw]
obs_aw_SH_mid_lt_hist = OBS_lt[obs_lat_m60_m30aw]

"""
# SS
all_ss_NH_low_lt_hist = all_lat_0_30ss.flatten()
obs_ss_NH_low_lt_hist = obs_lat_0_30ss.flatten()
all_ss_NH_mid_lt_hist = all_lat_30_60ss.flatten()
obs_ss_NH_mid_lt_hist = obs_lat_30_60ss.flatten()
all_ss_SH_low_lt_hist = all_lat_m30_0ss.flatten()
obs_ss_SH_low_lt_hist = obs_lat_m30_0ss.flatten()
all_ss_SH_mid_lt_hist = all_lat_m60_m30ss.flatten()
obs_ss_SH_mid_lt_hist = obs_lat_m60_m30ss.flatten()

# AW
all_aw_NH_low_lt_hist = all_lat_0_30aw.flatten()
obs_aw_NH_low_lt_hist = obs_lat_0_30aw.flatten()
all_aw_NH_mid_lt_hist = all_lat_30_60aw.flatten()
obs_aw_NH_mid_lt_hist = obs_lat_30_60aw.flatten()
all_aw_SH_low_lt_hist = all_lat_m30_0aw.flatten()
obs_aw_SH_low_lt_hist = obs_lat_m30_0aw.flatten()
all_aw_SH_mid_lt_hist = all_lat_m60_m30aw.flatten()
obs_aw_SH_mid_lt_hist = obs_lat_m60_m30aw.flatten()
"""

# ヒストグラムを作成
# Spring and Summer
obs_low_nh_ss_hist = np.zeros(24)
all_low_nh_ss_hist = np.zeros(24)
obs_mid_nh_ss_hist = np.zeros(24)
all_mid_nh_ss_hist = np.zeros(24)
obs_low_sh_ss_hist = np.zeros(24)
all_low_sh_ss_hist = np.zeros(24)
obs_mid_sh_ss_hist = np.zeros(24)
all_mid_sh_ss_hist = np.zeros(24)

# Autumn and Winter
obs_low_nh_aw_hist = np.zeros(24)
all_low_nh_aw_hist = np.zeros(24)
obs_mid_nh_aw_hist = np.zeros(24)
all_mid_nh_aw_hist = np.zeros(24)
obs_low_sh_aw_hist = np.zeros(24)
all_low_sh_aw_hist = np.zeros(24)
obs_mid_sh_aw_hist = np.zeros(24)
all_mid_sh_aw_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    # Spring and Summer
    # 緯度0-30度
    ind_ss_obs_low_nh = np.where((obs_ss_NH_low_lt_hist >= loop) & (obs_ss_NH_low_lt_hist < loop+1))
    obs_low_nh_ss_hist[loop] = len(obs_ss_NH_low_lt_hist[ind_ss_obs_low_nh])
    ind_ss_all_low_nh = np.where((all_ss_NH_low_lt_hist >= loop) & (all_ss_NH_low_lt_hist < loop+1))
    all_low_nh_ss_hist[loop] = len(all_ss_NH_low_lt_hist[ind_ss_all_low_nh])

    # 緯度30-60度
    ind_ss_obs_mid_nh = np.where((obs_ss_NH_mid_lt_hist >= loop) & (obs_ss_NH_mid_lt_hist < loop+1))
    obs_mid_nh_ss_hist[loop] = len(obs_ss_NH_mid_lt_hist[ind_ss_obs_mid_nh])
    ind_ss_all_mid_nh = np.where((all_ss_NH_mid_lt_hist >= loop) & (all_ss_NH_mid_lt_hist < loop+1))
    all_mid_nh_ss_hist[loop] = len(all_ss_NH_mid_lt_hist[ind_ss_all_mid_nh])

    # 緯度-30-0度
    ind_ss_obs_low_sh = np.where((obs_ss_SH_low_lt_hist >= loop) & (obs_ss_SH_low_lt_hist < loop+1))
    obs_low_sh_ss_hist[loop] = len(obs_ss_SH_low_lt_hist[ind_ss_obs_low_sh])
    ind_ss_all_low_sh = np.where((all_ss_SH_low_lt_hist >= loop) & (all_ss_SH_low_lt_hist < loop+1))
    all_low_sh_ss_hist[loop] = len(all_ss_SH_low_lt_hist[ind_ss_all_low_sh])

    # 緯度-60-30度
    ind_ss_obs_mid_sh = np.where((obs_ss_SH_mid_lt_hist >= loop) & (obs_ss_SH_mid_lt_hist < loop+1))
    obs_mid_sh_ss_hist[loop] = len(obs_ss_SH_mid_lt_hist[ind_ss_obs_mid_sh])
    ind_ss_all_mid_sh = np.where((all_ss_SH_mid_lt_hist >= loop) & (all_ss_SH_mid_lt_hist < loop+1))
    all_mid_sh_ss_hist[loop] = len(all_ss_SH_mid_lt_hist[ind_ss_all_mid_sh])

    # Autumn and Winter
    # 緯度0-30度
    ind_aw_obs_low_nh = np.where((obs_aw_NH_low_lt_hist >= loop) & (obs_aw_NH_low_lt_hist < loop+1))
    obs_low_nh_aw_hist[loop] = len(obs_aw_NH_low_lt_hist[ind_aw_obs_low_nh])
    ind_aw_all_low_nh = np.where((all_aw_NH_low_lt_hist >= loop) & (all_aw_NH_low_lt_hist < loop+1))
    all_low_nh_aw_hist[loop] = len(all_aw_NH_low_lt_hist[ind_aw_all_low_nh])

    # 緯度30-60度
    ind_aw_obs_mid_nh = np.where((obs_aw_NH_mid_lt_hist >= loop) & (obs_aw_NH_mid_lt_hist < loop+1))
    obs_mid_nh_aw_hist[loop] = len(obs_aw_NH_mid_lt_hist[ind_aw_obs_mid_nh])
    ind_aw_all_mid_nh = np.where((all_aw_NH_mid_lt_hist >= loop) & (all_aw_NH_mid_lt_hist < loop+1))
    all_mid_nh_aw_hist[loop] = len(all_aw_NH_mid_lt_hist[ind_aw_all_mid_nh])

    # 緯度-30-0度
    ind_aw_obs_low_sh = np.where((obs_aw_SH_low_lt_hist >= loop) & (obs_aw_SH_low_lt_hist < loop+1))
    obs_low_sh_aw_hist[loop] = len(obs_aw_SH_low_lt_hist[ind_aw_obs_low_sh])
    ind_aw_all_low_sh = np.where((all_aw_SH_low_lt_hist >= loop) & (all_aw_SH_low_lt_hist < loop+1))
    all_low_sh_aw_hist[loop] = len(all_aw_SH_low_lt_hist[ind_aw_all_low_sh])

    # 緯度-60-30度
    ind_aw_obs_mid_sh = np.where((obs_aw_SH_mid_lt_hist >= loop) & (obs_aw_SH_mid_lt_hist < loop+1))
    obs_mid_sh_aw_hist[loop] = len(obs_aw_SH_mid_lt_hist[ind_aw_obs_mid_sh])
    ind_aw_all_mid_sh = np.where((all_aw_SH_mid_lt_hist >= loop) & (all_aw_SH_mid_lt_hist < loop+1))
    all_mid_sh_aw_hist[loop] = len(all_aw_SH_mid_lt_hist[ind_aw_all_mid_sh])

# 確率を計算する
# Spring and Summer
pro_low_nh_ss = (obs_low_nh_ss_hist/all_low_nh_ss_hist) *100
pro_mid_nh_ss = (obs_mid_nh_ss_hist/all_mid_nh_ss_hist) *100
pro_low_sh_ss = (obs_low_sh_ss_hist/all_low_sh_ss_hist) *100
pro_mid_sh_ss = (obs_mid_sh_ss_hist/all_mid_sh_ss_hist) *100

# Autumn and Winter
pro_low_nh_aw = (obs_low_nh_aw_hist/all_low_nh_aw_hist) *100
pro_mid_nh_aw = (obs_mid_nh_aw_hist/all_mid_nh_aw_hist) *100
pro_low_sh_aw = (obs_low_sh_aw_hist/all_low_sh_aw_hist) *100
pro_mid_sh_aw = (obs_mid_sh_aw_hist/all_mid_sh_aw_hist) *100

# 各確率はどのくらいのものかを確認する
# Spring and Summer
# 緯度0-30度
print('--- Spring and Summer ---')
ind1 = np.where(obs_low_nh_ss_hist > 0)
print('緯度0-30度')
print('detection;', obs_low_nh_ss_hist[ind1])
print('Observation', all_low_nh_ss_hist[ind1])

# 緯度30-60度
ind2 = np.where(obs_mid_nh_ss_hist > 0)
print('緯度30-60度')
print('detection;', obs_mid_nh_ss_hist[ind2])
print('Observation', all_mid_nh_ss_hist[ind2])

# 緯度-30-0度
ind3 = np.where(obs_low_sh_ss_hist > 0)
print('緯度-30-0度')
print('detection;', obs_low_sh_ss_hist[ind3])
print('Observation', all_low_sh_ss_hist[ind3])

# 緯度-60-30度
ind4 = np.where(obs_mid_sh_ss_hist > 0)
print('緯度-60-30度')
print('detection;', obs_mid_sh_ss_hist[ind4])
print('Observation', all_mid_sh_ss_hist[ind4])

print('--- Autumn and Winter ---')
ind5 = np.where(obs_low_nh_aw_hist > 0)
print('緯度0-30度')
print('detection;', obs_low_nh_aw_hist[ind5])
print('Observation', all_low_nh_aw_hist[ind5])

# 緯度30-60度
ind6 = np.where(obs_mid_nh_aw_hist > 0)
print('緯度30-60度')
print('detection;', obs_mid_nh_aw_hist[ind6])
print('Observation', all_mid_nh_aw_hist[ind6])

# 緯度-30-0度
ind7 = np.where(obs_low_sh_aw_hist > 0)
print('緯度-30-0度')
print('detection;', obs_low_sh_aw_hist[ind7])
print('Observation', all_low_sh_aw_hist[ind7])

# 緯度-60-30度
ind8 = np.where(obs_mid_sh_aw_hist > 0)
print('緯度-60-30度')
print('detection;', obs_mid_sh_aw_hist[ind8])

# ヒストグラムの作成
# 北半球
fig, axs = plt.subplots(2,2,figsize=(10,10),dpi=800)
axs[0,0].set_title("Ls 0 to 180", fontsize=20)
axs[0,0].set_ylabel("Number of all orbits", fontsize=14)
axs[0,0].set_xlim(6, 18)
axs[0,0].set_ylim(0, 200)
axs[0,0].bar(range(0,24,1),all_low_nh_ss_hist, color='red', edgecolor='black',label='0-30°',align="edge", width= -0.3)
axs[0,0].bar(range(0,24,1),all_mid_nh_ss_hist, color='blue', edgecolor='black',label='30-60°',align="edge", width= 0.3)
axs[0,0].legend()

axs[0,1].set_title("Ls 180 to 360", fontsize=20)
axs[0,1].set_xlim(6, 18)
axs[0,1].set_ylim(0, 200)
axs[0,1].bar(range(0,24,1),all_low_nh_aw_hist, color='red', edgecolor='black',label='0-30°',align="edge", width= -0.3)
axs[0,1].bar(range(0,24,1),all_mid_nh_aw_hist, color='blue', edgecolor='black',label='30-60°',align="edge", width= 0.3)

# 確率
axs[1,0].set_xlabel("Local time [h]", fontsize=14)
axs[1,0].set_ylabel("Probability [%]", fontsize=14)
axs[1,0].set_xlim(6, 18)
axs[1,0].bar(range(0,24,1),pro_low_nh_ss, color='red', edgecolor='black',label='0-30°',align="edge", width= -0.3)
axs[1,0].bar(range(0,24,1),pro_mid_nh_ss, color='blue', edgecolor='black',label='30-60°',align="edge", width= 0.3)
axs[1,0].set_ylim(0, 30)

axs[1,1].set_xlabel("Local time [h]", fontsize=14)
axs[1,1].set_xlim(6, 18)
axs[1,1].set_ylim(0,30)
axs[1,1].bar(range(0,24,1),pro_low_nh_aw ,color='red', edgecolor='black',label='0-30°',align="edge", width= -0.3)
axs[1,1].bar(range(0,24,1),pro_mid_nh_aw ,color='blue', edgecolor='black',label='30-60°',align="edge", width= 0.3)
plt.show()

# 南半球
fig, axs = plt.subplots(2,2,figsize=(10,10),dpi=800)
axs[0,0].set_title("Ls 0 to 180", fontsize=20)
axs[0,0].set_ylabel("Number of all orbits", fontsize=14)
axs[0,0].set_xlim(6, 18)
axs[0,0].set_ylim(0, 200)
axs[0,0].bar(range(0,24,1),all_low_sh_ss_hist, color='red', edgecolor='black',label='-30-0°',align="edge", width= -0.3)
axs[0,0].bar(range(0,24,1),all_mid_sh_ss_hist, color='blue', edgecolor='black',label='-60-30°',align="edge", width= 0.3)
axs[0,0].legend()

axs[0,1].set_title("Ls 180 to 360", fontsize=20)
axs[0,1].set_xlim(6, 18)
axs[0,1].set_ylim(0, 200)
axs[0,1].bar(range(0,24,1),all_low_sh_aw_hist, color='red', edgecolor='black',label='-30-0°',align="edge", width= -0.3)
axs[0,1].bar(range(0,24,1),all_mid_sh_aw_hist, color='blue', edgecolor='black',label='-60-30°',align="edge", width= 0.3)

# 確率
axs[1,0].set_xlabel("Local time [h]", fontsize=14)
axs[1,0].set_ylabel("Probability [%]", fontsize=14)
axs[1,0].set_xlim(6, 18)
axs[1,0].bar(range(0,24,1),pro_low_sh_ss, color='red', edgecolor='black',label='-30-0°',align="edge", width= -0.3)
axs[1,0].bar(range(0,24,1),pro_mid_sh_ss, color='blue', edgecolor='black',label='-60-30°',align="edge", width= 0.3)
axs[1,0].set_ylim(0, 30)

axs[1,1].set_xlabel("Local time [h]", fontsize=14)
axs[1,1].set_xlim(6, 18)
axs[1,1].set_ylim(0,30)
axs[1,1].bar(range(0,24,1),pro_low_sh_aw ,color='red', edgecolor='black',label='-30-0°',align="edge", width= -0.3)
axs[1,1].bar(range(0,24,1),pro_mid_sh_aw ,color='blue', edgecolor='black',label='-60-30°',align="edge", width= 0.3)
plt.show()

# %%    
# ------------------------------------- Figure 4.3 -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/old/4章：Analysis of OMEGA data set/data/ALL_EW_detect_lon-lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/old/4章：Analysis of OMEGA data set/data/EW_detect_lon-lat.sav")
OBS_info = OBS_data['detect_number']

# データを転置する
ALL_info = ALL_info.T
OBS_info = OBS_info.T

# 経度の配列を作成する
# 配列型は(181,361)で、緯度は-90から90まで、経度は0から360まで
longi = np.arange(361) * 1
Longi = np.tile(longi, (181, 1))

latitude = np.arange(181) * 1 - 90
Lati = np.tile(latitude, (361, 1))
Lati = np.transpose(Lati)

# color mapの設定
cmap = plt.get_cmap('Reds')
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=8))
sm.set_array([])

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_xlim(-15, 360)
ax.set_ylim(-100, 100)

# 15°×15°の格子点を作成する
smooth_all = np.zeros((13, 24))
smooth_obs = np.zeros((13, 24))

smooth_lati = np.arange(13)*15 - 90
smooth_lat = np.tile(smooth_lati, (25, 1))
smooth_lat = np.transpose(smooth_lat)

smooth_longi = np.arange(25)*15
smooth_long = np.tile(smooth_longi, (13, 1))

for i in range(0, 13, 1):
    for j in range(0, 24, 1):
        ind_smooth= np.where((Lati >= smooth_lati[i]) & (Lati < smooth_lati[i] + 15) & (Longi >= smooth_longi[j]) & (Longi < smooth_longi[j] + 15))

        smooth_all[i, j] = np.sum(ALL_info[ind_smooth])
        smooth_obs[i, j] = np.sum(OBS_info[ind_smooth])

# ALL_infoが0の場所と0でない場所を特定します
zero_indices = np.where(smooth_all == 0)
non_zero_indices = np.where(smooth_all != 0)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(smooth_long[zero_indices], smooth_lat[zero_indices], color="grey", alpha=0.2, s=100, marker='s')

RATIO_map = (smooth_obs[non_zero_indices] / smooth_all[non_zero_indices]) * 100

colors = sm.to_rgba(RATIO_map.flatten())
ax.scatter(smooth_long[non_zero_indices], smooth_lat[non_zero_indices], color=colors,marker='s', s=100)

# カラーバーを作成
cbar = plt.colorbar(sm)
cbar.set_label("Probability of detection [%]", fontsize=14)
plt.show()

# %%
# ------------------------------------- Figure 4.4 -------------------------------------
# MY27のSSに着目して、可視の画像と比較した図を作成する

# set data
set_MY = 27
set_min_Ls = 133
set_max_Ls = 140
set_min_lon = 0
set_max_lon = 360
set_min_lat = -90
set_max_lat = 90

# 図の枠組をここで設定する
# color mapの設定
# LSで色を変える
cmap = plt.get_cmap('jet')

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [Deg]", fontsize=14)
ax.set_ylabel("Latitude [Deg]", fontsize=14)
ax.set_xlim(set_min_lon, set_max_lon)
ax.set_ylim(set_min_lat, set_max_lat)

# sav fileを読み込む
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

# xlsx fileを読み込む
excel_file = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/MDAD.xlsx"
# Excelファイルを読み込みます
df = pd.read_excel(excel_file)

# データを取り出す
MarsYear = df['Mars Year']
Ls = df['Ls']
predict = df['Confidence interval']
seaquence = df['Sequence ID']
center_lon = df['Centroid longitude']
center_lat = df['Centroid latitude']
max_lat = df['Maximum latitude']
min_lat = df['Minimum latitude']
area = df['Area (square km)']

# 指定された場所の各データを取り出す
#ind = np.where((MarsYear == set_MY) & (Ls >= set_min_Ls) & (Ls <= set_max_Ls) & (seaquence == "s01_01"))[0]
ind = np.where((MarsYear == set_MY) & (Ls >= set_min_Ls) & (Ls <= set_max_Ls) )[0]
center_lon = center_lon[ind]
center_lat = center_lat[ind]
max_lat = max_lat[ind]
min_lat = min_lat[ind]
area = area[ind]

longth_lat = (max_lat - min_lat) * 60  #km
wide_lat = (max_lat - min_lat)
wide_DS = (area / longth_lat) / 60  #km
wide_lon = wide_DS / 2
min_lon = center_lon - wide_lon
max_lon = center_lon + wide_lon

min_lon = np.array(min_lon)
min_lat = np.array(min_lat)
wide_lon = np.array(wide_lon)
wide_lat = np.array(wide_lat)

# colorの設定
norm = Normalize(vmin=set_min_Ls, vmax=set_max_Ls)
sm = ScalarMappable(cmap=cmap, norm=norm)

# longitudeがマイナスのときに360を足す
for i in range(len(min_lon)):
    if min_lon[i] < 0:
        min_lon[i] = min_lon[i] + 360

for loop in range(len(min_lon)):
    color = sm.to_rgba(Ls[ind[loop]])
    rectangles = patches.Rectangle((min_lon[loop], min_lat[loop]), wide_lon[loop], wide_lat[loop], edgecolor=color, facecolor='none', linewidth=2,alpha=0.5)
    ax.add_patch(rectangles)

# 上のファイルを一つずつ読み込んでいく
for i in range(0,81,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    # LSがsetで指定された範囲内のデータのみを取り出す
    if Ls > set_min_Ls and Ls < set_max_Ls:
        LDS_ind = np.where(bin_med > 0.2)
        bin_lat = bin_lat[LDS_ind]
        bin_lon = bin_lon[LDS_ind]

        bin_med = bin_med[LDS_ind]

        color = sm.to_rgba(Ls)
        ax.scatter(bin_lon, bin_lat, color=color, s=1)
        print(files[i])

# カラーバーを作成
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label("Solar longitude [Deg]", fontsize=14)
plt.show()

# %%
# ------------------------------------- Figure 4.5 -------------------------------------
# ORB4477_2の例を示す図を作る
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/ORB4477_2.sav"
sav_data = readsav(sav_fname)
longitude = sav_data["longi"]
latitude = sav_data["lati"]
dust_opacity = sav_data["dust_opacity"]
altitude = sav_data["ret_alt_dust"]

dust_opacity = dust_opacity + 0

ind = np.where(dust_opacity <= 0)
dust_opacity[ind] = np.nan

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
c = ax.contourf(longitude, latitude, dust_opacity, cmap="jet",levels=np.linspace(0, 0.75))
cbar = fig.colorbar(c, ax=ax, orientation="vertical", format='%.2f',ticks=np.arange(0, 0.76, 0.15))
cbar.set_label("Dust optical depth", fontsize=12)
plt.show()

# %%
# ------------------------------------- Figure 4.6 -------------------------------------
# ORB4477_2がGDSにどんなふうに影響を与えるかを示す図を作る
fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [Deg]", fontsize=14)
ax.set_ylabel("Latitude [Deg]", fontsize=14)
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)

# set ls range
set_min_Ls = 257
set_max_Ls = 267

# color mapの設定
min_dust = 0.001
max_dust = 0.6
cmap = plt.get_cmap('jet')

# predict GDS [Wang and Richardson, 2015]
GDS_location_north = [28.4,319.7]
GDS_location_south = [-42.4,70.5]

# 星で示す
ax.scatter(GDS_location_south[1], GDS_location_south[0], color="red", s=100, label="GDS",marker='*')
ax.scatter(GDS_location_north[1], GDS_location_north[0], color="red", s=100,marker='*')

# sav fileを読み込む
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

for i in range(81,120,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    # LSがsetで指定された範囲内のデータのみを取り出す
    if Ls > set_min_Ls and Ls < set_max_Ls:
        LDS_ind = np.where(bin_med > 0.2)
        bin_lat = bin_lat[LDS_ind]
        bin_lon = bin_lon[LDS_ind]

        bin_med = bin_med[LDS_ind]
        # color mapの設定
        norm = Normalize(vmin=min_dust, vmax=max_dust)
        # Create a ScalarMappable with the colormap and normalization
        sm = ScalarMappable(cmap=cmap, norm=norm)
        # Use the ScalarMappable to get the color
        color = sm.to_rgba(np.median(bin_med))

        ax.scatter(bin_lon, bin_lat, color=color, s=1)

# カラーバーを作成
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label("Dust optical depth", fontsize=14)

key_lds = files[109]
lds_data = readsav(key_lds)
bin_lat = lds_data['bin_lat']
bin_lon = lds_data['bin_lon']
bin_med = lds_data['bin_med']
LDS_ind = np.where(bin_med > 0.2)
bin_lat = bin_lat[LDS_ind]
bin_lon = bin_lon[LDS_ind]

lds_lat = np.median(bin_lat)
lds_lon = np.median(bin_lon)
ax.scatter(lds_lon, lds_lat, color="blue", s=100, label="KEY LDS",marker='*')
ax.legend()
plt.show()

# %%
# ------------------------------------- Pour mars conference -------------------------------------
# Local dust stormを検出したファイルを読み込む
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

# color mapの設定
min_dust = 0.2
max_dust = 0.6
cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(3,1,figsize=(15,12),dpi=300,sharex=True,sharey=True)
# color mapの設定
norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

axs[0].set_ylabel("Latitude [Deg]", fontsize=20)
axs[0].set_title("MY27", fontsize=24)
axs[0].set_xlim(0, 360)
axs[0].set_ylim(-90, 90)

# MY27
for j in range(0,81,1):
    # ファイルを読み込む
    data = readsav(files[j])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)

    bin_med = bin_med[LDS_ind]

    # Use the ScalarMappable to get the color
    color = sm.to_rgba(np.median(bin_med))
    axs[0].scatter(Ls,med_lat, color=color)

# MY28
axs[1].set_ylabel("Latitude [Deg]", fontsize=20)
axs[1].set_title("MY28", fontsize=24)
axs[1].set_xlim(0, 360)
axs[1].set_ylim(-90, 90)

for j in range(81,118,1):
    # ファイルを読み込む
    data = readsav(files[j])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)

    bin_med = bin_med[LDS_ind]
    color = sm.to_rgba(np.median(bin_med))

    axs[1].scatter(Ls,med_lat, color=color)

# MY29
#ax4 = fig.add_subplot(414)
axs[2].set_xlabel("Solar longitude [Deg]", fontsize=24)
axs[2].set_ylabel("Latitude [Deg]", fontsize=20)
axs[2].set_title("MY29", fontsize=24)
axs[2].set_xlim(0, 360)
axs[2].set_ylim(-90, 90)

for j in range(118,count,1):
    # ファイルを読み込む
    data = readsav(files[j])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm.to_rgba(np.median(bin_med))
    axs[2].scatter(Ls,med_lat, color=color, linewidth=4)

# カラーバーを作成
cbar = plt.colorbar(sm, ax=axs.ravel().tolist(),orientation='vertical',aspect=90)
cbar.set_label("Dust optical depth", fontsize=20)

plt.show()
# %%
# ダストの分布をLTごとに示す
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/histo_info.sav")

# dust optical depth
min_dust = 0.15
max_dust = 0.4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']
OBS_lon = data['obs_lon_ind']
OBS_dust = data['obs_dust_ind']
file_orbit = data['files_info']

for i in range(8,18,1):
          min_lt = i
          max_lt = i+1

          ind_lt = np.where((OBS_lt >= min_lt) & (OBS_lt < max_lt))
          lt_lat = OBS_lat[ind_lt]
          lt_lon = OBS_lon[ind_lt]
          dust_color = OBS_dust[ind_lt]

          fig = plt.figure(dpi=800)
          ax = fig.add_subplot(111)
          ax.set_title("LDS map: LT" + str(min_lt) + " - " + str(max_lt))
          ax.set_xlabel("Longitude [Deg]", fontsize=14)
          ax.set_ylabel("Latitude [Deg]", fontsize=14)
          ax.set_xlim(-5, 365)
          ax.set_ylim(-60, 60)

          color = sm2.to_rgba(dust_color)
          ax.scatter(lt_lon,lt_lat,s=7,color=color)

          cbar = plt.colorbar(sm2,orientation='vertical')
          cbar.set_label("Dust optical depth", fontsize=7)

          if i == 16:
                    print(file_orbit[ind_lt])
