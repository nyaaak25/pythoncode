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
# ------------------------------------- Figure 10 -------------------------------------
# detection critriaについての図を作成する
# まずは生のリトリーバルデータをプロットする
# ------------------------------------- Figure 10a -------------------------------------
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/ORB0920_3_1.sav"
# 1448_4, 1339_1, 4482_3
sav_data = readsav(sav_fname)
longitude = sav_data["longi"]
latitude = sav_data["lati"]

dust_opacity = sav_data["dust_opacity"]
dust_opacity = dust_opacity + 0

ind = np.where(dust_opacity <= 0)
dust_opacity[ind] = np.nan
dust_opacity[:,80:96] = np.nan

# dust optical depth
min_dust = 0.01
max_dust = 1.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)
im = ax.scatter(longitude, latitude, c=dust_opacity, cmap="jet", vmin=min_dust, vmax=max_dust,s=2)
fig.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')

data_sav = np.stack((longitude, latitude, dust_opacity), axis=-1)
data_sav_2d = data_sav.reshape(-1, data_sav.shape[-1])
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure10/ORB1448_1-raw.txt", data_sav_2d)
# %%
# 1deg×1degマップを作成する
# ------------------------------------- Figure 10b -------------------------------------
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/ORB1339_1.sav"
sav_data = readsav(sav_fname)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)
im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
fig.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')

data_bin = np.stack((lon, lat, med), axis=-1)
data_bin_2d = data_bin.reshape(-1, data_bin.shape[-1])
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure10/ORB1339_1-bin.txt", data_bin_2d)


# %%
# non-detection例のデータを作成する
# ------------------------------------- Figure 10c -------------------------------------
sav_frame = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ORB0920_3.sav"
sav_data = readsav(sav_frame)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 1.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)
im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
fig.colorbar(im, orientation="vertical", label="Dust optical depth")

# %%
# detection例を示す
# ------------------------------------- Figure 10d -------------------------------------
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/EW-detect/ORB1448_4.sav"
sav_data = readsav(sav_fname)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)
# medの値が2.0以上のところは赤くプロットする
ind = np.where(med > 2.0)
ax.scatter(lon[ind], lat[ind], c="black",s=80)

im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
#ax.scatter(lon[ind], lat[ind], c="red")
fig.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')

# %%
# non-detection例のデータを作成する
# ------------------------------------- Figure 10e -------------------------------------
sav_frame = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ORB4482_3.sav"
sav_data = readsav(sav_frame)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)

ind = np.where(med > 2.0)
#ax.scatter(lon[ind], lat[ind], c="black",s=80)

im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
fig.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')

# %%
# non-detection例のデータを作成する
# ------------------------------------- Figure 10f -------------------------------------
sav_frame = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ORB1339_1.sav"
sav_data = readsav(sav_frame)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [°E]", fontsize=14)
ax.set_ylabel("Latitude [°N]", fontsize=14)

ind = np.where(med > 2.0)
#ax.scatter(lon[ind], lat[ind], c="black",s=80)

im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
fig.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')



# %%
# ------------------------------------- Figure 11a -------------------------------------
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
min_dust = 2.0
max_dust = 4.0
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
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
MY27_all_hist = Ls_good[derease_indices_all[0]+1:derease_indices_all[1]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY27", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY27_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_all_ls.txt", MY27_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[0], derease_indices_all[1], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My27_all_localtime = local_good[:,0,derease_indices_all[0]+1:derease_indices_all[1]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_all_localtime.txt", My27_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
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

my27_detect_lds = np.zeros((95))
my27_detect_lds_lc = np.zeros((95))
my27_detect_lds_ls = np.zeros((95))

for i in range(0,94,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)
    my27_detect_lds[i] = np.median(bin_med)
    my27_detect_lds_lc[i] = np.median(bin_lat)
    my27_detect_lds_ls[i] = Ls

my27_data_lds = np.stack((my27_detect_lds,my27_detect_lds_lc, my27_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_detect_lds.txt", my27_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)

# %%
# ------------------------------------- Figure 11b -------------------------------------
# datac coverageをplotした図を作成する
# MY28のデータをplotする

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
min_dust = 2.0
max_dust = 4.0
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
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
MY28_all_hist = Ls_good[derease_indices_all[1]+1:derease_indices_all[2]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY28", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY28_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_all_ls.txt", MY28_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[1]+1, derease_indices_all[2], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My28_all_localtime = local_good[:,0,derease_indices_all[1]+1:derease_indices_all[2]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_all_localtime.txt", My28_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
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

my28_detect_lds = np.zeros((35))
my28_detect_lds_lc = np.zeros((35))
my28_detect_lds_ls = np.zeros((35))

for i in range(95,130,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

    my28_detect_lds[i-95] = np.median(bin_med)
    my28_detect_lds_lc[i-95] = np.median(bin_lat)
    my28_detect_lds_ls[i-95] = Ls

my28_data_lds = np.stack((my28_detect_lds,my28_detect_lds_lc,my28_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_detect_lds.txt", my28_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)
# %%
# ------------------------------------- Figure 11c -------------------------------------
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
min_dust = 2
max_dust = 4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
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
MY29_all_hist = Ls_good[derease_indices_all[2]+1:derease_indices_all[3]]
ax0.set_ylabel("Number of orbits", fontsize=20)
ax0.set_title("MY29", fontsize=30)
ax0.set_xlim(0, 360)
ax0.hist(MY29_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_all_ls.txt", MY29_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=20)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)

for i in range(derease_indices_all[2]+1, derease_indices_all[3], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My29_all_localtime = local_good[:,0,derease_indices_all[2]+1:derease_indices_all[3]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_all_localtime.txt", My29_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
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

my29_detect_lds = np.zeros((15))
my29_detect_lds_lc = np.zeros((15))
my29_detect_lds_ls = np.zeros((15))

for i in range(131, count, 1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

    my29_detect_lds[i-131] = np.median(bin_med)
    my29_detect_lds_lc[i-131] = np.median(bin_lat)
    my29_detect_lds_ls[i-131] = Ls

my29_data_lds = np.stack((my29_detect_lds,my29_detect_lds_lc,my29_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_detect_lds.txt", my29_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.set_label("Local time [h]", fontsize=18)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.set_label("Dust optical depth", fontsize=18)

# %%
# ------------------------------------- Figure 12a -------------------------------------
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']

data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/coverage.sav")
# データを取り出す
ALL_ls = data['Ls_ind']
ALL_lt = data['LT_ind']
ALL_lat = data['lat_ind']

# LS 0-180度における領域の時間軸ヒストグラムを作成する
all_spring_summer = np.where((ALL_ls >= 0) & (ALL_ls < 180))
obs_spring_summer = np.where((OBS_ls >= 0) & (OBS_ls < 180))

ALL_ls_ss = ALL_ls[all_spring_summer]
ALL_lt_ss = ALL_lt[all_spring_summer]
OBS_ls_ss = OBS_ls[obs_spring_summer]
OBS_lt_ss = OBS_lt[obs_spring_summer]

ls_obs_ss_hist = OBS_ls_ss.flatten()
ls_all_ss_hist = ALL_ls_ss.flatten()
lt_obs_ss_hist = OBS_lt_ss.flatten()
lt_all_ss_hist = ALL_lt_ss.flatten()

# からの配列を作成
obs_ss_hist = np.zeros(24)
all_ss_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    ind_ss_obs = np.where((OBS_lt_ss >= loop) & (OBS_lt_ss < loop+1))
    obs_ss_hist[loop] = len(OBS_lt_ss[ind_ss_obs])

    ind_ss_all = np.where((ALL_lt_ss >= loop) & (ALL_lt_ss < loop+1))
    all_ss_hist[loop] = len(ALL_lt_ss[ind_ss_all])

# all_ss_histがゼロでない場合のみ確率を計算する
pro_ss = np.zeros(24)
nonzero_indices = all_ss_hist > 0
pro_ss[nonzero_indices] = (obs_ss_hist[nonzero_indices] / all_ss_hist[nonzero_indices]) * 100

# エラーバーの計算（確率を0-1に変換）
er_bar = np.zeros(24)
p = pro_ss[nonzero_indices] / 100  # パーセンテージから確率に変換
n = all_ss_hist[nonzero_indices]
er_bar[nonzero_indices] = np.sqrt(p * (1 - p) / n) * 100  # 百分率に戻す

# 0から24までの数字の配列を作成する
time = np.arange(24)

# エラーバーにNaNが含まれていないか確認
print("pro_ss:", pro_ss)
print("er_bar:", er_bar)

# エラー処理としてゼロでないデータを表示
ind1 = np.where(obs_ss_hist > 0)
print('detection;', obs_ss_hist[ind1])
print('Observation', all_ss_hist[ind1])

# グラフを描画
fig, axs = plt.subplots(dpi=500, figsize=(5, 5))  # DPIとサイズを調整
axs.set_title("(a) Ls 0°-180°", fontsize=20)
axs.set_xlabel("Local time [h]", fontsize=14)
axs.set_ylabel("Probability [%]", fontsize=14)
axs.set_xlim(0, 23)  # ヒストグラムの範囲に合わせる
#axs.bar(range(0, 24, 1), pro_ss, color='red', edgecolor='black', label='Probability')

# サンプル数が200以下の場合は、ビンの色を変更する
for i in range(0, 24, 1):
    if all_ss_hist[i] < 200:
        axs.bar(i, pro_ss[i], color='red', edgecolor='black', alpha=0.3)
    if all_ss_hist[i] > 200:
        axs.bar(i, pro_ss[i], color='red', edgecolor='black')

# エラーバーを描画（ゼロの部分はスキップ）
axs.errorbar(range(0, 24, 1), pro_ss, yerr=er_bar, color='black', ecolor='black', fmt='o', capsize=5, label='Error Bars')
axs.set_xlim(6, 18)
axs.set_ylim(0, 5)
axs.legend()
plt.show()

ss_data = np.stack((time,all_ss_hist, obs_ss_hist), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_probability.txt", ss_data)

# %%
# ------------------------------------- Figure 12b -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ALL_EW_detect_0-180_lon_lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW_detect_0-180_lon_lat.sav")
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
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=15))
sm.set_array([])

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("(b) Ls 0°-180°", fontsize=20)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_xlim(-185, 195)
ax.set_ylim(-100, 100)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))

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

# longitudeの値を-180°から180°に変換する
smooth_long = np.where(smooth_long > 180, smooth_long - 360, smooth_long)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(smooth_long[zero_indices], smooth_lat[zero_indices], color="grey", alpha=0.2, s=100, marker='s')
RATIO_map = (smooth_obs[non_zero_indices] / smooth_all[non_zero_indices]) * 100
colors = sm.to_rgba(RATIO_map.flatten())
ax.scatter(smooth_long[non_zero_indices], smooth_lat[non_zero_indices], color=colors,marker='s', s=100)

# カラーバーを作成
cbar = plt.colorbar(sm)
cbar.set_label("Probability of detection [%]", fontsize=14)
plt.show()

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_all-observation-spatial.txt", smooth_all)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_observation-spatial.txt", smooth_obs)

# ------------------------------------- Figure 13a -------------------------------------
# %%
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']

data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/coverage.sav")
# データを取り出す
ALL_ls = data['Ls_ind']
ALL_lt = data['LT_ind']
ALL_lat = data['lat_ind']

# Ls 180 - 360°のヒストグラムを作成する
all_autumn_winter = np.where((ALL_ls >= 180) & (ALL_ls < 360))
obs_autumn_winter = np.where((OBS_ls >= 180) & (OBS_ls < 360))

ALL_ls_aw = ALL_ls[all_autumn_winter]
ALL_lt_aw = ALL_lt[all_autumn_winter]
OBS_ls_aw = OBS_ls[obs_autumn_winter]
OBS_lt_aw = OBS_lt[obs_autumn_winter]

ls_obs_aw_hist = OBS_ls_aw.flatten()
ls_all_aw_hist = ALL_ls_aw.flatten()
lt_obs_aw_hist = OBS_lt_aw.flatten()
lt_all_aw_hist = ALL_lt_aw.flatten()

# からの配列を作成
obs_aw_hist = np.zeros(24)
all_aw_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    ind_aw_obs = np.where((OBS_lt_aw >= loop) & (OBS_lt_aw < loop+1))
    obs_aw_hist[loop] = len(OBS_lt_aw[ind_aw_obs])

    ind_aw_all = np.where((ALL_lt_aw >= loop) & (ALL_lt_aw < loop+1))
    all_aw_hist[loop] = len(ALL_lt_aw[ind_aw_all])

# all_aw_histがゼロでない場合のみ確率を計算する
pro_aw = np.zeros(24)
nonzero_indices = all_aw_hist > 0
pro_aw[nonzero_indices] = (obs_aw_hist[nonzero_indices] / all_aw_hist[nonzero_indices]) * 100

# エラーバーの計算（確率を0-1に変換）
er_bar = np.zeros(24)
p = pro_aw[nonzero_indices] / 100  # パーセンテージから確率に変換
n = all_aw_hist[nonzero_indices]
er_bar[nonzero_indices] = np.sqrt(p * (1 - p) / n) * 100  # 百分率に戻す

# エラーバーにNaNが含まれていないか確認
print("pro_aw:", pro_aw)
print("er_bar:", er_bar)

# エラー処理としてゼロでないデータを表示
ind1 = np.where(obs_aw_hist > 0)
print('detection;', obs_aw_hist[ind1])
print('Observation', all_aw_hist[ind1])

# 0から24の数字の配列を作成する
time = np.arange(24)

# グラフを描画
fig, axs = plt.subplots(dpi=800, figsize=(5, 5))  # DPIとサイズを調整
axs.set_title("(c) Ls 180°-360°", fontsize=20)
axs.set_xlabel("Local time [h]", fontsize=14)
axs.set_ylabel("Probability [%]", fontsize=14)
axs.set_xlim(0, 23)  # ヒストグラムの範囲に合わせる

# サンプル数が200以下の場合は、ビンの色を変更する
for i in range(0, 24, 1):
    if all_aw_hist[i] < 200:
        axs.bar(i, pro_aw[i], color='blue', edgecolor='black', alpha=0.3)
    if all_aw_hist[i] > 200:
        axs.bar(i, pro_aw[i], color='blue', edgecolor='black')

# エラーバーを描画（ゼロの部分はスキップ）
axs.errorbar(range(0, 24, 1), pro_aw, yerr=er_bar, color='black', ecolor='black', fmt='o', capsize=5, label='Error Bars')

axs.set_xlim(6, 18)
axs.set_ylim(-1, 20)
axs.legend()
plt.show()

aw_data = np.stack((time, all_aw_hist, obs_aw_hist), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_probability.txt", aw_data)

# %%    
# ------------------------------------- Figure 13b -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ALL_EW_detect_180-360_lon_lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW_detect_180-360_lon_lat.sav")
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
cmap = plt.get_cmap('Blues')
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=25))
sm.set_array([])

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("(d) Ls 180°-360°", fontsize=20)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_xlim(-185, 195)
ax.set_ylim(-100, 100)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))

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

# longitudeの値を-180°から180°に変換する
smooth_long = np.where(smooth_long > 180, smooth_long - 360, smooth_long)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(smooth_long[zero_indices], smooth_lat[zero_indices], color="grey", alpha=0.2, s=100, marker='s')
RATIO_map = (smooth_obs[non_zero_indices] / smooth_all[non_zero_indices]) * 100
colors = sm.to_rgba(RATIO_map.flatten())
ax.scatter(smooth_long[non_zero_indices], smooth_lat[non_zero_indices], color=colors,marker='s', s=100)

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/smooth_longitude.txt", smooth_long)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/smooth_latitude.txt", smooth_lat)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_all-observation-spatial.txt", smooth_all)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_detection-spatial.txt", smooth_obs)

# カラーバーを作成
cbar = plt.colorbar(sm)
cbar.set_label("Probability of detection [%]", fontsize=14)
plt.show()

# %%
