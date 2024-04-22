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

# %%
# ------------------------------------- Figure 4.1 -------------------------------------
# Local dust stormを検出したファイルを読み込む
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

# color mapの設定
min_dust = 0.001
max_dust = 0.6
cmap = plt.get_cmap('jet')

fig, axs = plt.subplots(4,1,figsize=(10,20),dpi=800,sharex=True,sharey=True)
# color mapの設定
norm = Normalize(vmin=min_dust, vmax=max_dust)
# Create a ScalarMappable with the colormap and normalization
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

#ax1 = fig.add_subplot(411)
axs[0].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[0].set_ylabel("Latitude [Deg]", fontsize=14)
axs[0].set_title("MY27-29", fontsize=14)
axs[0].set_xlim(0, 360)
axs[0].set_ylim(-90, 90)

# MY27は0から81まで
# MY28は81から118まで
# MY29は118から最後まで

# 上のファイルを一つずつ読み込んでいく
for i in range(count):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 0.2)
    bin_lat = bin_lat[LDS_ind]
    bin_lon = bin_lon[LDS_ind]
    min_lat = np.min(bin_lat)
    max_lat = np.max(bin_lat)
    min_lon = np.min(bin_lon)
    max_lon = np.max(bin_lon)

    bin_med = bin_med[LDS_ind]

    # color mapの設定
    norm = Normalize(vmin=min_dust, vmax=max_dust)
    # Create a ScalarMappable with the colormap and normalization
    sm = ScalarMappable(cmap=cmap, norm=norm)
    # Use the ScalarMappable to get the color
    color = sm.to_rgba(np.median(bin_med))

    axs[0].plot([Ls,Ls],[min_lat,max_lat], color=color, linewidth=4)
    #axs[0].plot(Ls,[min_lat,max_lat], color=color, linewidth=4)


#ax2 = fig.add_subplot(412)
axs[1].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[1].set_ylabel("Latitude [Deg]", fontsize=14)
axs[1].set_title("MY27", fontsize=14)
axs[1].set_xlim(0, 360)
axs[1].set_ylim(-90, 90)

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
    bin_lon = bin_lon[LDS_ind]
    min_lat = np.min(bin_lat)
    max_lat = np.max(bin_lat)
    min_lon = np.min(bin_lon)
    max_lon = np.max(bin_lon)

    bin_med = bin_med[LDS_ind]

    # Use the ScalarMappable to get the color
    color = sm.to_rgba(np.median(bin_med))

    axs[1].plot([Ls,Ls],[min_lat,max_lat], color=color, linewidth=4)

# MY28
#ax3 = fig.add_subplot(413)
axs[2].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[2].set_ylabel("Latitude [Deg]", fontsize=14)
axs[2].set_title("MY28", fontsize=14)
axs[2].set_xlim(0, 360)
axs[2].set_ylim(-90, 90)

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
    bin_lon = bin_lon[LDS_ind]
    min_lat = np.min(bin_lat)
    max_lat = np.max(bin_lat)
    min_lon = np.min(bin_lon)
    max_lon = np.max(bin_lon)

    bin_med = bin_med[LDS_ind]

    color = sm.to_rgba(np.median(bin_med))

    axs[2].plot([Ls,Ls],[min_lat,max_lat], color=color, linewidth=4)

# MY29
#ax4 = fig.add_subplot(414)
axs[3].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[3].set_ylabel("Latitude [Deg]", fontsize=14)
axs[3].set_title("MY29", fontsize=14)
axs[3].set_xlim(0, 360)
axs[3].set_ylim(-90, 90)

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
    bin_lon = bin_lon[LDS_ind]
    min_lat = np.min(bin_lat)
    max_lat = np.max(bin_lat)
    min_lon = np.min(bin_lon)
    max_lon = np.max(bin_lon)

    bin_med = bin_med[LDS_ind]

    color = sm.to_rgba(np.median(bin_med))

    axs[3].plot([Ls,Ls],[min_lat,max_lat], color=color, linewidth=4)

# カラーバーを作成
cbar = plt.colorbar(sm, ax=axs.ravel().tolist(),orientation='vertical',aspect=90)
cbar.set_label("Dust optical depth", fontsize=14)

plt.show()

"""
# fileorbit, all_min_lon, all_max_lon, all_min_lat, all_max_latを1つのtxt fileで保存する
# 保存するデータを作成する
data = np.vstack((fileorbit, all_min_lon, all_max_lon, all_min_lat, all_max_lat))
data = data.T
# 保存するデータの名前を設定する
filename = "/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW_detect_info.txt"
# ファイルを保存する
np.savetxt(filename, data, fmt='%s')
"""

# %%
# ------------------------------------- Figure 4.2 -------------------------------------
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
ALL_ls = data['all_Ls_ind']
OBS_lt = data['obs_lt_ind']
ALL_lt = data['all_LT_ind']

# MYごとのデータを取り出す
# MY27,MT28,MY29の境目を探す
decrease_indices_all = np.where(np.diff(ALL_ls) < 0)[0]
decrease_indices_obs = np.where(np.diff(OBS_ls) < 0)[0]

# MY27
MY27_all_ls = ALL_ls[0:decrease_indices_all[0]+1]
MY27_obs_ls = OBS_ls[0:decrease_indices_obs[0]+1]
MY27_obs_lt = OBS_lt[0:decrease_indices_obs[0]+1]
MY27_all_lt = ALL_lt[0:decrease_indices_all[0]+1]

# MY28
MY28_all_ls = ALL_ls[decrease_indices_all[0]+1:decrease_indices_all[1]+1]
MY28_obs_ls = OBS_ls[decrease_indices_obs[0]+1:decrease_indices_obs[1]+1]
MY28_obs_lt = OBS_lt[decrease_indices_obs[0]+1:decrease_indices_obs[1]+1]
MY28_all_lt = ALL_lt[decrease_indices_all[0]+1:decrease_indices_all[1]+1]

# MY29
MY29_all_ls = ALL_ls[decrease_indices_all[1]+1:]
MY29_obs_ls = OBS_ls[decrease_indices_obs[1]+1:]
MY29_obs_lt = OBS_lt[decrease_indices_obs[1]+1:]
MY29_all_lt = ALL_lt[decrease_indices_all[1]+1:]

# ヒストグラムを作成
# ALL periond in M27-29
ls_obs_hist = OBS_ls.flatten()
ls_all_hist = ALL_ls.flatten()
lt_obs_hist = OBS_lt.flatten()
lt_all_hist = ALL_lt.flatten()

# MY27
MY27_ls_obs_hist = MY27_obs_ls.flatten()
MY27_ls_all_hist = MY27_all_ls.flatten()
MY27_lt_obs_hist = MY27_obs_lt.flatten()
MY27_lt_all_hist = MY27_all_lt.flatten()

# MY28
MY28_ls_obs_hist = MY28_obs_ls.flatten()
MY28_ls_all_hist = MY28_all_ls.flatten()
MY28_lt_obs_hist = MY28_obs_lt.flatten()
MY28_lt_all_hist = MY28_all_lt.flatten()

# MY29
MY29_ls_obs_hist = MY29_obs_ls.flatten()
MY29_ls_all_hist = MY29_all_ls.flatten()
MY29_lt_obs_hist = MY29_obs_lt.flatten()
MY29_lt_all_hist = MY29_all_lt.flatten()

# ヒストグラムを作成
# ALL periond histogram
# LSのヒストグラム
fig, axs = plt.subplots(4,1,figsize=(5,20),dpi=800,sharex=True,sharey=True)
# MY27-29
axs[0].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[0].set_ylabel("Number of observations", fontsize=14)
axs[0].set_title("MY27-29", fontsize=14)
axs[0].set_xlim(0, 360)
axs[0].hist([ls_obs_hist,ls_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
axs[0].legend()

# MY27
axs[1].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[1].set_ylabel("Number of observations", fontsize=14)
axs[1].set_title("MY27", fontsize=14)
axs[1].set_xlim(0, 360)
axs[1].hist([MY27_ls_obs_hist,MY27_ls_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
axs[1].legend()

# MY28
axs[2].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[2].set_ylabel("Number of observations", fontsize=14)
axs[2].set_title("MY28", fontsize=14)
axs[2].set_xlim(0, 360)
axs[2].hist([MY28_ls_obs_hist,MY28_ls_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
axs[2].legend()

# MY29
axs[3].set_xlabel("Solar longitude [Deg]", fontsize=14)
axs[3].set_ylabel("Number of observations", fontsize=14)
axs[3].set_title("MY29", fontsize=14)
axs[3].set_xlim(0, 360)
axs[3].hist([MY29_ls_obs_hist,MY29_ls_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
axs[3].legend()

# LTのヒストグラム
fig2, axs2 = plt.subplots(2,2,figsize=(10,10),dpi=800,sharex=True,sharey=True)
# MY27-29
axs2[0,0].set_xlabel("Local time [h]", fontsize=14)
axs2[0,0].set_ylabel("Number of observations", fontsize=14)
axs2[0,0].set_title("MY27-29", fontsize=14)
axs2[0,0].set_xlim(0, 24)
axs2[0,0].hist([lt_obs_hist,lt_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
h1, l1 = axs2[0,0].get_legend_handles_labels()
axs2[0,0].legend(h1, l1, loc="upper left", fontsize=9)

# MY27
axs2[0,1].set_xlabel("Local time [h]", fontsize=14)
axs2[0,1].set_ylabel("Number of observations", fontsize=14)
axs2[0,1].set_title("MY27", fontsize=14)
axs2[0,1].set_xlim(0, 24)
axs2[0,1].hist([MY27_lt_obs_hist,MY27_lt_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
h1, l1 = axs2[0,1].get_legend_handles_labels()
axs2[0,1].legend(h1, l1, loc="upper left", fontsize=9)

# MY28
axs2[1,0].set_xlabel("Local time [h]", fontsize=14)
axs2[1,0].set_ylabel("Number of observations", fontsize=14)
axs2[1,0].set_title("MY28", fontsize=14)
axs2[1,0].set_xlim(0, 24)
axs2[1,0].hist([MY28_lt_obs_hist,MY28_lt_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
h1, l1 = axs2[1,0].get_legend_handles_labels()
axs2[1,0].legend(h1, l1, loc="upper left", fontsize=9)

# MY29
axs2[1,1].set_xlabel("Local time [h]", fontsize=14)
axs2[1,1].set_ylabel("Number of observations", fontsize=14)
axs2[1,1].set_title("MY29", fontsize=14)
axs2[1,1].set_xlim(0, 24)
axs2[1,1].hist([MY29_lt_obs_hist,MY29_lt_all_hist],bins=10,histtype='bar',label=['Detect','All observation'],color=['blue','red'],edgecolor='black')
h1, l1 = axs2[1,1].get_legend_handles_labels()
axs2[1,1].legend(h1, l1, loc="upper left", fontsize=9)

plt.show()

# %%
# ------------------------------------- Figure 4.3 -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/ALL_EW_detect_lon-lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/4章：Analysis of OMEGA data set/data/EW_detect_lon-lat.sav")
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
cmap = plt.get_cmap('Purples')

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)

# ALL_infoが0の場所と0でない場所を特定します
zero_indices = np.where(ALL_info == 0)
non_zero_indices = np.where(ALL_info != 0)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(Longi[zero_indices], Lati[zero_indices], color="lavender", alpha=0.2)

# ALL_infoが0でない場所を計算し、それをプロットします
RATIO_map = (OBS_info[non_zero_indices] / ALL_info[non_zero_indices]) * 100
colors = cmap(RATIO_map / 30)
ax.scatter(Longi[non_zero_indices], Lati[non_zero_indices], color=colors)

# カラーバーを作成
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=30))
sm.set_array([])
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
