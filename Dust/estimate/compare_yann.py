# ----------------------------------------------
# Compare the results of the Yann's result

# 20024.05.13 14:45
# created by Akira Kazama
# compare_yann.py
# Yannのデータと比較するためのプログラム

# ----------------------------------------------
# %%
# import library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.io import readsav

# %%
# --------------------------------------------------------------------------
# Load the data
# note: YannのデータはLsは5°刻み、Latは3°刻みで平均化されている
# ----- Yann data -----
data_27 = np.load('/Users/nyonn/Desktop/pythoncode/Dust/estimate/Yann_data/dust_optical_depth_yann_latitude_Ls_diagram_MY27.npz')
taudust_grid_27 = data_27["arr_0"]
Ls_grid_27 = data_27["arr_1"]
lat_grid_27 = data_27["arr_2"]

data_28 = np.load('/Users/nyonn/Desktop/pythoncode/Dust/estimate/Yann_data/dust_optical_depth_yann_latitude_Ls_diagram_MY28.npz')
taudust_grid_28 = data_28["arr_0"]
Ls_grid_28 = data_28["arr_1"]
lat_grid_28 = data_28["arr_2"]

data_29 = np.load('/Users/nyonn/Desktop/pythoncode/Dust/estimate/Yann_data/dust_optical_depth_yann_latitude_Ls_diagram_MY29.npz')
taudust_grid_29 = data_29["arr_0"]
Ls_grid_29 = data_29["arr_1"]
lat_grid_29 = data_29["arr_2"]

ip_27 = np.size(Ls_grid_27[0,:])
io_27 = np.size(Ls_grid_27[:,0])

ip_28 = np.size(Ls_grid_28[0,:])
io_28 = np.size(Ls_grid_28[:,0])

ip_29 = np.size(Ls_grid_29[0,:])
io_29 = np.size(Ls_grid_29[:,0])

# ----- 2.77 μｍ data -----
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/dust_seasonal.sav")
dust_color = all_data["dust_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(dust_color == 0)
dust_color = dust_color + 0
dust_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

Ls_array = np.arange(0, 360, 5)
Ls_array = Ls_array + 2.5

MY27_lat = lat_grid_27[:,0]
MY27_Ls = Ls_good[0:2046]
MY27_tau = dust_good[:,0,0:2046]

MY28_lat = lat_grid_28[:,0]
MY28_Ls = Ls_good[derease_indices_all[1]+1:derease_indices_all[2]]
MY28_tau = dust_good[:,0,derease_indices_all[1]+1:derease_indices_all[2]]

MY29_lat = lat_grid_29[:,0]
MY29_Ls = Ls_good[derease_indices_all[2]+1:]
MY29_tau = dust_good[:,0,derease_indices_all[2]+1:]

# create the array for the dust
dust_277_27 = np.zeros((io_27, ip_27))
dust_277_old_27 = np.zeros((len(lat_ind), ip_27))

dust_277_28 = np.zeros((io_28, ip_28))
dust_277_old_28 = np.zeros((len(lat_ind), ip_28))

dust_277_29 = np.zeros((io_29, ip_29))
dust_277_old_29 = np.zeros((len(lat_ind), ip_29))

for loop in range(len(Ls_array)):
        # MY27
        ind_ls_27 = np.where((MY27_Ls <= Ls_array[loop]+2.5) & (MY27_Ls >= Ls_array[loop]-2.5))[0]
        dust_277_old_27[:,loop] = np.nanmedian(MY27_tau[:,ind_ls_27], axis=1)

        # MY28
        ind_ls_28 = np.where((MY28_Ls <= Ls_array[loop]+2.5) & (MY28_Ls >= Ls_array[loop]-2.5))[0]
        dust_277_old_28[:,loop] = np.nanmedian(MY28_tau[:,ind_ls_28], axis=1)

        # MY29
        ind_ls_29 = np.where((MY29_Ls <= Ls_array[loop]+2.5) & (MY29_Ls >= Ls_array[loop]-2.5))[0]
        dust_277_old_29[:,loop] = np.nanmedian(MY29_tau[:,ind_ls_29], axis=1)

        for loop_lat in range(len(MY27_lat)):
            if loop_lat == 62:
                dev_27 = np.abs(MY27_lat[loop_lat] - 90)
                dev_28 = np.abs(MY28_lat[loop_lat] - 90)
                dev_29 = np.abs(MY29_lat[loop_lat] - 90)

            if loop_lat < 62:
                dev_27 = np.abs(MY27_lat[loop_lat + 1] - MY27_lat[loop_lat])
                dev_28 = np.abs(MY28_lat[loop_lat + 1] - MY28_lat[loop_lat])
                dev_29 = np.abs(MY29_lat[loop_lat + 1] - MY29_lat[loop_lat])
            
            # MY27
            ind_lat_27 = np.where((lat_ind <= MY27_lat[loop_lat] + dev_27) & (lat_ind >= MY27_lat[loop_lat] - dev_27))[0]
            dust_277_27[loop_lat,loop] = np.nanmedian(dust_277_old_27[ind_lat_27,loop])

            # MY28
            ind_lat_28 = np.where((lat_ind <= MY28_lat[loop_lat] + dev_28) & (lat_ind >= MY28_lat[loop_lat] - dev_28))[0]
            dust_277_28[loop_lat,loop] = np.nanmedian(dust_277_old_28[ind_lat_28,loop])

            # MY29
            ind_lat_29 = np.where((lat_ind <= MY29_lat[loop_lat] + dev_29) & (lat_ind >= MY29_lat[loop_lat] - dev_29))[0]
            dust_277_29[loop_lat,loop] = np.nanmedian(dust_277_old_29[ind_lat_29,loop])
                    

# ------ pure substract -----------------------
#sub_27 = taudust_grid_29 - dust_277_29
#sub_28 = taudust_grid_28 - dust_277_28
#sub_29 = taudust_grid_29 - dust_277_29
#medi = np.nanmedian(sub)
# ----------------------------------------------

# ------ ratio plot --------------
ratio_27 = taudust_grid_27 / dust_277_27
ratio_28 = taudust_grid_28 / dust_277_28
ratio_29 = taudust_grid_29 / dust_277_29

# color mapの設定
min_dust = 0
max_dust = 20

#min_dust = 0.1 - 0.01
#max_dust = 2.0 - 0.6
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_ylabel("Latitude [deg]", fontsize=10)
axs.set_ylim(-90, 90)
axs.set_xlim(0,360)

for i in range(0, ip_28, 1):
    color = sm.to_rgba(ratio_28[:,i])
    axs.scatter(Ls_grid_28[:,i], lat_grid_28[:,i], c=color)

cbar = plt.colorbar(sm,orientation='vertical',aspect=90)
cbar.set_label("tau at 0.9 μm / tau at 2.7 μm", fontsize=10)
# %%
# --------------------------------------------------------------------------
min_ls = 0
max_ls = 360
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_ls, vmax=max_ls)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# MY27
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_xlim(0,0.8)
axs.set_ylim(0,3.2)
axs.scatter(dust_277_27, taudust_grid_27,c=Ls_grid_27,cmap='jet',vmin=0,vmax=360)

cbar = plt.colorbar(sm2,orientation='vertical',aspect=90)
cbar.set_label("Ls variation", fontsize=10)

# MY28
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_xlim(0,0.8)
axs.set_ylim(0,3.2)
axs.scatter(dust_277_28, taudust_grid_28,c=Ls_grid_28,cmap='jet',vmin=0,vmax=360)

cbar = plt.colorbar(sm2,orientation='vertical',aspect=90)
cbar.set_label("Ls variation", fontsize=10)

# MY29
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_xlim(0,0.8)
axs.set_ylim(0,3.2)
axs.scatter(dust_277_29, taudust_grid_29,c=Ls_grid_29,cmap='jet',vmin=0,vmax=360)

cbar = plt.colorbar(sm2,orientation='vertical',aspect=90)
cbar.set_label("Ls variation", fontsize=10)
# --------------------------------------------------------------------------

# %%
# --------------------------------------------------------------------------
# ratio別にplotをしてみる
min_ratio = 1.0
max_ratio = 5.0

# zorfer
good = 2
low = 3
large = 1

# raiioが2.0-3.0のデータを取得
ind_ratio_good_27 = np.where((ratio_27 >= min_ratio) & (ratio_27 <= max_ratio))
ind_ratio_good_28 = np.where((ratio_28 >= min_ratio) & (ratio_28 <= max_ratio))
ind_ratio_good_29 = np.where((ratio_29 >= min_ratio) & (ratio_29 <= max_ratio))

# ratioが2.0以下のデータを取得
ind_ratio_bad_27 = np.where(ratio_27 <= min_ratio)
ind_ratio_bad_28 = np.where(ratio_28 <= min_ratio)
ind_ratio_bad_29 = np.where(ratio_29 <= min_ratio)

# ratioが3.0以上のデータを取得
ind_ratio_in_bad_27 = np.where(ratio_27 >= max_ratio)
ind_ratio_in_bad_28 = np.where(ratio_28 >= max_ratio)
ind_ratio_in_bad_29 = np.where(ratio_29 >= max_ratio)

# それぞれの色でプロット
# MY27
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_ylabel("Latitude [deg]", fontsize=10)
axs.set_ylim(-90, 90)
axs.set_xlim(0,360)
axs.scatter(Ls_grid_27[ind_ratio_good_27[0],ind_ratio_good_27[1]], lat_grid_27[ind_ratio_good_27[0],ind_ratio_good_27[1]], color="lightgreen",label="ratio = " +str(min_ratio) + " - " + str(max_ratio),zorder=good)
axs.scatter(Ls_grid_27[ind_ratio_in_bad_27[0],ind_ratio_in_bad_27[1]], lat_grid_27[ind_ratio_in_bad_27[0],ind_ratio_in_bad_27[1]], color="lightcoral",label="ratio = rather than " + str(max_ratio),zorder=large)
axs.scatter(Ls_grid_27[ind_ratio_bad_27[0],ind_ratio_bad_27[1]], lat_grid_27[ind_ratio_bad_27[0],ind_ratio_bad_27[1]], color="royalblue",label="ratio= less than " + str(min_ratio),zorder=low)
axs.legend(fontsize=7)

# MY28
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_ylabel("Latitude [deg]", fontsize=10)
axs.set_ylim(-90, 90)
axs.set_xlim(0,360)
axs.scatter(Ls_grid_28[ind_ratio_good_28[0],ind_ratio_good_28[1]], lat_grid_28[ind_ratio_good_28[0],ind_ratio_good_28[1]], color="lightgreen",label="ratio = " +str(min_ratio) + " - " + str(max_ratio),zorder=good)
axs.scatter(Ls_grid_28[ind_ratio_in_bad_28[0],ind_ratio_in_bad_28[1]], lat_grid_28[ind_ratio_in_bad_28[0],ind_ratio_in_bad_28[1]], color="lightcoral",label="ratio = rather than " + str(max_ratio),zorder=large)
axs.scatter(Ls_grid_28[ind_ratio_bad_28[0],ind_ratio_bad_28[1]], lat_grid_28[ind_ratio_bad_28[0],ind_ratio_bad_28[1]], color="royalblue",label="ratio= less than " + str(min_ratio),zorder=low)
axs.legend(fontsize=7)

# MY29
# プロットをする
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_ylabel("Latitude [deg]", fontsize=10)
axs.set_ylim(-90, 90)
axs.set_xlim(0,360)
axs.scatter(Ls_grid_29[ind_ratio_good_29[0],ind_ratio_good_29[1]], lat_grid_29[ind_ratio_good_29[0],ind_ratio_good_29[1]], color="lightgreen",label="ratio = " +str(min_ratio) + " - " + str(max_ratio),zorder=good)
axs.scatter(Ls_grid_29[ind_ratio_in_bad_29[0],ind_ratio_in_bad_29[1]], lat_grid_29[ind_ratio_in_bad_29[0],ind_ratio_in_bad_29[1]], color="lightcoral",label="ratio = rather than " + str(max_ratio),zorder=large)
axs.scatter(Ls_grid_29[ind_ratio_bad_29[0],ind_ratio_bad_29[1]], lat_grid_29[ind_ratio_bad_29[0],ind_ratio_bad_29[1]], color="royalblue",label="ratio= less than " + str(min_ratio),zorder=low)
axs.legend(fontsize=7)

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# %%
# ratioのヒストグラムを作成
# MY27
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(ratio_27.flatten(), bins=50, range=(0,10), color="lightskyblue",edgecolor="black")

# MY28
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(ratio_28.flatten(), bins=50, range=(0,10), color="lightskyblue",edgecolor="black")

# MY29
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(ratio_29.flatten(), bins=50, range=(0,10), color="lightskyblue",edgecolor="black")
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# %%
# MYを込みでLs別のヒストグラムを作成
bin_size = 30

set_min_ls = 250
set_max_ls = 270

# MY27
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_ls_27 = np.where((Ls_grid_27 >= set_min_ls) & (Ls_grid_27 <= set_max_ls))
axs.hist(ratio_27[ind_ls_27].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY28
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_ls_28 = np.where((Ls_grid_28 >= set_min_ls) & (Ls_grid_28 <= set_max_ls))
axs.hist(ratio_28[ind_ls_28].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY29
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_ls_29 = np.where((Ls_grid_29 >= set_min_ls) & (Ls_grid_29 <= set_max_ls))
axs.hist(ratio_29[ind_ls_29].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY27-29
data1 = ratio_27[ind_ls_27].flatten()
data2 = ratio_28[ind_ls_28].flatten()
data3 = ratio_29[ind_ls_29].flatten()
#all_data = np.concatenate([data1,data2,data3])
all_data = np.concatenate([data1,data3])

fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27 and MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(all_data, bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# %%
# Latitude ごとにプロットをしてみる
bin_size = 30
min_lat = -50
max_lat = 50

# MY27
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_lat_27 = np.where((lat_grid_27 >= min_lat) & (lat_grid_27 <= max_lat)) 
axs.hist(ratio_27[ind_lat_27].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY28
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_lat_28 = np.where((lat_grid_28 >= min_lat) & (lat_grid_28 <= max_lat))
axs.hist(ratio_28[ind_lat_28].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY29
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_lat_29 = np.where((lat_grid_29 >= min_lat) & (lat_grid_29 <= max_lat))
axs.hist(ratio_29[ind_lat_29].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY27-29
data1 = ratio_27[ind_lat_27].flatten()
data2 = ratio_28[ind_lat_28].flatten()
data3 = ratio_29[ind_lat_29].flatten()
all_data = np.concatenate([data1,data2,data3])

fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27 and MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(all_data, bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# %%
# LsとLatitudeを組み合わせてプロットをしてみる
bin_size = 30

min_Ls = 0
max_Ls = 360

min_lat = -40
max_lat = 0

# MY27
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_good_27 = np.where((lat_grid_27 >= min_lat) & (lat_grid_27 <= max_lat)&(Ls_grid_27 >= min_Ls) & (Ls_grid_27 <= max_Ls))
axs.hist(ratio_27[ind_good_27].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY28
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY28", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_good_28 = np.where((lat_grid_28 >= min_lat) & (lat_grid_28 <= max_lat)&(Ls_grid_28 >= min_Ls) & (Ls_grid_28 <= max_Ls))
axs.hist(ratio_28[ind_good_28].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY29
fig,axs = plt.subplots(dpi=300)
axs.set_title("MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
ind_good_29 = np.where((lat_grid_29 >= min_lat) & (lat_grid_29 <= max_lat)&(Ls_grid_29 >= min_Ls) & (Ls_grid_29 <= max_Ls))
axs.hist(ratio_29[ind_good_29].flatten(), bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")

# MY27-29
data1 = ratio_27[ind_good_27].flatten()
data2 = ratio_28[ind_good_28].flatten()
data3 = ratio_29[ind_good_29].flatten()
all_data = np.concatenate([data1,data2,data3])

fig,axs = plt.subplots(dpi=300)
axs.set_title("MY27 and MY29", fontsize=10)
axs.set_xlabel("Ratio", fontsize=10)
axs.set_ylabel("Frequency", fontsize=10)
axs.hist(all_data, bins=bin_size, range=(0,bin_size), color="lightskyblue",edgecolor="black")
# --------------------------------------------------------------------------

# %%
# --------------------------------------------------------------------------
# Ls別でプロットをしてみる
min_Ls = 272
max_Ls = 330

fig,axs = plt.subplots(dpi=300)
axs.set_title("Ls" + str(min_Ls) + "-" + str(max_Ls), fontsize=10)
axs.set_xlabel("Dust at 2.77 μm", fontsize=10)
axs.set_ylabel("Dust at 0.9 μm", fontsize=10)
axs.set_xlim(-0.01,0.8)
axs.set_ylim(-0.1,3.2)

ind27 = np.where((Ls_grid_27 >= min_Ls) & (Ls_grid_27 <= max_Ls))
ind28 = np.where((Ls_grid_28 >= min_Ls) & (Ls_grid_28 <= max_Ls))
ind29 = np.where((Ls_grid_29 >= min_Ls) & (Ls_grid_29 <= max_Ls))

axs.scatter(dust_277_27[ind27], taudust_grid_27[ind27],color="red",label="MY27")
axs.scatter(dust_277_28[ind28], taudust_grid_28[ind28],color="blue",label="MY28")
axs.scatter(dust_277_29[ind29], taudust_grid_29[ind29],color="green",label="MY29")

axs.legend()
# --------------------------------------------------------------------------
# %%
