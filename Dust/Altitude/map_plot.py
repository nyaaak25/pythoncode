# --------------------------------------------------
# 2025.05.21 created by AKira Kazama
# ダスト光学的厚さのマップを作成する
# 2.77 μm, 2.01 μm, その差のマップを作成する
# --------------------------------------------------
# %%
# import modules
import numpy as np
import matplotlib.pyplot as plt

# -------- set parameter ----------
martian_year = 27

# How to method
method = 2
if method == 1:
        method_name = 'ratio'
elif method == 2:
        method_name = 'difference'
# 1: ratio, 2: difference

# set the color bar
set_min_ratio = -1 #-1
set_max_ratio = 1 #1

# latitude grid
lat_grid = 1
# ----------------------------------
# .npz dataを読み込む
path_data = '/Users/nyonn/Desktop/pythoncode/Dust/Altitude/data/'
data = np.load(path_data + f"MY{martian_year}_grid{lat_grid}_method_{method_name}.npz")

data_277 = data['sav_277_data']
data_201 = data['sav_201_data']
data_diff = data['sav_method_data']
lat_arr = data['array_lat']

# LS dataを読み込む
path_yann = '/Users/nyonn/Desktop/pythoncode/Dust/estimate/Yann_data/'
orbfile_data = np.loadtxt(path_yann + 'OMEGA_obsname_Ls_values_2004_2010.ref',dtype=np.str)
ls_data = orbfile_data[:,1].astype(float)
derease_ind = np.where(np.diff(ls_data) < 0)[0]

if martian_year == 27:
        ls_data = ls_data[derease_ind[0]+1:derease_ind[2]+1]
        index_alp = '(a)'

if martian_year == 28:
        ls_data = ls_data[derease_ind[2]+1:derease_ind[3]+1]
        index_alp = '(b)'

if martian_year == 29:
        ls_data = ls_data[derease_ind[3]+1:derease_ind[4]+1]
        index_alp = '(c)'

# data = 0のところをnp.nanに置き換える
data_277[data_277 == 0] = np.nan
data_201[data_201 == 0] = np.nan
data_diff[data_diff == 0] = np.nan

# data_201 = 3のところは全てのデータをnanに置き換える
bad_index = np.where(data_201 >= 2.9)
data_277[bad_index] = np.nan
data_201[bad_index] = np.nan
data_diff[bad_index] = np.nan

# plotの枠組を指定する
#fig,axs = plt.subplots(dpi=300, figsize=(9, 3))
fig,axs = plt.subplots(dpi=300)
axs.set_title(index_alp + " MY" + str(martian_year), fontsize=20)
#axs.set_xlabel("Ls [Deg]", fontsize=18)
axs.set_ylabel("Latitude [Deg]", fontsize=18)
axs.set_xlim(0, 360)
axs.set_ylim(-90, 90)
axs.set_xticks(np.arange(240, 361, 60))
axs.set_yticks(np.arange(-90, 91, 30))
axs.set_xticklabels(np.arange(240, 361, 60), fontsize=12)
axs.set_yticklabels(np.arange(-90, 91, 30), fontsize=12)

for i in range(0,np.size(ls_data),1):
        # ls_dataの配列をlat_arrの配列に合わせる
        Ls = np.full((np.size(lat_arr)), ls_data[i])
        im = axs.scatter(Ls,lat_arr, c=data_diff[i,:], cmap="jet", vmin=set_min_ratio, vmax=set_max_ratio,s=3)

# y = 0のラインを引く
#axs.axhline(0, color='black', linewidth=0.5, linestyle='--')
#axs.axhline(5, color='black', linewidth=0.5, linestyle='--')
cbar = fig.colorbar(im)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label='2.77 um - 2.01 um [tau]', fontsize=14)
#cbar.set_label(label='Dust Optical Depth [tau]', fontsize=14)
cbar.set_ticks(np.arange(set_min_ratio, set_max_ratio+0.1, 0.5), fontsize=8) 

# %%
