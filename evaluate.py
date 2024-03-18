# %%

# MY27-29における3年分のevaluate用のplot

from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt


# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/MY27-29_+-30_277.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["mean_dust"]
Ls_ind = sav_data["Ls_ind"]
good_ind = np.where(Ls_ind > 0)

akira_ls = Ls_ind[good_ind]

MY27_akira_ls = akira_ls[0:1319]
MY28_akira_ls = akira_ls[1320:2074] + 360.0
MY29_akira_ls = akira_ls[2075:2795] + 720.0

akira_dust = dust_277[good_ind]

MY27_akira_dust = akira_dust[0:1319]
MY28_akira_dust = akira_dust[1320:2074]
MY29_akira_dust = akira_dust[2075:2795]

# Yann result
yanns_result_27 = np.loadtxt(
    "/Users/nyonn/Desktop/yann_taudust_midlatitudes30_median_values_MY27.txt"
)
MY27_yann_ls = yanns_result_27[:, 0]
MY27_yann_dust = yanns_result_27[:, 1]

yanns_result_28 = np.loadtxt(
    "/Users/nyonn/Desktop/yann_taudust_midlatitudes30_median_values_MY28.txt"
)
MY28_yann_ls = yanns_result_28[:, 0] + 360.0
MY28_yann_dust = yanns_result_28[:, 1]

yanns_result_29 = np.loadtxt(
    "/Users/nyonn/Desktop/yann_taudust_midlatitudes30_median_values_MY29.txt"
)
MY29_yann_ls = yanns_result_29[:, 0] + 720.0
MY29_yann_dust = yanns_result_29[:, 1]

# LS 10°刻み
Ls_grid = np.arange(0, 1080, 10)
yann_result = np.zeros(len(Ls_grid))
akira_result = np.zeros(len(Ls_grid))

for i in range(len(Ls_grid) - 1):
    if Ls_grid[i] <= 360:
        ind = np.where((MY27_yann_ls > Ls_grid[i]) & (MY27_yann_ls < Ls_grid[i + 1]))
        yann_result[i] = np.nanmedian(MY27_yann_dust[ind])

        ind2 = np.where((MY27_akira_ls > Ls_grid[i]) & (MY27_akira_ls < Ls_grid[i + 1]))
        akira_result[i] = np.nanmedian(MY27_akira_dust[ind2])

    elif Ls_grid[i] <= 720 and Ls_grid[i] > 360:
        ind = np.where((MY28_yann_ls > Ls_grid[i]) & (MY28_yann_ls < Ls_grid[i + 1]))
        yann_result[i] = np.nanmedian(MY28_yann_dust[ind])
        ind2 = np.where((MY28_akira_ls > Ls_grid[i]) & (MY28_akira_ls < Ls_grid[i + 1]))
        akira_result[i] = np.nanmedian(MY28_akira_dust[ind2])

    else:
        ind = np.where((MY29_yann_ls > Ls_grid[i]) & (MY29_yann_ls < Ls_grid[i + 1]))
        yann_result[i] = np.nanmedian(MY29_yann_dust[ind])

        ind2 = np.where((MY29_akira_ls > Ls_grid[i]) & (MY29_akira_ls < Ls_grid[i + 1]))
        akira_result[i] = np.nanmedian(MY29_akira_dust[ind2])

    if i == 84:
        print(MY29_yann_dust[ind])
        print(MY29_akira_dust[ind2])
        print("ind: ", np.size(ind))
        print("ind2: ", np.size(ind2))
        print("Before Ls: ", Ls_grid[i] - 720)
        print("after Ls:", Ls_grid[i + 1] - 720)
        print("Ls: ", MY29_akira_ls[ind2])


plt.scatter(akira_result, yann_result)
# %%

ind_valid = ~np.isnan(akira_result) & ~np.isnan(yann_result)
a, b = np.polyfit(akira_result[ind_valid], yann_result[ind_valid], 1)
cont0 = b + a * akira_result

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.set_xlabel("dust optical depth at akira", fontsize=10)
ax.set_ylabel("dust optical depth at yann", fontsize=10)
ax.scatter(akira_result, yann_result)
ax.plot(akira_result, cont0)

# %%
akira_ls_all = [MY27_akira_ls, MY28_akira_ls, MY29_akira_ls]
akira_dust_all = [MY27_akira_dust, MY28_akira_dust, MY29_akira_dust]

yann_ls_all = [MY27_yann_ls, MY28_yann_ls, MY29_yann_ls]
yann_dust_all = [MY27_yann_dust, MY28_yann_dust, MY29_yann_dust]

cont1 = a * np.concatenate(akira_dust_all)  # 修正：各akira_dustに対して線形回帰の結果を使用

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title="Comparison Yann to Akira")
ax.scatter(
    np.concatenate(yann_ls_all), np.concatenate(yann_dust_all), color="black", s=1
)
ax.scatter(np.concatenate(akira_ls_all), cont1, color="red", s=1)  # 修正：リストを展開

# %%
