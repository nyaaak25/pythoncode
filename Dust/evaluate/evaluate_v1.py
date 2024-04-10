# %%

# MER SITEにおけるyannと2.77 μmの比較

from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt


# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/MER_site_dust.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
file_name = sav_data["file_name"]

# Yann result
yann_result = np.loadtxt(
    "/Users/nyonn/Desktop/yann_taudust_610Panormalised_for_MERsites_median_values.txt"
)

zero_ind = np.where(dust_277 == 0.0)
search_ind = np.where((dust_277 >= 0.0) & (dust_277 < 0.1) & (yann_result > 0.5))

ind_sza = [27, 32, 79, 5, 64, 60, 54, 32, 55, 17, 17, 50, 49, 48, 44, 69]
high_sza = np.where(np.array(ind_sza) >= 60)
non_mix = np.where(np.array(ind_sza) < 60)

ind = np.where(dust_277 > 0)
a, b = np.polyfit(dust_277, yann_result, 1)
cont0 = b + a * dust_277

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.set_xlabel("dust optical depth at akira", fontsize=10)
ax.set_ylabel("dust optical depth at yann", fontsize=10)
ax.scatter(dust_277, yann_result, color="black")

ax.scatter(
    dust_277[search_ind[0][non_mix]],
    yann_result[search_ind[0][non_mix]],
    color="blue",
    label="non-mixed",
)

ax.scatter(
    dust_277[search_ind[0][high_sza]],
    yann_result[search_ind[0][high_sza]],
    color="red",
    label="high sza",
)

ax.plot(dust_277, cont0, color="black")


"""
ax.scatter(
    dust_277[zero_ind[0][0:3]],
    yann_result[zero_ind[0][0:3]],
    color="red",
    label="high sza",
)
ax.scatter(
    dust_277[zero_ind[0][3:7]],
    yann_result[zero_ind[0][3:7]],
    color="blue",
    label="non-mixed",
)
"""

# ax.scatter(dust_277[near_zero_ind], yann_result[near_zero_ind], color="green")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper left", fontsize=8)


# 共分散を計算する
conv = np.cov(dust_277, yann_result)
# 出力は[xの分散、xyの共分散、yxの共分散、yの分散]

# 相関係数を計算する
correlate = np.corrcoef(dust_277, yann_result)
# 出力は[1, 相関係数、相関係数、1]

# %%
