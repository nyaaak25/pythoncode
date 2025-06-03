# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
# Section 2のrevise version
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# %%
# -------------------------------------- Figure 7_0 --------------------------------------
# 横軸opacity, 縦軸stdの図を作成
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/evaluate_mathieu_points_v1_2.sav"  # revise version: 100pix平均
sav_data = readsav(sav_fname)

com_data = sav_data["corre_two"]

Akira_277_before = sav_data["mean_dust"] + 0
Akira_277 = Akira_277_before *1.12

Mathieu_slope = com_data[1,:] + 0

std_data_file = np.load("/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/dust_std.npz")
std = std_data_file["dust_tau_std"]

std[Akira_277 == 0] = np.nan
Mathieu_slope[Akira_277 == 0] = np.nan
Akira_277[Akira_277 == 0] = np.nan

plt.figure()
plt.scatter(Akira_277, std, s=25, facecolors='none', edgecolors='black', label="Our retrievals")
#plt.scatter(Mathieu_slope, std, s=25, facecolors='none', edgecolors='red', label="Vincendon et al. 2009 retrievals")
plt.xlabel("Dust opacity")
plt.ylabel("Standard deviation")
# %%
# -------------------------------------- Figure 7 --------------------------------------
# Mathieuとの図を作成する
# y =axの図を作成

data_dir = pjoin(dirname(sio.__file__), "tests", "data")
#sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/evaluate_mathieu_points.sav"
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/evaluate_mathieu_points_v1_2.sav"  # revise version: 100pix平均
sav_data = readsav(sav_fname)

com_data = sav_data["corre_two"]

#Akira_277_before = sav_data["mean_dust"] + 0
Akira_277_before = com_data[0,:] + 0
Akira_277 = Akira_277_before *1.12
Mathieu_slope = com_data[1,:] + 0

# 標準偏差のデータを読み込む
std_data_file = np.load("/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/dust_std.npz")
std = std_data_file["dust_tau_std"]

Mathieu_slope[std == 0] = np.nan
Akira_277[std == 0] = np.nan
std[std == 0] = np.nan

valid_indices = ~np.isnan(Mathieu_slope) & ~np.isnan(Akira_277) & ~np.isinf(Mathieu_slope) & ~np.isinf(Akira_277)
Mathieu_slope = Mathieu_slope[valid_indices]
Akira_277 = Akira_277[valid_indices]
std = std[valid_indices]

# 相関をとる
coef = np.polyfit(Mathieu_slope, Akira_277,1,w=1/std)
appr = np.poly1d(coef)(Mathieu_slope)

# 一次関数y=ax+bの係数と誤差を求める
p,cov = np.polyfit(Mathieu_slope, Akira_277,1,w=1/std,cov=True)
a = p[0]
b = p[1]
sigma_a = np.sqrt(cov[0,0])
sigma_b = np.sqrt(cov[1,1])
print(sigma_a,sigma_b,a,b)

# 強制的にy=axのfitをする
aaa, _, _, _ = np.linalg.lstsq(Mathieu_slope[:, np.newaxis], Akira_277, rcond=None)
a = aaa[0]
print(f'フィット係数 a = {aaa[0]:.4f}')

# zeroのところにnanを入れる
#Akira_277[Akira_277 == 0] = np.nan

# x=yの直線を引く
data = [np.min(Mathieu_slope), np.max(Mathieu_slope)]

# version 1 #
# 縦軸と横軸の値を入れ替えてプロット
fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Vincendon et al. 2009 retrievals", fontsize=10)
ax.set_ylabel("Our retrievals", fontsize=10)
ax.plot(Mathieu_slope, appr,  color = 'black', lw=1, zorder=1)
ax.errorbar(Mathieu_slope, Akira_277, yerr=std, fmt='o', color='black', markersize=5, capsize=3, elinewidth=1)
ax.scatter(Mathieu_slope, Akira_277, s=25, facecolors='none', edgecolors='black')
ax.plot(data, [a * x for x in data], color='red', lw=1, linestyle='dashed', label=f"y = {a:.2f}x")
ax.legend(loc='upper left', fontsize=10)

data_comparison = np.stack([Mathieu_slope, Akira_277, appr], axis=1)
data_comparison = data_comparison.T
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure7/data_comparison_revise.txt", data_comparison.T, delimiter=",")

# 標準偏差を計算
# フィット直線の値を計算
# y_fit = aaa * Mathieu_slope
y_fit = appr

# 残差（データとフィット直線の差）
residuals = Akira_277 - y_fit
std_dev = np.std(residuals, ddof=1) 

print(f'データの標準偏差 = {std_dev:.4f}')

# %%
