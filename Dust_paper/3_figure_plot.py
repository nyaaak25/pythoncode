# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio

# %%
# FIgure 3.2e MERとの比較をplotしたもの
# 温度の不確定性
# Lander siteに着目したもの

# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MER_site_dust_nan.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
akira_result = dust_277 + 0
file_name = sav_data["file_name"]

# Yann result
yann_result = np.loadtxt("/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/tauMER_norm610Pa_for_MERsites.txt")

xerr = np.zeros(np.size(akira_result))

coef = np.polyfit(akira_result, yann_result, 1)
cont0 = np.poly1d(coef)(akira_result)

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 0.91

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.035

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.022

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(a) Temperature uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using MER", fontsize=10)
ax.errorbar(akira_result, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %% 
# FIgure 3.2f MERとの比較をplotしたもの
# 圧力の不確定性
# Lander siteに着目したもの
# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MER_site_dust_nan.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
akira_result = dust_277 + 0
file_name = sav_data["file_name"]

# Yann result
MER_result = np.loadtxt("/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/tauMER_norm610Pa_for_MERsites.txt")

xerr = np.zeros(np.size(akira_result))

coef = np.polyfit(akira_result, MER_result, 1)
cont0 = np.poly1d(coef)(akira_result)

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 2.03

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.21

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.082

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(b) Surface pressure uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using MER", fontsize=10)
ax.errorbar(akira_result, MER_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %%
# FIgure 3.2c Yannとの比較をplotしたもの
# 温度の不確定性
# Lander siteに着目したもの

# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MER_site_dust_nan.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
akira_result = dust_277 + 0
file_name = sav_data["file_name"]

# Yann result
yann_result = np.loadtxt("/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_610Panormalised_for_MERsites_median_values.txt")

xerr = np.zeros(np.size(akira_result))

coef = np.polyfit(akira_result, yann_result, 1)
cont0 = np.poly1d(coef)(akira_result)

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 0.91

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.035

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.022

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(a) Temperature uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using 2.0 μm", fontsize=10)
ax.errorbar(akira_result, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %%
# FIgure 3.2d Yannとの比較をplotしたもの
# 圧力の不確定性
# Lander siteに着目したもの

# read the retrieval result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MER_site_dust_nan.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
akira_result = dust_277 + 0
file_name = sav_data["file_name"]

# Yann result
yann_result = np.loadtxt(
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_610Panormalised_for_MERsites_median_values.txt"
)

xerr = np.zeros(np.size(akira_result))

coef = np.polyfit(akira_result, yann_result, 1)
cont0 = np.poly1d(coef)(akira_result)

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 2.03

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.21

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.082

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(b) Surface pressure uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using 2.0 μm", fontsize=10)
ax.errorbar(akira_result, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %%
# Figure 3.2aのYannとの比較をplotしてもの
# 温度の不確定性
# MY27-29、3年分をLs 10°刻みで平均を取って、plotをしてみる
# ±30°領域に焦点を当てて解析を行った

# 2.77 μmのretreival result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MY27-29_+-30_277.sav"
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
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY27.txt"
)
MY27_yann_ls = yanns_result_27[:, 0]
MY27_yann_dust = yanns_result_27[:, 1]

yanns_result_28 = np.loadtxt(
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY28.txt"
)
MY28_yann_ls = yanns_result_28[:, 0] + 360.0
MY28_yann_dust = yanns_result_28[:, 1]

yanns_result_29 = np.loadtxt(
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY29.txt"
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

# 相関係数を書く
ind_valid = ~np.isnan(akira_result) & ~np.isnan(yann_result)
coef = np.polyfit(akira_result[ind_valid], yann_result[ind_valid], 1)
cont0 = np.poly1d(coef)(akira_result)

xerr = np.zeros(np.size(akira_result))

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 0.91

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.035

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.022

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(a) Temperature uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using 2.0 μm", fontsize=10)
ax.errorbar(akira_result, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %%
# Figure 3.2bのYannとの3年分の比較
# 表面圧力の比較
# 2.77 μmのretreival result
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/MY27-29_+-30_277.sav"
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
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY27.txt"
)
MY27_yann_ls = yanns_result_27[:, 0]
MY27_yann_dust = yanns_result_27[:, 1]

yanns_result_28 = np.loadtxt(
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY28.txt"
)
MY28_yann_ls = yanns_result_28[:, 0] + 360.0
MY28_yann_dust = yanns_result_28[:, 1]

yanns_result_29 = np.loadtxt(
    "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/yann_taudust_midlatitudes30_median_values_MY29.txt"
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

# 相関係数を書く
ind_valid = ~np.isnan(akira_result) & ~np.isnan(yann_result)
coef = np.polyfit(akira_result[ind_valid], yann_result[ind_valid], 1)
cont0 = np.poly1d(coef)(akira_result)

xerr = np.zeros(np.size(akira_result))

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = akira_result[ind1] * 2.03

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((akira_result < 0.5) & (akira_result >= 0.01))
xerr[ind2] = akira_result[ind2] * 0.21

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(akira_result >= 0.5)
xerr[ind3] = akira_result[ind3] * 0.082

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(b) Surface pressure uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using 2.0 μm", fontsize=10)
ax.errorbar(akira_result, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)

# %%
# Figure 3.1aのMathieuとの比較をplotするもの
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/evaluate_mathieu_points.sav"
sav_data = readsav(sav_fname)

com_data = sav_data["corre_two"]
Akira_277 = com_data[0,:] + 0
Mathieu_slope = com_data[1,:]

xerr = np.zeros(np.size(Akira_277))

# ±10Kの不確定性
# (1)では+10 Kのときに最大で91%の差が生じる
ind1 = np.where(Akira_277 < 0.01)
Akira_277[Akira_277 == 0] = 0.000000000001
xerr[ind1] = Akira_277[ind1] * 0.91

# (2)では-10 Kのときに最大で3.5%の差が生じる
ind2 = np.where((Akira_277 < 0.5) & (Akira_277 >= 0.01))
xerr[ind2] = Akira_277[ind2] * 0.035

# (3)では-10 Kのときに最大で-2.2%の差が生じる
ind3 = np.where(Akira_277 >= 0.5)
xerr[ind3] = Akira_277[ind3] * 0.022

# 相関をとる
coef=np.polyfit(Akira_277, Mathieu_slope, 1)
appr = np.poly1d(coef)(Akira_277)


fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(a) Temperature uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using slope", fontsize=10)
ax.errorbar(Akira_277, Mathieu_slope, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(Akira_277, appr,  color = 'black', lw=0.5, zorder=1)

# %%
# Figure 3.1bのMathieuとの比較をplotするもの
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/3章：Evaluate the method/data/evaluate_mathieu_points.sav"
sav_data = readsav(sav_fname)

com_data = sav_data["corre_two"]
Akira_277 = com_data[0,:] + 0
Mathieu_slope = com_data[1,:]

xerr = np.zeros(np.size(Akira_277))

# ±10Kの不確定性
# (1)では+10 Kのときに最大で203%の差が生じる
ind1 = np.where(Akira_277 < 0.01)
Akira_277[Akira_277 == 0] = 0.000000000001
xerr[ind1] = Akira_277[ind1] * 2.03

# (2)では-10 Kのときに最大で21%の差が生じる
ind2 = np.where((Akira_277 < 0.5) & (Akira_277 >= 0.01))
xerr[ind2] = Akira_277[ind2] * 0.21

# (3)では-10 Kのときに最大で-8.2%の差が生じる
ind3 = np.where(Akira_277 >= 0.5)
xerr[ind3] = Akira_277[ind3] * 0.082

# 相関をとる
coef=np.polyfit(Akira_277, Mathieu_slope, 1)
appr = np.poly1d(coef)(Akira_277)

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(b) Surface pressure uncertainties")
ax.set_xlabel("Dust optical depth at 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth using slope", fontsize=10)
ax.errorbar(Akira_277, Mathieu_slope, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(Akira_277, appr,  color = 'black', lw=0.5, zorder=1)
# %%
