# %%
# -*- coding: utf-8 -*-
"""
計算された光学的厚みから透過率を計算し、plotをする
"""

# import
import numpy as np
import matplotlib.pyplot as plt


# 波数
v_txt = np.loadtxt('test1_trans_mono.dat')
v1 = v_txt[:, 0]
cm_v = (1/v1)*10000
v = cm_v[::-1]

# 光学的厚み cut-offでの差を計算させている
# In = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
tau_txt = np.loadtxt('Tau_file/LUtable_1_tau.txt')
cut120_1 = tau_txt[:, 1]
cut120 = cut120_1[::-1]
# sza_theta = 18.986036
sza_theta = 0
theta = 0
I0 = np.exp(-cut120/np.cos(sza_theta))
Iobs = I0 * np.exp(-cut120)

# ARSで計算を走らせたものをInput
ARS_A = np.loadtxt('test1_trans_mono.dat')
In_1 = ARS_A[:, 1]
ARS_T = In_1[::-1]

"""
tau_txt1 = np.loadtxt('4545-5556_0.01step_cutoff_80.txt')
cut80 = tau_txt1[:, 1]

tau_txt2 = np.loadtxt('4445-5656_0.01step.txt')
Nocut = tau_txt2[10000:111100, 1]

tau_txt3 = np.loadtxt('4545-5556_0.01step_cutoff.txt')
cut100 = tau_txt3[:, 1]

tau_txt4 = np.loadtxt('4545-5556_0.01step_cutoff_50.txt')
cut50 = tau_txt4[:, 1]

# 透過率
A = np.exp(-cut120)

# 相対誤差など、、
# DDD = (τ-tau)*100/tau     #normalaizeされているよ

Cut100 = Nocut-cut100
Cut120 = Nocut-cut120
Cut80 = Nocut-cut80
Cut50 = Nocut-cut50

NCut100 = Cut100/Nocut

x1 = v
y1 = Nocut
"""
"""
# 装置関数の差を取る
wav1 = np.loadtxt('0.0005_OMEGAinstrumentfunction.txt')
wav = wav1[:, 0]

# 0.01のとき
step01_txt = np.loadtxt('0.01_OMEGAinstrumentfunction.txt')
step01 = step01_txt[:, 1]

# 0.001のとき
step001_txt = np.loadtxt('0.001_OMEGAinstrumentfunction.txt')
step001 = step001_txt[:, 1]

# 0.0005のとき
step0005_txt = np.loadtxt('0.0005_OMEGAinstrumentfunction.txt')
step0005 = step0005_txt[:, 1]

# 0.0001のとき
step0001_txt = np.loadtxt('0.0001_OMEGAinstrumentfunction.txt')
step0001 = step0001_txt[:, 1]

# BOX関数の平均
# 0.01のとき
box01_txt = np.loadtxt('0.01_BOXinstrumentfunction.txt')
box01 = box01_txt[:, 1]

# 0.001のとき
box001_txt = np.loadtxt('0.001_BOXinstrumentfunction.txt')
box001 = box001_txt[:, 1]

# 0.0005のとき
box0005_txt = np.loadtxt('0.0005_BOXinstrumentfunction.txt')
box0005 = box0005_txt[:, 1]

# 0.0001のとき
box0001_txt = np.loadtxt('0.0001_BOXinstrumentfunction.txt')
box0001 = box0001_txt[:, 1]

# 標準偏差の違うGAUSSIAN
gauss001_txt = np.loadtxt('instrumentfunction_001.txt')
gauss001 = gauss001_txt[:, 1]

gauss100_txt = np.loadtxt('instrumentfunction_100.txt')
gauss100 = gauss100_txt[:, 1]

error1 = (step01 - step0001) * 100 / step0001
error2 = (step001 - step0001) * 100 / step0001
error3 = (step0005 - step0001) * 100 / step0001

error4 = (step0001 - box0001)  # * 100 / box0001
error5 = (step001 - box001)  # * 100 / box001
error6 = (step0005 - box0005)  # * 100 / box0005
error7 = (step01 - box01) * 100 / box01

error10 = (box01 - gauss001) * 100 / box01
error11 = (box01 - gauss100) * 100 / box01

"""

error = ARS_T - Iobs
# %%
# ---------------グラフ作成----------------------
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)

# ----plot 変換--------
# zorderで表示順が決められる。値が大きいほど前面に出てくる。lwはplotの線の太さが変更可能。
ax.plot(v, ARS_T, color='blue', label="difference", zorder=2, lw=0.1)
ax.plot(v, Iobs, color='red', zorder=1, label="0.01", lw=0.1)
# ax.plot(wav, error10, color='green', zorder=1, label="box 0.01")
# ax.plot(wav, error11, color='orange', zorder=1,label="box 100")
# ax.scatter(wav, step01, color='red', zorder=2)
# ax.plot(wav, step0001, color='orange', label="0.0001")

# ------ set axis ----------
# ax.set_xlim(4958.05, 4958.15)
# ax.set_xlim(4681.3, 4681.35)
# ax.set_xlim(2.0, 2.005)
# ax.set_ylim(1e-5, 1.8e-5)
# ax.set_ylim(0.3, 1.05)

# ----- set label ------
# ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
ax.set_xlabel('Wavelengh [μm]', fontsize=14)
# ax.set_ylabel('Difference', fontsize=14)
# ax.set_ylabel('Radiance', fontsize=14)
ax.set_ylabel('Transmittance', fontsize=14)


# 凡例
# ax.set_yscale('log')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=6)
plt.show()

# %%

# %%
