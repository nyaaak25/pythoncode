# %%
# -*- coding: utf-8 -*-
"""
計算された光学的厚みから透過率を計算し、plotをする
"""

# import
import numpy as np
import matplotlib.pyplot as plt

"""
# 波数
v_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
v = v_txt[:, 0]

# 光学的厚み cut-offでの差を計算させている
tau_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
cut120 = tau_txt[:, 1]

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

error1 = (step01 - step0001) * 100 / step0001
error2 = (step001 - step0001) * 100 / step0001
error3 = (step0005 - step0001) * 100 / step0001


# %%
# ---------------グラフ作成----------------------
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)

# ----plot 変換--------
ax.plot(wav, error1, color='blue', label="0.01")
ax.plot(wav, error2, color='green', label="0.001")
ax.plot(wav, error3, color='red', label="0.0005")
# ax.plot(wav, step0001, color='orange', label="0.0001")

# ------ set axis ----------
# ax.set_xlim(4958.05, 4958.15)
# ax.set_xlim(4681.3, 4681.35)
# ax.set_xlim(4800, 5100)
# ax.set_ylim(1e-5, 1.8e-5)
# ax.set_ylim(0.3, 1.05)

# ----- set label ------
# ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
ax.set_xlabel('Wavelengh [um]', fontsize=14)
# ax.set_ylabel('error (%)', fontsize=14)
ax.set_ylabel('Defference [%]', fontsize=14)
# ax.set_ylabel('Transmittance', fontsize=14)


# 凡例
# ax.set_yscale('log')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='upper right', fontsize=6)
plt.show()

# %%
