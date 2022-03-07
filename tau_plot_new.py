# %%
# -*- coding: utf-8 -*-
"""
計算された光学的厚みから透過率を計算し、plotをする
"""

# import
import numpy as np
import matplotlib.pyplot as plt


# 波数
v_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
v = v_txt[:, 0]

# 光学的厚み
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

# %%
# ---------------グラフ作成----------------------
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)

# ----plot 変換--------
ax.plot(x1, Cut120, color='blue', label="cutoff 120")
ax.plot(x1, Cut100, color='green', label="cutoff 100")
ax.plot(x1, Cut80, color='red', label="cutoff 80")
ax.plot(x1, Cut50, color='orange', label="cutoff 50")

# ------ set axis ----------
# ax.set_xlim(4958.05, 4958.15)
# ax.set_xlim(4681.3, 4681.35)
ax.set_xlim(4800, 5100)
# ax.set_ylim(1e-5, 1.8e-5)
# ax.set_ylim(-0.1, 0.1)

# ----- set label ------
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
# ax.set_ylabel('error (%)', fontsize=14)
ax.set_ylabel('Defference[%]', fontsize=14)
# ax.set_ylabel('Transmittance', fontsize=14)


# 凡例
# ax.set_yscale('log')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='upper right', fontsize=6)
plt.show()

# %%
