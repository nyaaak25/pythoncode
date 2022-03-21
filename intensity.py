# %%
"""
-*- coding: utf-8 -*-
放射強度、装置関数を走らせるプログラム
@author: A.Kazama kazama@pparc.gp.tohoku.ac.jp

ver. 1.0: 計算された光学的厚みから放射強度に変換
Created on Thu Feb 17 22:53:00 2022

ver. 2.0: OMEGA Insturment function導入
created on Mon Mar 21 14:48:00 2022
"""

# library import
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats

# Optical Depth >> Intensity
Tau_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
Tau = Tau_txt[:, 1]
v = Tau_txt[:, 0]
sza_theta = 18.986036

I0 = np.exp(-Tau/np.cos(sza_theta))
Iobs = I0 * np.exp(-Tau)

x1 = v
y1 = Iobs


# 装置関数実装部分
# 畳み込む装置関数定義。OMEGAはGAUSSIUN
mu_conv = 0.0  # Centroid
sigma_conv = 0.5  # Width
convolving_term = stats.norm(mu_conv, sigma_conv)
xconv = np.linspace(-20, 20, 4000)
yconv = convolving_term.pdf(xconv)

# 畳み込みを行う部分
convolved_pdf = signal.convolve(Iobs, yconv, mode='same')/sum(Iobs)


# plot部分
fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
# ax.plot(x1, y1, color='b', linewidth=0.1)
ax.plot(v, convolved_pdf, color='b')
# ax.set_xlim(4973, 4975)
# ax.set_yscale('log')
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
ax.set_ylabel('Intensity', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

# 凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=14)
plt.show()

# %%
