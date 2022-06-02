# %%
"""
-*- coding: utf-8 -*-
放射強度、装置関数を走らせるプログラム
@author: A.Kazama kazama@pparc.gp.tohoku.ac.jp

ver. 1.0: 計算された光学的厚みから放射強度に変換
Created on Thu Feb 17 22:53:00 2022

ver. 2.0: OMEGA Insturment function導入
created on Mon Mar 21 14:48:00 2022

ver. 2.1: OMEGA Instrument function改変
created on Thu Mar 24 14:48:00 2022
"""

# library import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

from sympy import OmegaPower

# 波数
v_txt = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Tau_file/LUtable_1_tau.txt')
v1 = v_txt[:, 0]  # cm-1
cm_v = (1/v1)*10000
wav = cm_v[::-1]  # um

# Optical Depth >> Intensity
Tau_txt = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Tau_file/LUtable_1_tau.txt')
Tau1 = Tau_txt[:, 1]
Tau = Tau1[::-1]
# sza_theta = 18.986036
sza_theta = 0

I0 = np.exp(-Tau/np.cos(sza_theta))
Iobs = I0 * np.exp(-Tau)

# OMEGAのchannel center listとなまらせた後の配列作成
OMEGAcenter_list = np.array([1.8143300, 1.8284900, 1.8426300, 1.8567700, 1.8708900, 1.8850000, 1.8990901, 1.9131700, 1.9272400, 1.9412900, 1.9553300, 1.9693500, 1.9833500,
                             1.9973400, 2.0113201, 2.0252800, 2.0392201, 2.0531399, 2.0670500, 2.0809400, 2.0948100, 2.1086600, 2.1224899, 2.1363101, 2.1501000, 2.1638801, 2.1776299])

OMEGAchannel = np.zeros(len(OMEGAcenter_list))
um_cm_OMEGA = (1/OMEGAcenter_list)*10000
ten_OMEGA = um_cm_OMEGA[::-1]

# %%
# 装置関数実装部分　[GAUSSIAN実装]
# OMEGAの中心分、装置関数をかけ合わせる
for k in range(len(OMEGAcenter_list)):
    # OMEGAの中心波長aについての GAUSSIANを定義
    mu = OMEGAcenter_list[k]
    # sig = 6.5e-3  # OMEGAの波長分解能は13nm
    sig = 6.5e-3

    # wav 1 ~ wav n までのガウシアンの値を求める
    C1 = np.where((wav <= mu + 0.013) & (mu - 0.013 < wav))
    new_wav = wav[C1]
    GAUSSIAN_func = (1/np.sqrt(2*np.pi*(sig**2))) * \
        np.exp(-((new_wav-mu)**2)/(2*sig**2))

    # ガウシアンと計算されたスペクトルをかけ合わせる
    new_I = Iobs[C1]
    multiple = GAUSSIAN_func * new_I

    # 畳み込みの台形近似で積分を行う
    S_conv = 0
    for i in range(len(new_wav)-1):
        S_conv += ((multiple[i]+multiple[i+1])*(new_wav[i+1]-new_wav[i]))/2

    # ガウシアンの台形近似で積分を行う
    S_gauss = 0
    for i in range(len(new_wav)-1):
        S_gauss += ((GAUSSIAN_func[i]+GAUSSIAN_func[i+1])
                    * (new_wav[i+1]-new_wav[i]))/2

    # OMEGA channelに落とし込み
    OMEGAchannel[k] = S_conv/S_gauss
    # print(OMEGAchannel)

tau_v = np.stack([OMEGAcenter_list, OMEGAchannel], 1)
# np.savetxt('instrumentfunction_001.txt', tau_v, fmt='%.10e')

"""
# 観測スペクトルと比較
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = 'ORB0006_1.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())
wvl = sav_data['wvl']

CO2 = np.where((wvl > 1.8) & (wvl < 2.2))
CO2_wav = wvl[CO2]
"""

# ARSMで計算されたものとの比較
tau_txt = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/not use file/test1_trans.dat')
cut120_1 = tau_txt[:, 1]
cut120 = cut120_1[::-1]

error = cut120-OMEGAchannel
# ---------- plot部分 -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
ax.plot(OMEGAcenter_list, error, color='b', linewidth=1)
# ax.plot(OMEGAcenter_list, cut120, color='r',label="ARSM Calc", lw=1)
# ax.plot(OMEGAcenter_list, OMEGAchannel, color='b',label = "My Calc",lw=1)
# ax.set_xlim(4973, 4975)
# ax.set_yscale('log')
# ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
ax.set_xlabel('Wavenumber [μm]', fontsize=14)
ax.set_ylabel('Difference [Transmittance]', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

# 凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=8)
plt.show()

# %%

# %%
