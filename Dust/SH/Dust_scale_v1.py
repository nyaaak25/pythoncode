
# %%
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama
# Dust_scale_v1.py;; 計算で出てきた高度依存性を確認するためのプログラム

import numpy as np
import matplotlib.pyplot as plt

# Base radiance
ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/ORG/ORG_D0_rad.dat")
ORG_wave = ORG_base[0]
ORG_wav = 1 / ORG_wave
ORG_wave = (1 / ORG_wave) * 10000

ORG_rad = ORG_base[1]
ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

# Baseの高度をここに入れる
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/1d.hc")
altitude = HD_base[:,0]
orginal = HD_base[:,1]

exp_dust = np.zeros(len(altitude))

# exponetialによっているかどうかを確認
for i in range(len(altitude)):
    exp_dust[i] = np.exp(-altitude)


new_opacity = np.zeros(len(altitude))
new_opacity2 = np.zeros(len(altitude))

# Dust profileを変えたときのもの
for i in range(len(altitude)):
    HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/D0/" + str(i) + "_rad.dat")

    rad = HD[1]
    rad = (rad / (ORG_wav**2)) * 1e-7
    new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

plt.scatter(new_opacity, altitude, label="D=0.0", color="black")
plt.ylim([0, 75])

# %%
