# %%
# 気圧と温度プロファイルを導出する
"""
-*- coding: utf-8 -*-
放射強度、装置関数を走らせるプログラム
@author: A.Kazama kazama@pparc.gp.tohoku.ac.jp

ver. 1.0: ある場所の気圧、温度プロファイルを作成
Created on Mon Apr 11 16:05:00 2022

"""

import numpy as np

# parameter指定 MKS単位系
g = 3.72  # m s-1
# R = 8.314  # E+7  # (g*cm^2*s^2)/(mol*K)
R = 192

# Look-up-Table [Forget+, 2007]
T1 = [160, 213, 260]  # K
T2 = [80, 146, 200]  # K
Surface_pressure = [50, 150, 180, 215, 257, 308,
                    369, 442, 529, 633, 758, 907, 1096, 1300, 1500]  # Pa

# hight profile
Hight_km = np.arange(0, 62, 2)  # km
Hight = Hight_km * 1000  # m

# Scale Hight
SH = (R*T1[1])/g

# pressure profile
pre = Surface_pressure[9] * np.exp(-Hight/SH)

# Temp profile
H1 = SH*0.1
H2 = SH*4.0

a = (T1[1]-T2[1])/(H1-H2)
b = T1[1]-a*H1

Temp = np.zeros(len(Hight))
for i in range(len(Hight)):
    if Hight[i] <= H2:
        Temp[i] = a * Hight[i] + b
    else:
        Temp[i] = T2[1]

savearray = np.array([Hight_km, pre, Temp])
np.savetxt('pre_temp_profile.txt', savearray.T)
