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

# Look-up-Table [Forget+, 2007] + T1に135, 285を足す
T1 = np.array([135, 160, 213, 260, 285])  # K
T2 = np.array([80, 146, 200])  # K
Surface_pressure = np.array([50, 150, 180, 215, 257, 308,
                             369, 442, 529, 633, 758, 907, 1096, 1300, 1500])  # Pa

# hight profile
Hight_km = np.arange(0, 62, 2)  # km
Hight = Hight_km * 1000  # m

# %%
for i in range(T1.size):
    for j in range(T2.size):
        for k in range(Surface_pressure.size):
            # Scale Hight
            SH = (R*T1[i])/g

            # pressure profile
            pre = Surface_pressure[k] * np.exp(-Hight/SH)

            # Temp profile
            H1 = SH*0.1
            H2 = SH*4.0

            a = (T1[i]-T2[j])/(H1-H2)
            b = T1[i]-a*H1

            Temp = np.zeros(len(Hight))
            for l in range(len(Hight)):
                if Hight[l] <= H2:
                    Temp[l] = a * Hight[l] + b
                else:
                    Temp[l] = T2[j]

savearray = np.array([Hight_km, pre, Temp])
np.savetxt('LookUpTable_HTP/LUTable_T1_'+str(T1[i])+'_T2_'+str(
    T2[j])+'_PRS'+str(Surface_pressure[k])+'.txt', savearray.T)

# %%
