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
R = 192  # R* = R/M, MはCO2分子量

# Look-up-Table [Forget+, 2007] + T1に135, 285を足す
# T1 = np.array([135, 160, 213, 260, 285])  # K
# T2 = np.array([80, 146, 200])  # K
# Surface_pressure = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500])  # Pa

# dust LUT
T1 = np.array([160, 213, 260, 285])  # K
T2 = np.array([100, 146, 200])  # K
Surface_pressure = np.array([100, 350, 600, 850, 1100, 1350])  # Pa

# dust test profile case1
# T1 = np.array([236.62156677246094])
# T2 = np.array([151.51045227050781])
# Surface_pressure = np.array([853.68115234375000])

# dust test profile case2
# T1 = np.array([233.85560607910156])
# T2 = np.array([156.81031799316406])
# Surface_pressure = np.array([864.64227294921875])

# hight profile
Height_km = np.arange(0, 62, 2)  # km
Height = Height_km * 1000  # m

# %%
for i in range(T1.size):
    for j in range(T2.size):
        for k in range(Surface_pressure.size):
            # Scale Hight
            SH = (R * T1[i]) / g

            # pressure profile
            pre = Surface_pressure[k] * np.exp(-Height / SH)

            # Temp profile
            H1 = SH * 0.1
            H2 = SH * 4.0

            a = (T1[i] - T2[j]) / (H1 - H2)
            b = T1[i] - a * H1

            Temp = np.zeros(len(Height))
            for l in range(len(Height)):
                if Height[l] <= H2:
                    Temp[l] = a * Height[l] + b
                else:
                    Temp[l] = T2[j]

            savearray = np.array([Height_km, pre, Temp])
            np.savetxt(
                "dust_ret/LUT_HTP/LUTable_T1_"
                + str(T1[i])
                + "_T2_"
                + str(T2[j])
                + "_PRS"
                + str(Surface_pressure[k])
                + ".txt",
                savearray.T,
            )
# %%
