"""
-*- coding: utf-8 -*-
LUT内の気圧精度確認するプログラム
@author: A.Kazama kazama@pparc.gp.tohoku.ac.jp

ver. 1.0: 気圧変化をplotし、どのくらいの精度があるかを確認
Created on Tue Jun 28 17:48:00 2022

"""
# %%
# インポート
import matplotlib.pylab as plt
import numpy as np

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.set_ylabel('width', fontsize=10)

Pa_list = ['Pa=50', 'Pa=150', 'Pa=180', 'Pa=215', 'Pa=257', 'Pa=308', 'Pa=369',
           'Pa=442', 'Pa=529', 'Pa=633', 'Pa=758', 'Pa=907', 'Pa=1096', 'Pa=1300', 'Pa=1500']
TA_list = ['TA=135', 'TA=160', 'TA=213', 'TA=260', 'TA=285']
TB_list = ['TB=80', 'TB=146', 'TB=200']
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60', 'SZA=75']
EA_list = ['EA=0', 'EA=5', 'EA=10']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180']
Dust_list = ['Dust=0', 'Dust=0.3', 'Dust=0.6',
             'Dust=0.9', 'Dust=1.2', 'Dust=1.5']
WaterI_list = ['WaterI=0', 'WaterI=0.5', 'WaterI=1.0']
Albedo_list = ['Albedo=0.05', 'Albedo=0.1', 'ALbedo=0.2',
               'Albedo=0.3', 'Albedo=0.4', 'Albedo=0.5']

for i in range(1, 15, 1):
    ARS = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                     str(i)+'_TA3_TB2_SZA4_EA2_PA2_Dust1_WaterI1_SurfaceA3_rad.dat')
    ARS_x = ARS[:, 0]
    ARS_x = (1/ARS_x)*10000
    ARS_x = ARS_x[::-1]
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]

    ARS1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                      str(i+1)+'_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
    ARS_x1 = ARS1[:, 0]
    ARS_x1 = (1/ARS_x)*10000
    ARS_x1 = ARS_x1[::-1]
    ARS_y1 = ARS1[:, 1]
    ARS_y1 = ARS_y1[::-1]

    error1 = (ARS_y1 - ARS_y) * 100 / ARS_y

    ax.plot(ARS_x, ARS_y, label=Pa_list[i])
    # ax.set_xlim(1.81, 1.825)
    # ax.set_ylim(-1e-15, 1e-15)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)


"""
# どのくらいの差分があるかをみる
ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP1_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = (1/ARS_x)*10000
ARS_x = ARS_x[::-1]
ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]

ARS1 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP2_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x1 = ARS1[:, 0]
ARS_x1 = (1/ARS_x)*10000
ARS_x1 = ARS_x1[::-1]
ARS_y1 = ARS1[:, 1]
ARS_y1 = ARS_y1[::-1]

error1 = (ARS_y1 - ARS_y) * 100 / ARS_y


ax.plot(ARS_x, error1)
"""

# %%
# 気圧プロファイルを作成する
# SZA＝0、EA=0, PA=0, TA=135, TB=80, SurfaceA=0.05, waterI=0, Dust=0
for i in range(1, 16, 1):
    ARS = np.loadtxt('/Users/nyonn/Desktop/pythoncode/ARS_calc/SP' +
                     str(i)+'_TA3_TB2_SZA4_EA2_PA2_Dust1_WaterI1_SurfaceA3_rad.dat')
    ARS_x = ARS[:, 0]
    ARS_x = (1/ARS_x)*10000
    ARS_x = ARS_x[::-1]
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]

    POLY_x = [ARS_x[0], ARS_x[3], ARS_x[5], ARS_x[23], ARS_x[24], ARS_x[25]]
    POLY_y = [ARS_y[0], ARS_y[3], ARS_y[5], ARS_y[23], ARS_y[24], ARS_y[25]]
    a, b = np.polyfit(POLY_x, POLY_y, 1)

    cont0 = b + a*ARS_x
    y_calc = 1 - ARS_y/cont0
    y_total = np.sum(y_calc[7:17])

    ax.plot(ARS_x, ARS_y, label=Pa_list[i-1], zorder=i)
    print(y_total)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=5)
# fig.savefig(
#     "/Users/nyonn/Desktop/plot_image/TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1.png")
plt.show()
# %%
