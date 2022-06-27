# %%
# インポート
import matplotlib.pylab as plt
import numpy as np

# SZA＝0、EA=0, PA=0, TA=135, TB=80, SurfaceA=0.05, waterI=0, Dust=0
# pressure index 1 :: 50 Pa
ARS_1 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP1_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x1 = ARS_1[:, 0]
ARS_x1 = (1/ARS_x1)*10000
ARS_x1 = ARS_x1[::-1]
ARS_y1 = ARS_1[:, 1]
ARS_y1 = ARS_y1[::-1]

# pressure index 2 :: 150 Pa
ARS_2 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP2_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x2 = ARS_2[:, 0]
ARS_x2 = (1/ARS_x2)*10000
ARS_x2 = ARS_x2[::-1]
ARS_y2 = ARS_2[:, 1]
ARS_y2 = ARS_y2[::-1]

# pressure index 3 :: 180 Pa
ARS_3 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP3_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x3 = ARS_3[:, 0]
ARS_x3 = (1/ARS_x3)*10000
ARS_x3 = ARS_x3[::-1]
ARS_y3 = ARS_3[:, 1]
ARS_y3 = ARS_y3[::-1]

# pressure index 4 :: 215 Pa
ARS_4 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP4_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x4 = ARS_4[:, 0]
ARS_x4 = (1/ARS_x4)*10000
ARS_x4 = ARS_x4[::-1]
ARS_y4 = ARS_4[:, 1]
ARS_y4 = ARS_y4[::-1]

# pressure index 5 :: 257 Pa
ARS_5 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP5_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x5 = ARS_5[:, 0]
ARS_x5 = (1/ARS_x5)*10000
ARS_x5 = ARS_x5[::-1]
ARS_y5 = ARS_5[:, 1]
ARS_y5 = ARS_y5[::-1]

# pressure index 56:: 308 Pa
ARS_6 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP6_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x6 = ARS_6[:, 0]
ARS_x6 = (1/ARS_x6)*10000
ARS_x6 = ARS_x6[::-1]
ARS_y6 = ARS_6[:, 1]
ARS_y6 = ARS_y6[::-1]

# pressure index 7 :: 369 Pa
ARS_7 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP7_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x7 = ARS_7[:, 0]
ARS_x7 = (1/ARS_x7)*10000
ARS_x7 = ARS_x7[::-1]
ARS_y7 = ARS_7[:, 1]
ARS_y7 = ARS_y7[::-1]

# pressure index 8 :: 442 Pa
ARS_8 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP8_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x8 = ARS_8[:, 0]
ARS_x8 = (1/ARS_x8)*10000
ARS_x8 = ARS_x8[::-1]
ARS_y8 = ARS_8[:, 1]
ARS_y8 = ARS_y8[::-1]

# pressure index 9 :: 529 Pa
ARS_9 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP9_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x9 = ARS_9[:, 0]
ARS_x9 = (1/ARS_x9)*10000
ARS_x9 = ARS_x9[::-1]
ARS_y9 = ARS_9[:, 1]
ARS_y9 = ARS_y9[::-1]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.plot(ARS_x1, ARS_y1, color='blue', label="Pa = 50", zorder=1)
ax.plot(ARS_x2, ARS_y2, color='red', label="Pa = 150", zorder=2)
ax.plot(ARS_x3, ARS_y3, color='green', label="Pa = 180", zorder=3)
ax.plot(ARS_x4, ARS_y4, color='orange', label="Pa = 215", zorder=4)
ax.plot(ARS_x5, ARS_y5, color='pink', label="Pa = 257", zorder=5)
ax.plot(ARS_x6, ARS_y6, color='purple', label="Pa = 308", zorder=6)
ax.plot(ARS_x7, ARS_y7, color='skyblue', label="Pa = 369", zorder=7)
ax.plot(ARS_x8, ARS_y8, color='yellow', label="Pa = 442", zorder=8)
ax.plot(ARS_x9, ARS_y9, color='black', label="Pa = 529", zorder=8)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=8)
plt.show()
# %%
