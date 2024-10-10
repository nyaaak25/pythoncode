# Look up tableの値が正しいかを確認するためのプログラム

# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#fig2, ax2 = plt.subplots()
#ax2.set_title("Bad condition", fontsize=16)

#count = 0
# loopは17まで指定可能
for loop in range(6,6):


          fig2, ax2 = plt.subplots()
          ax2.set_title("Bad condition", fontsize=16)

          count = 0

          # ファイルの読み込み
          all_data_12 = readsav("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/lut/Table_dust_calc_" + str(loop) + ".sav")
          table_12 = all_data_12["Table_Equivalent_dust"]

          all_data_13 = readsav("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/old lut/Table_dust_calc_" + str(loop) + ".sav")
          table_13 = all_data_13["Table_Equivalent_dust"]

          dust = [0, 0.3, 0.6, 0.9, 1.2]
          SP = [100, 350, 600, 850, 1100, 1350]
          EA = [0, 5, 10, 30]
          SZA = [0, 15, 30, 45, 60, 75]
          TA = [160, 213, 260, 285]
          TB = [100, 146, 200]
          index = [5, 4, 6, 3, 4, 6]

          num = [1,2,3,4,5]

          for i_1 in range(0,6):
                    for i_2 in range(0,4):
                              for i_3 in range(0,3):
                                        for i_4 in range(0,6):
                                                  for i_5 in range(0,4):


                                                            # table_12, 13の傾きを確認
                                                            slope = np.polyfit(dust,table_12[:,i_5,i_4,i_3,i_2,i_1],1)[0]
                                                            slope2 = np.polyfit(dust,table_13[:,i_5,i_4,i_3,i_2,i_1],1)[0]

                                                            if slope < 0.0:
                                                                      ind_number = [i_1,i_2,i_3,i_4,i_5]
                                                                      ax2.scatter(num, ind_number)
                                                                      count += 1
          print(count)
                                                                      #print(ind_number)



# %%
# aero fileを読み込んでみる
Before_dust_aero = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/aero file/dust_2500_8500_fixed.aero")
WV_before = Before_dust_aero[:,0]
EXT_before = Before_dust_aero[:,1]
SSA_before = Before_dust_aero[:,2]
PPA_before = Before_dust_aero[-1,3:]

Mathieu_dust_aero = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/aero file/dust_2500-25000_mathieu.aero")
WV_MAT = Mathieu_dust_aero[:,0]
EXT_MAT = Mathieu_dust_aero[:,1]
SSA_MAT = Mathieu_dust_aero[:,2]
PPA_MAT = Mathieu_dust_aero[39,3:]

fig, ax = plt.subplots()
ax.plot(WV_before, EXT_before, label="Before")
ax.plot(WV_MAT, EXT_MAT, label="Mathieu")
ax.set_xlabel("Wavelength [cm-1]")
ax.set_ylabel("Extinction [cm^2/g]")
ax.set_xlim(3400, 3600)
ax.legend()

fig, ax = plt.subplots()
ax.plot(WV_before, SSA_before, label="Before")
ax.plot(WV_MAT, SSA_MAT, label="Mathieu")
ax.set_xlabel("Wavelength [cm-1]")
ax.set_ylabel("Single Scattering Albedo")
ax.set_xlim(3400, 3600)
ax.legend()

fig, ax = plt.subplots()
ax.plot(PPA_before, label="Before")
ax.plot(PPA_MAT, label="Mathieu")
ax.set_xlabel("Particle Size [um]")
ax.set_ylabel("Phase Function")
ax.legend()
# %%
