# Look up tableの値が正しいかを確認するためのプログラム

# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

for loop in range(11,12):
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

          fig, ax = plt.subplots()
          ax.set_title("OLD look up table", fontsize=16)

          fig2, ax2 = plt.subplots()
          ax2.set_title("NEW look up table", fontsize=16)

          # dust plot
          # plotをしてみる

          for i_1 in range(0,6):
                    for i_2 in range(0,4):
                              for i_3 in range(0,3):
                                        for i_4 in range(0,6):
                                                  for i_5 in range(0,4):
                                                            #ax.plot(table_12[:,i_5,i_4,i_3,i_2,i_1],table_13[:,i_5,i_4,i_3,i_2,i_1])

                                                            # table_12とtable_13の値の相関関係を撮ってみる
                                                            corr = np.corrcoef(table_12[:,i_5,i_4,i_3,i_2,i_1],table_13[:,i_5,i_4,i_3,i_2,i_1])

                                                            if corr[0,1] < 0.0:
                                                                      #print(i_1,i_2,i_3,i_4,i_5)
                                                                      ax2.plot(table_12[:,i_5,i_4,i_3,i_2,i_1],table_13[:,i_5,i_4,i_3,i_2,i_1])
                                                            if corr[0,1] > 0.0:
                                                                      ax.plot(table_12[:,i_5,i_4,i_3,i_2,i_1],table_13[:,i_5,i_4,i_3,i_2,i_1])
          
          """
                    #ax.scatter(dust,table_13[:,0,4,2,1,i_1])
                    #ax.scatter(dust,table_12[:,0,4,2,1,i_1])  
                    ax.plot(table_13[:,0,4,2,1,i_1],table_12[:,0,4,2,1,i_1])

          for i_2 in range(0,4):
                    #ax.scatter(dust,table_13[:,2,4,0,i_2,2])
                    #ax2.scatter(dust,table_12[:,2,4,0,i_2,2])
                    ax.plot(table_13[:,2,4,0,i_2,1],table_12[:,2,4,0,i_2,1])

          for i_3 in range(0,3):
                    #ax.scatter(dust,table_13[:,2,4,i_3,3,2])
                    #ax2.scatter(dust,table_12[:,2,4,i_3,3,2])
                    ax.plot(table_13[:,2,4,i_3,3,1],table_12[:,2,4,i_3,3,1])

          for i_4 in range(0,6):
                    #ax.scatter(dust,table_13[:,2,i_4,0,3,2])
                    #ax2.scatter(dust,table_12[:,2,i_4,0,3,2])
                    ax.plot(table_13[:,2,i_4,0,3,1],table_12[:,2,i_4,0,3,1])
          
          for i_5 in range(0,4):
                    #ax.scatter(dust,table_13[:,i_5,4,0,3,2])
                    #ax2.scatter(dust,table_12[:,i_5,4,0,3,2])
                    ax.plot(table_13[:,i_5,4,0,3,1],table_12[:,i_5,4,0,3,1])
"""
"""
          # SP plot
          for i_1 in range(0,5):
                    ax.scatter(SP,table_13[i_1,2,4,0,3,:])
                    ax2.scatter(SP,table_12[i_1,2,4,0,3,:])
          
          for i_2 in range(0,4):
                    ax.scatter(SP,table_13[2,2,4,0,i_2,:])
                    ax2.scatter(SP,table_12[2,2,4,0,i_2,:])
          
          for i_3 in range(0,3):
                    ax.scatter(SP,table_13[2,2,4,i_3,3,:])
                    ax2.scatter(SP,table_12[2,2,4,i_3,3,:])
          
          for i_4 in range(0,5):
                    ax.scatter(SP,table_13[2,2,i_4,0,3,:])
                    ax2.scatter(SP,table_12[2,2,i_4,0,3,:])
          
          for i_5 in range(0,4):
                    ax.scatter(SP,table_13[2,i_5,4,0,3,:])
                    ax2.scatter(SP,table_12[2,i_5,4,0,3,:])
          
          """
          

# %%
