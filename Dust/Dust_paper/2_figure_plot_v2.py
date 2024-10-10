# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
# dust propertiesデータを変更後のLUTを使った評価

# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# %%
# -------------------------------------- Figure 1 --------------------------------------
# Figure 1はOMEGA-like spectralを表示して、2.77 μmがどのように見えるかを示す
# 2.77 μmのradianceを表示する
Dust_list  = 0.0 + np.arange(0, 1.6, 0.1)
Dust_legend = ["Dust=0.0", "Dust=0.1", "Dust=0.2", "Dust=0.3", "Dust=0.4", "Dust=0.5", "Dust=0.6", "Dust=0.7", "Dust=0.8", "Dust=0.9", "Dust=1.0", "Dust=1.1", "Dust=1.2", "Dust=1.3", "Dust=1.4", "Dust=1.5"]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Wavenumber [μm]", fontsize=12)
ax.set_ylabel("Radiance [W/m2/sr/μm]", fontsize=12)

plt.rcParams['image.cmap'] = 'gray'

for i in range(0, 16, 5):
    ARS = np.loadtxt("/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/OMEGA-like/Test_loc2_dust" + str(i) + ".dat")
    ARS_x = ARS[:, 0]
    ARS_x = ARS_x[::-1]
    ARS_wav = 1 / ARS_x
    ARS_x = (1 / ARS_x) * 10000
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]
    ARS_y = (ARS_y / (ARS_wav * ARS_wav)) * 1e-7

    ax.plot(ARS_x, ARS_y, label=Dust_legend[i], zorder=i, lw=3)
    print(ARS_y[5])
    #ax.scatter(ARS_x[5], ARS_y[5])

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper left", fontsize=10)
ax.axvline(x=ARS_x[5], color="black", linestyle="dashed")
ax.set_ylim(0, 0.75)

# %%
# -------------------------------------- Figure 2 --------------------------------------
# ダストの高度依存性を示す図を作成する
Dust_list = ["τ=0.35", "τ=3.5", "τ=0.035"]
color_list = ["black", "orange", "green"]
linestyle = ["-", "--", "-."]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function", fontsize=16)
ax.set_ylabel("Altitude [km]", fontsize=16)
ax.set_ylim(0, 70)
ax.set_xlim(0, 1.05)

# 2.7 μmのプロット
for loop in range(0, 3, 1):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3
    ax.plot(normarize, altitude, label=Dust_list[loop], color='black', linestyle=linestyle[loop])


ax.legend()

# %%
# -------------------------------------- Figure 3 --------------------------------------
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/ORB1212_3.sav"
sav_data = readsav(sav_fname)
longitude = sav_data["longi"]
latitude = sav_data["lati"]
new_radiance = sav_data["new_rad"]
dust_opacity = sav_data["dust_opacity"]
altitude = sav_data["ret_alt_dust"]

dust_opacity = dust_opacity + 0

ind = np.where(dust_opacity <= 0)
dust_opacity[ind] = np.nan

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_title("Retrieval dust optical depth at 2.77 μm", fontsize=14)
c = ax.contourf(longitude, latitude, dust_opacity, cmap="jet",vmin=0)
cbar = fig.colorbar(c, ax=ax, orientation="vertical")
cbar.set_label("Dust optical depth", fontsize=12)
#plt.axis('equal')
plt.show()

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_title("Radiance at 2.77 μm", fontsize=14)
c = ax.contourf(longitude, latitude, new_radiance, cmap="jet",vmin=0)
cbar2 = fig2.colorbar(c, ax=ax, orientation="vertical")
cbar2.set_label("Radiance", fontsize=12)
#plt.axis('equal')
plt.show()

# %%
# -------------------------------------- Figure 4_1 --------------------------------------
# Figure 2.3を作成するプログラム
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
color_pallet = ["tan","sandybrown", color_pallet[3], "chocolate", color_pallet[2], "black"]
label_tau = ["(1) τ = 0.02", "(2) τ = 0.3", "(3) τ = 1.2", "(4) τ = 2.7", "(5) τ = 3.9", "(6) τ = 6.6"]

# Temperature profile deviation
# condition1を代入
sav_fname1 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con1_TA_result.sav"
sav_data1 = readsav(sav_fname1)
temp = sav_data1["dev"]
dust_1 = sav_data1["dust_result"]
dust_con1 = dust_1 - dust_1[10]
dust_con1_per = (dust_con1 * 100) / dust_1[10]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con2_TA_result.sav"
sav_data2 = readsav(sav_fname2)
dust_2 = sav_data2["dust_result"]
dust_con2 = dust_2 - dust_2[10]
dust_con2_per = (dust_con2 * 100) / dust_2[10]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con3_TA_result.sav"
sav_data3 = readsav(sav_fname3)
dust_3 = sav_data3["dust_result"]
dust_con3 = dust_3 - dust_3[10]
dust_con3_per = (dust_con3 * 100) / dust_3[10]

# condition4を代入
sav_fname4 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con4_TA_result.sav"
sav_data4 = readsav(sav_fname4)
dust_4 = sav_data4["dust_result"]
dust_con4 = dust_4 - dust_4[10]
dust_con4_per = (dust_con4 * 100) / dust_4[10]

# condition5を代入
sav_fname5 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con5_TA_result.sav"
sav_data5 = readsav(sav_fname5)
dust_5 = sav_data5["dust_result"]
dust_con5 = dust_5 - dust_5[10]
dust_con5_per = (dust_con5 * 100) / dust_5[10]

# condition6を代入
sav_fname6 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con6_TA_result.sav"
sav_data6 = readsav(sav_fname6)
dust_6 = sav_data6["dust_result"]
dust_con6 = dust_6 - dust_6[10]
dust_con6_per = (dust_con6 * 100) / dust_6[10]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_title("(a) Temperature deviation", fontsize=14)
ax.set_xlabel("Temperature deviation (K)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(temp, dust_con1, color=color_pallet[0], label=label_tau[0],s=10)
ax.scatter(temp, dust_con2, color=color_pallet[1], label=label_tau[1],s=10)
ax.scatter(temp, dust_con3, color=color_pallet[2], label=label_tau[2],s=10)
ax.scatter(temp, dust_con4, color=color_pallet[3], label=label_tau[3],s=10)
ax.scatter(temp, dust_con5, color=color_pallet[4], label=label_tau[4],s=10)
ax.scatter(temp, dust_con6, color=color_pallet[5], label=label_tau[5],s=10)
ax.plot(temp, dust_con1, color=color_pallet[0], alpha=0.5)
ax.plot(temp, dust_con2, color=color_pallet[1], alpha=0.5)
ax.plot(temp, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(temp, dust_con4, color=color_pallet[3], alpha=0.5)
ax.plot(temp, dust_con5, color=color_pallet[4], alpha=0.5)
ax.plot(temp, dust_con6, color=color_pallet[5], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=9)

print('------------------------------------')
print('BASE CONDITION')
print('dev', temp[10])
print('retreival dust (1): ',dust_1[10])
print('retreival dust (2): ',dust_2[10])
print('retreival dust (3): ',dust_3[10])
print('retreival dust (4): ',dust_4[10])
print('retreival dust (5): ',dust_5[10])
print('retreival dust (6): ',dust_6[10])
print('------------------------------------')
print('CONDITION (1)')
print('dev', temp[0])
print('-10 K: ', dust_con1[0])
print('-10 K: ', dust_con1_per[0])
print('retrieval dust: ', dust_1[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con1[20])
print('+10 K: ', dust_con1_per[20])
print('retrieval dust: ', dust_1[20])
print('------------------------------------')
print('CONDITION (2)')
print('dev', temp[0])
print('-10 K: ', dust_con2[0])
print('-10 K: ', dust_con2_per[0])
print('retrieval dust: ', dust_2[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con2[20])
print('+10 K: ', dust_con2_per[20])
print('retrieval dust: ', dust_2[20])
print('------------------------------------')
print('CONDITION (3)')
print('dev', temp[0])
print('-10 K: ', dust_con3[0])
print('-10 K: ', dust_con3_per[0])
print('retrieval dust: ', dust_3[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con3[20])
print('+10 K: ', dust_con3_per[20])
print('retrieval dust: ', dust_3[20])
print('------------------------------------')
print('CONDITION (4)')
print('dev', temp[0])
print('-10 K: ', dust_con4[0])
print('-10 K: ', dust_con4_per[0])
print('retrieval dust: ', dust_4[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con4[20])
print('+10 K: ', dust_con4_per[20])
print('retrieval dust: ', dust_4[20])
print('------------------------------------')
print('CONDITION (5)')
print('dev', temp[0])
print('-10 K: ', dust_con5[0])
print('-10 K: ', dust_con5_per[0])
print('retrieval dust: ', dust_5[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con5[20])
print('+10 K: ', dust_con5_per[20])
print('retrieval dust: ', dust_5[20])
print('------------------------------------')
print('CONDITION (6)')
print('dev', temp[0])
print('-10 K: ', dust_con6[0])
print('-10 K: ', dust_con6_per[0])
print('retrieval dust: ', dust_6[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con6[20])
print('+10 K: ', dust_con6_per[20])
print('retrieval dust: ', dust_6[20])
print('------------------------------------')

# %%
# -------------------------------------- Figure 4_2 --------------------------------------
# Surface pressure deviation
color_pallet = ["tan","sandybrown", color_pallet[3], "chocolate", color_pallet[2], "black"]
label_tau = ["(1) τ = 0.02", "(2) τ = 0.3", "(3) τ = 1.2", "(4) τ = 2.7", "(5) τ = 3.9", "(6) τ = 6.6"]

# condition1を代入
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con1_SP_result.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_1 = sav_data1["dust_result"]
dust_con1 = dust_1 - dust_1[50]
dust_con1_per = (dust_con1 * 100) / dust_1[50]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con2_SP_result.sav"
sav_data2 = readsav(sav_fname2)
dust_2 = sav_data2["dust_result"]
dust_con2 = dust_2 - dust_2[50]
dust_con2_per = (dust_con2 * 100) / dust_2[50]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con3_SP_result.sav"
sav_data3 = readsav(sav_fname3)
dust_3 = sav_data3["dust_result"]
dust_con3 = dust_3 - dust_3[50]
dust_con3_per = (dust_con3 * 100) / dust_3[50]

# condition4を代入
sav_fname4 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con4_SP_result.sav"
sav_data4 = readsav(sav_fname4)
dust_4 = sav_data4["dust_result"]
dust_con4 = dust_4 - dust_4[50]
dust_con4_per = (dust_con4 * 100) / dust_4[50]

# condition5を代入
sav_fname5 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con5_SP_result.sav"
sav_data5 = readsav(sav_fname5)
dust_5 = sav_data5["dust_result"]
dust_con5 = dust_5 - dust_5[50]
dust_con5_per = (dust_con5 * 100) / dust_5[50]

# condition6を代入
sav_fname6 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con6_SP_result.sav"
sav_data6 = readsav(sav_fname6)
dust_6 = sav_data6["dust_result"]
dust_con6 = dust_6 - dust_6[50]
dust_con6_per = (dust_con6 * 100) / dust_6[50]


# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_title("(b) Surface pressure deviation", fontsize=14)
ax.set_xlabel("Surface pressure deviation (Pa)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(surface_pressure, dust_con1, color=color_pallet[0], label=label_tau[0],s=2)
ax.scatter(surface_pressure, dust_con2, color=color_pallet[1], label=label_tau[1],s=2)
ax.scatter(surface_pressure, dust_con3, color=color_pallet[2], label=label_tau[2],s=2)
ax.scatter(surface_pressure, dust_con4, color=color_pallet[3], label=label_tau[3],s=2)
ax.scatter(surface_pressure, dust_con5, color=color_pallet[4], label=label_tau[4],s=2)
ax.scatter(surface_pressure, dust_con6, color=color_pallet[5], label=label_tau[5],s=2)
ax.plot(surface_pressure, dust_con1, color=color_pallet[0], alpha=0.5)
ax.plot(surface_pressure, dust_con2, color=color_pallet[1], alpha=0.5)
ax.plot(surface_pressure, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(surface_pressure, dust_con4, color=color_pallet[3], alpha=0.5)
ax.plot(surface_pressure, dust_con5, color=color_pallet[4], alpha=0.5)
ax.plot(surface_pressure, dust_con6, color=color_pallet[5], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

print('------------------------------------')
print('BASE CONDITION')
print('dev', surface_pressure[50])
print('retreival dust (1): ',dust_1[50])
print('retreival dust (2): ',dust_2[50])
print('retreival dust (3): ',dust_3[50])
print('retreival dust (4): ',dust_4[50])
print('retreival dust (5): ',dust_5[50])
print('retreival dust (6): ',dust_6[50])
print('------------------------------------')
print('CONDITION (1)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con1[0])
print('-50 Pa: ', dust_con1_per[0])
print('retrieval dust: ', dust_1[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con1[100])
print('+50 Pa: ', dust_con1_per[100])
print('retrieval dust: ', dust_1[100])
print('------------------------------------')
print('CONDITION (2)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con2[0])
print('-50 Pa: ', dust_con2_per[0])
print('retrieval dust: ', dust_2[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con2[100])
print('+50 Pa: ', dust_con2_per[100])
print('retrieval dust: ', dust_2[100])
print('------------------------------------')
print('CONDITION (3)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con3[0])
print('-50 Pa: ', dust_con3_per[0])
print('retrieval dust: ', dust_3[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con3[100])
print('+50 Pa: ', dust_con3_per[100])
print('retrieval dust: ', dust_3[100])
print('------------------------------------')
print('CONDITION (4)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con4[0])
print('-50 Pa: ', dust_con4_per[0])
print('retrieval dust: ', dust_4[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con4[100])
print('+50 Pa: ', dust_con4_per[100])
print('retrieval dust: ', dust_4[100])
print('------------------------------------')
print('CONDITION (5)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con5[0])
print('-50 Pa: ', dust_con5_per[0])
print('retrieval dust: ', dust_5[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con5[100])
print('+50 Pa: ', dust_con5_per[100])
print('retrieval dust: ', dust_5[100])
print('------------------------------------')
print('CONDITION (6)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con6[0])
print('-50 Pa: ', dust_con6_per[0])
print('retrieval dust: ', dust_6[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con6[100])
print('+50 Pa: ', dust_con6_per[100])
print('retrieval dust: ', dust_6[100])
print('------------------------------------')


# %%
# -------------------------------------- Figure 4_3 --------------------------------------

data_dir = pjoin(dirname(sio.__file__), "tests", "data")
color_pallet = ["tan","sandybrown", color_pallet[3], "chocolate", color_pallet[2], "black"]
label_tau = ["(1) τ = 0.02", "(2) τ = 0.3", "(3) τ = 1.2", "(4) τ = 2.7", "(5) τ = 3.9", "(6) τ = 6.6"]

sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con1_SN_result.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_1= sav_data1["dust_result"]
dust_con1 = dust_1 - dust_1[37]
dust_con1_per = (dust_con1 * 100) / dust_1[37]

# condition2
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con2_SN_result.sav"
sav_data2 = readsav(sav_fname2)
dust_2 = sav_data2["dust_result"]
dust_con2 = dust_2 - dust_2[37]
dust_con2_per = (dust_con2 * 100) / dust_2[37]

# condition3
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con3_SN_result.sav"
sav_data3 = readsav(sav_fname3)
dust_3 = sav_data3["dust_result"]
dust_con3 = dust_3 - dust_3[37]
dust_con3_per = (dust_con3 * 100) / dust_3[37]

# condition4
sav_fname4 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con4_SN_result.sav"
sav_data4 = readsav(sav_fname4)
dust_4 = sav_data4["dust_result"]
dust_con4 = dust_4 - dust_4[37]
dust_con4_per = (dust_con4 * 100) / dust_4[37]

# condition5
sav_fname5 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con5_SN_result.sav"
sav_data5 = readsav(sav_fname5)
dust_5 = sav_data5["dust_result"]
dust_con5 = dust_5 - dust_5[37]
dust_con5_per = (dust_con5 * 100) / dust_5[37]

# condition6
sav_fname6 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con6_SN_result.sav"
sav_data6 = readsav(sav_fname6)
dust_6 = sav_data6["dust_result"]
dust_con6 = dust_6 - dust_6[37]
dust_con6_per = (dust_con6 * 100) / dust_6[37]


# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_title("(c) Signal noize deviation", fontsize=14)
ax.set_xlabel("Signal noize deviation (DN)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(surface_pressure, dust_con1, color=color_pallet[0], label=label_tau[0],s=2)
ax.scatter(surface_pressure, dust_con2, color=color_pallet[1], label=label_tau[1],s=2)
ax.scatter(surface_pressure, dust_con3, color=color_pallet[2], label=label_tau[2],s=2)
ax.scatter(surface_pressure, dust_con4, color=color_pallet[3], label=label_tau[3],s=2)
ax.scatter(surface_pressure, dust_con5, color=color_pallet[4], label=label_tau[4],s=2)
ax.scatter(surface_pressure, dust_con6, color=color_pallet[5], label=label_tau[5],s=2)
ax.plot(surface_pressure, dust_con1, color=color_pallet[0], alpha=0.5)
ax.plot(surface_pressure, dust_con2, color=color_pallet[1], alpha=0.5)
ax.plot(surface_pressure, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(surface_pressure, dust_con4, color=color_pallet[3], alpha=0.5)
ax.plot(surface_pressure, dust_con5, color=color_pallet[4], alpha=0.5)
ax.plot(surface_pressure, dust_con6, color=color_pallet[5], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

print('------------------------------------')
print('BASE CONDITION')
print('dev', surface_pressure[37])
print('retreival dust (1): ',dust_1[37])
print('retreival dust (2): ',dust_2[37])
print('retreival dust (3): ',dust_3[37])
print('retreival dust (4): ',dust_4[37])
print('retreival dust (5): ',dust_5[37])
print('retreival dust (6): ',dust_6[37])
print('------------------------------------')
print('CONDITION (1)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con1[0])
print('-1.85 DN: ', dust_con1_per[0])
print('retrieval dust: ', dust_1[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con1[74])
print('+1.85 DN: ', dust_con1_per[74])
print('retrieval dust: ', dust_1[74])
print('------------------------------------')
print('CONDITION (2)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con2[0])
print('-1.85 DN: ', dust_con2_per[0])
print('retrieval dust: ', dust_2[0])
print('   ')
print('dev', surface_pressure[74])
print('+50 Pa: ', dust_con2[74])
print('+50 Pa: ', dust_con2_per[74])
print('retrieval dust: ', dust_2[74])
print('------------------------------------')
print('CONDITION (3)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con3[0])
print('-1.85 DN: ', dust_con3_per[0])
print('retrieval dust: ', dust_3[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con3[74])
print('+1.85 DN: ', dust_con3_per[74])
print('retrieval dust: ', dust_3[74])
print('------------------------------------')
print('CONDITION (4)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con4[0])
print('-50 Pa: ', dust_con4_per[0])
print('retrieval dust: ', dust_4[0])
print('   ')
print('dev', surface_pressure[74])
print('+50 Pa: ', dust_con4[74])
print('+50 Pa: ', dust_con4_per[74])
print('retrieval dust: ', dust_4[74])
print('------------------------------------')
print('CONDITION (5)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con5[0])
print('-1.85 DN: ', dust_con5_per[0])
print('retrieval dust: ', dust_5[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con5[74])
print('+1.85 DN: ', dust_con5_per[74])
print('retrieval dust: ', dust_5[74])
print('------------------------------------')
print('CONDITION (6)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con6[0])
print('-1.85 DN: ', dust_con6_per[0])
print('retrieval dust: ', dust_6[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con6[74])
print('+1.85 DN: ', dust_con6_per[74])
print('retrieval dust: ', dust_6[74])
print('------------------------------------')


# %%
# -------------------------------------- Figure 5 --------------------------------------

color_pallet = ["tan","sandybrown", color_pallet[3], "chocolate", color_pallet[2], "black"]
label_tau = ["(1) τ = 0.02", "(2) τ = 0.3", "(3) τ = 1.2", "(4) τ = 2.7", "(5) τ = 3.9", "(6) τ = 6.6"]

# condition 1
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con1_all_result.sav"
sav_data1 = readsav(sav_fname1)
dust_1 = sav_data1["dust_result"]
base_1 = dust_1[10,50,37]

dust_histo_1 = dust_1.flatten()
dust_histo_11 = dust_histo_1 #- base_thin
stev1 = np.std(dust_histo_11)

# condition 2
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con2_all_result.sav"
sav_data2 = readsav(sav_fname2)
dust_2 = sav_data2["dust_result"]
base_2 = dust_2[10,50,37]

dust_histo_2 = dust_2.flatten()
dust_histo_21 = dust_histo_2 #- base_DS
stev2 = np.std(dust_histo_21)

# condition 3
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con3_all_result.sav"
sav_data3 = readsav(sav_fname3)
dust_3 = sav_data3["dust_result"]
base_3 = dust_3[10,50,37]

dust_histo_3 = dust_3.flatten()
dust_histo_31 = dust_histo_3 #- base_GDS
stev3 = np.std(dust_histo_31)

# condition 4
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname4 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con4_all_result.sav"
sav_data4 = readsav(sav_fname4)
dust_4 = sav_data4["dust_result"]
base_4 = dust_4[10,50,37]

dust_histo_4 = dust_4.flatten()
dust_histo_41 = dust_histo_4 #- base_GDS
stev4 = np.std(dust_histo_41)

# condition 5
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname5 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con5_all_result.sav"
sav_data5 = readsav(sav_fname5)
dust_5 = sav_data5["dust_result"]
base_5 = dust_5[10,50,37]

dust_histo_5 = dust_5.flatten()
dust_histo_51 = dust_histo_5 #- base_GDS
stev5 = np.std(dust_histo_51)

# condition 6
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname6 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con6_all_result.sav"
sav_data6 = readsav(sav_fname6)
dust_6 = sav_data6["dust_result"]
base_6 = dust_6[10,50,37]

dust_histo_6 = dust_6.flatten()
dust_histo_61 = dust_histo_6 #- base_GDS
stev6 = np.std(dust_histo_61)

fig1 = plt.figure(dpi=500)
ax = fig1.add_subplot(111)
ax.set_title(str(label_tau[0]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_11,bins=20,histtype='bar',color=color_pallet[0],edgecolor=color_pallet[0],alpha=0.3)
ax.axvline(x=base_1 + stev1,lw=2,color=color_pallet[0])
ax.axvline(x=base_1-stev1,lw=2,color=color_pallet[0])

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_title(str(label_tau[1]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_21,bins=20,histtype='bar',color=color_pallet[1],edgecolor=color_pallet[1],alpha=0.3)
ax.axvline(x=base_2 + stev2,lw=2,color=color_pallet[1])
ax.axvline(x=base_2 - stev2,lw=2,color=color_pallet[1])

fig3 = plt.figure(dpi=500)
ax = fig3.add_subplot(111)
ax.set_title(str(label_tau[2]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_31,bins=20,histtype='bar',color=color_pallet[2],edgecolor=color_pallet[2],alpha=0.3)
ax.axvline(x=stev3 + base_3,lw=2,color=color_pallet[2])
ax.axvline(x=base_3 - stev3,lw=2,color=color_pallet[2])

fig4 = plt.figure(dpi=500)
ax = fig4.add_subplot(111)
ax.set_title(str(label_tau[3]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_41,bins=20,histtype='bar',color=color_pallet[3],edgecolor=color_pallet[3],alpha=0.3)
ax.axvline(x=stev4 + base_4,lw=2,color=color_pallet[3])
ax.axvline(x=base_4 - stev4,lw=2,color=color_pallet[3])

fig5 = plt.figure(dpi=500)
ax = fig5.add_subplot(111)
ax.set_title(str(label_tau[4]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_51,bins=20,histtype='bar',color=color_pallet[4],edgecolor=color_pallet[4],alpha=0.3)
ax.axvline(x=stev5 + base_5,lw=2,color=color_pallet[4])
ax.axvline(x=base_5 - stev5,lw=2,color=color_pallet[4])

fig6 = plt.figure(dpi=500)
ax = fig6.add_subplot(111)
ax.set_title(str(label_tau[5]), fontsize=14)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_61,bins=20,histtype='bar',color=color_pallet[5],edgecolor=color_pallet[5],alpha=0.3)
ax.axvline(x=stev6 + base_6,lw=2,color=color_pallet[5])
ax.axvline(x=base_6 - stev6,lw=2,color=color_pallet[5])

print('CONDITION (1)', stev1)
print('CONDITION (2)', stev2)
print('CONDITION (3)', stev3)
print('CONDITION (4)', stev4)
print('CONDITION (5)', stev5)
print('CONDITION (6)', stev6)
# %%
# -------------------------------------- Figure 6_1 --------------------------------------
# Mathieuとの図を作成する

data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/evaluate_mathieu_points.sav"
sav_data = readsav(sav_fname)

com_data = sav_data["corre_two"]
Akira_277_before = com_data[0,:] + 0
Akira_277 = Akira_277_before * 1.12
Mathieu_slope = com_data[1,:] + 0

xerr = np.zeros(np.size(Akira_277))

# Total errorは(1)で0.032
ind1 = np.where(Akira_277 < 0.1)
xerr[ind1] = 0.32

# Total errorは(2)で0.034
ind2 = np.where((Akira_277 < 0.5) & (Akira_277 >= 0.1))
xerr[ind2] = 0.5

# Total errorは(3)で0.040
ind3 = np.where((Akira_277 < 1.2) & (Akira_277 >=0.5))
xerr[ind3] = 0.19

# Total errorは(3)で0.040
ind4 = np.where((Akira_277 < 3.0) & (Akira_277 >=1.2))
xerr[ind4] = 0.40

# Total errorは(3)で0.040
ind5 = np.where((Akira_277 < 5.0) & (Akira_277 >=3.0))
xerr[ind5] = 0.42

# Total errorは(3)で0.040
ind6 = np.where(Akira_277 >= 5)
xerr[ind6] = 0.47

# 相関をとる
coef = np.polyfit(Akira_277, Mathieu_slope, 1)
appr = np.poly1d(coef)(Akira_277)

# zeroのところにnanを入れる
#Akira_277[Akira_277 == 0] = np.nan


fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Dust optical depth at 0.88 μm using 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth at 0.88 μm using slope", fontsize=10)
ax.plot(Akira_277, Akira_277,  color = 'black', lw=0.5)
ax.errorbar(Akira_277, Mathieu_slope, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
#ax.plot(Akira_277, appr,  color = 'black', lw=0.5, zorder=1)
#ax.set_xlim(-0.3, 2)
#ax.set_ylim(-0.3, 2)

"""
# Mathieu_slopeとAkira_277の比のヒストグラムを作成
fig2 = plt.figure(dpi=800)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel("Dust optical depth using slope / Dust optical depth using 2.77 μm", fontsize=10)
ax2.set_ylabel("Number of points", fontsize=10)
ax2.hist(Mathieu_slope/Akira_277, bins=50, color='w', edgecolor='black')
"""

# -------------------------------------- Figure 6_2 --------------------------------------
# %%
# 着陸機の値と比較する
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/MER_site_dust_nan.sav"
sav_data = readsav(sav_fname)

dust_277 = sav_data["dust_tau_277"]
akira_result = dust_277 + 0

# nan dataを除外する
idx = ~np.isnan(dust_277)

akira_result = dust_277 + 0
akira_result = akira_result[idx]

file_name = sav_data["file_name"]

# Yann result
yann_result = np.loadtxt("/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/tauMER_norm610Pa_for_MERsites.txt")
yann_result = yann_result[idx]

xerr = np.zeros(np.size(akira_result))

coef = np.polyfit(akira_result, yann_result, 1)
cont0 = np.poly1d(coef)(akira_result)

# Total errorは(1)で0.032
ind1 = np.where(akira_result < 0.01)
akira_result[akira_result == 0] = 0.000000000001
xerr[ind1] = 0.32

# Total errorは(2)で0.034
ind2 = np.where((akira_result < 2) & (akira_result >= 0.01))
xerr[ind2] = 0.41

# Total errorは(3)で0.040
ind2 = np.where((akira_result < 2) & (akira_result >= 6))
xerr[ind2] = 0.47

# ind4
ind4 = np.where(akira_result >= 6)
xerr[ind4] = 0.67

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Dust optical depth using 2.77 μm", fontsize=10)
ax.set_ylabel("Dust optical depth at MER", fontsize=10)
ax.errorbar(akira_result*1.3, yann_result, xerr=xerr,capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.plot(akira_result, cont0,  color = 'black', lw=0.5, zorder=1)
#ax.plot(akira_result, akira_result,  color = 'red', lw=0.5, zorder=1)

# %%
# Section2のメモ帳
# 必要に応じて再度評価する場

# ================ Digital numberの評価 (特にDNが小さすぎる場所) ================
# Digital numberとRadianceの関係を見る
sav_fname12 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con1_rad_sum2.sav"
sav_data12 = readsav(sav_fname12)
rev_dn = sav_data12["re_dn"]
new_jdat = sav_data12["re_shift_jdat"]

sav_frame11 = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-2/data/con4_rad_rev.sav"
sav_data11 = readsav(sav_frame11)
rev_dn1 = sav_data11["re_dn"]
new_jdat1 = sav_data11["re_shift_jdat"]

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("raw digital number", fontsize=14)
ax.set_ylabel("Radiance", fontsize=14)
ax.scatter(rev_dn, new_jdat, color="black",s=2)

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("raw digital number", fontsize=14)
ax.set_ylabel("Radiance", fontsize=14)
ax.scatter(rev_dn1, new_jdat1, color="red",s=2)

# 縦軸のスケールを揃える
fig1_min = np.min(new_jdat)
fig1_max = np.max(new_jdat)
rn_fig1 = fig1_max - fig1_min

fig2_min = np.min(new_jdat1)
fig2_max = fig2_min + rn_fig1

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("raw digital number", fontsize=14)
ax.set_ylabel("Radiance", fontsize=14)
ax.scatter(rev_dn1, new_jdat1, color="black",s=2)
ax.set_ylim(fig1_min, fig2_max)

# %%
# summaition=1として評価をする

data_dir = pjoin(dirname(sio.__file__), "tests", "data")

sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con1_rad_result.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[37]
dust_con1_per = (dust_con1 * 100) / dust_thin[37]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con2_rad_result_sum1.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[37]
dust_con2_per = (dust_con2 * 100) / dust_DS[37]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con3_rad_result_sum1.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
dust_con3 = dust_GDS - dust_GDS[37]
dust_con3_per = (dust_con3 * 100) / dust_GDS[37]

# condition4を代入
sav_fname4 = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/con4_rad_result_sum1.sav"
sav_data4 = readsav(sav_fname4)
dust_thin_2 = sav_data4["dust_result"]
dust_con4 = dust_thin_2 - dust_thin_2[37]
dust_con4_per = (dust_con4 * 100) / dust_thin_2[37]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Signal noize deviation (DN)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(surface_pressure, dust_con1, color="tan", label="(1) τ = 0.02",s=2)
ax.scatter(surface_pressure, dust_con2, color="chocolate", label="(2) τ = 2.66",s=2)
ax.scatter(surface_pressure, dust_con3, color=color_pallet[2], label="(3) τ = 6.66",s=2)
ax.scatter(surface_pressure, dust_con4, color="black", label="(4) τ = 0.27",s=2)
ax.plot(surface_pressure, dust_con1, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_con2, color="chocolate", alpha=0.5)
ax.plot(surface_pressure, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(surface_pressure, dust_con4, color="black", alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Signal noise deviation (DN)", fontsize=12)
ax.set_ylabel("Dust optical depth", fontsize=12)
ax.scatter(surface_pressure, dust_thin, color="tan", label="(1) τ = 0.02",s=2)
ax.scatter(surface_pressure, dust_DS, color="chocolate", label="(2) τ = 2.66",s=2)
ax.scatter(surface_pressure, dust_GDS, color=color_pallet[2], label="(3) τ = 6.66",s=2)
ax.scatter(surface_pressure, dust_thin_2, color="black", label="(4) τ = 0.27",s=2)
ax.plot(surface_pressure, dust_thin, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_DS, color="chocolate",alpha=0.5)
ax.plot(surface_pressure, dust_GDS, color=color_pallet[2],alpha=0.5)
ax.plot(surface_pressure, dust_thin_2, color="black",alpha=0.5)
#ax.set_yscale('log')
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper left", fontsize=9)

print('------------------------------------')
print('BASE CONDITION')
print('dev', surface_pressure[37])
print('retreival dust (1): ',dust_thin[37])
print('retreival dust (2): ',dust_DS[37])
print('retreival dust (3): ',dust_GDS[37])
print('------------------------------------')
print('CONDITION (1)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con1[0])
print('-1.85 DN: ', dust_con1_per[0])
print('retrieval dust: ', dust_thin[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con1[74])
print('+1.85 DN: ', dust_con1_per[74])
print('retrieval dust: ', dust_thin[74])
print('------------------------------------')
print('CONDITION (2)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con2[0])
print('-1.85 DN: ', dust_con2_per[0])
print('retrieval dust: ', dust_DS[0])
print('   ')
print('dev', surface_pressure[74])
print('+50 Pa: ', dust_con2[74])
print('+50 Pa: ', dust_con2_per[74])
print('retrieval dust: ', dust_DS[74])
print('------------------------------------')
print('CONDITION (3)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con3[0])
print('-1.85 DN: ', dust_con3_per[0])
print('retrieval dust: ', dust_GDS[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con3[74])
print('+1.85 DN: ', dust_con3_per[74])
print('retrieval dust: ', dust_GDS[74])
print('------------------------------------')
print('CONDITION (4)')
print('dev', surface_pressure[0])
print('-1.85 DN: ', dust_con4[0])
print('-1.85 DN: ', dust_con4_per[0])
print('retrieval dust: ', dust_thin_2[0])
print('   ')
print('dev', surface_pressure[74])
print('+1.85 DN: ', dust_con4[74])
print('+1.85 DN: ', dust_con4_per[74])
print('retrieval dust: ', dust_thin_2[74])
print('------------------------------------')

# %%
