# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio


# %%
# Figure 2.1bのシフト量のtime variationをplotする
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop/shift_variation.sav"
sav_data = readsav(sav_fname)
orbit = sav_data["file_number"]
shift_value = sav_data["shift_amount"]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(b) Shift amount time variation")
ax.set_xlabel("Orbit number", fontsize=10)
ax.set_ylabel("Wavelength [μm]", fontsize=10)
ax.axhline(y=0.02, color="grey",zorder=1)
ax.axhline(y=-0.02, color="grey",zorder=2)
ax.scatter(orbit, shift_value, color="black",s=3,zorder=3)
ax.set_ylim(-0.075, 0.075)

# %%
# Figure 2.1aのスペクトルシフト問題を説明するためのplot
data_dir = pjoin(dirname(sio.__file__), "tests", "data")

# ORB0518_3
sav_fname1 = "/Users/nyonn/Desktop/ORB0518_3.sav"
sav_data1 = readsav(sav_fname1)
wvl = sav_data1["wvl"]
jdat_1= sav_data1["jdat"]
wvl_L = wvl[128:255]
flux_my27 = jdat_1[155, 128:255, 121]
jdat_my27 = flux_my27 + 0
jdat_my27[jdat_my27 < 0.0000000001] = np.nan

# ORB3198_5
sav_fname2 = "/Users/nyonn/Desktop/ORB3198_5.sav"
sav_data2 = readsav(sav_fname2)
jdat_2= sav_data2["jdat"]
flux_my28 = jdat_2[169, 128:255, 8]
jdat_my28 = flux_my28 + 0
jdat_my28[jdat_my28 < 0.0000000001] = np.nan

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111, title="(a) Comparison of MY27 and MY28 spectrum")
ax.set_xlabel("Wavelength [μm]", fontsize=10)
ax.set_ylabel("Radiance", fontsize=10)
ax.plot(wvl_L, jdat_my27, color="red", label="MY27: ORB0518_3")
ax.plot(wvl_L,jdat_my28, color="black", label="MY28: ORB3198_5")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper right", fontsize=9)


fig2 = plt.figure(dpi=800)
ax = fig2.add_subplot(111)
ax.plot(wvl_L, jdat_my27, color="red", label="MY27: ORB0518_3")
ax.plot(wvl_L,jdat_my28, color="black", label="MY28: ORB3198_5")
ax.scatter(wvl_L[12], jdat_my27[12], color="red")
ax.scatter(wvl_L[12],jdat_my28[12], color="black")
ax.set_xlim(2.65, 2.85)
ax.set_ylim(-0.01, 0.2)


# %%
# Figure 2,4-2,5を作成するプログラム
data_dir = pjoin(dirname(sio.__file__), "tests", "data")

# Temperature profile deviation
# condition1を代入
sav_fname1 = "/Users/nyonn/Desktop/con1_TA.sav"
sav_data1 = readsav(sav_fname1)
temp = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[10]
dust_con1_per = (dust_con1 * 100) / dust_thin[10]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/con2_TA.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[10]
dust_con2_per = (dust_con2 * 100) / dust_DS[10]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/con3_TA.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
dust_con3 = dust_GDS - dust_GDS[10]
dust_con3_per = (dust_con3 * 100) / dust_GDS[10]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(
    111, title="Influence of the Temperature profile uncertainties on the dust retrieval"
)
ax.set_xlabel("Temperature deviation (K)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation", fontsize=10)
ax.plot(temp, dust_con1, color="blue", label="(1) dust thin condition")
ax.plot(temp, dust_con2, color="red", label="(2) dust thick condition")
ax.plot(temp, dust_con3, color="black", label="(3) global dust storm period")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(
    111, title="Influence of the Temperature profile uncertainties on the dust retrieval"
)
ax.set_xlabel("Temperature deviation (K)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation (%)", fontsize=10)
ax.plot(temp, dust_con1_per, color="blue", label="(1) dust thin condition")
ax.plot(temp, dust_con2_per, color="red", label="(2) dust thick condition")
ax.plot(temp, dust_con3_per, color="black", label="(3) global dust storm period")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=9)


print('------------------------------------')
print('BASE CONDITION')
print('dev', temp[10])
print('retreival dust (1): ',dust_thin[10])
print('retreival dust (2): ',dust_DS[10])
print('retreival dust (3): ',dust_GDS[10])
print('------------------------------------')
print('CONDITION (1)')
print('dev', temp[0])
print('-10 K: ', dust_con1[0])
print('-10 K: ', dust_con1_per[0])
print('retrieval dust: ', dust_thin[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con1[20])
print('+10 K: ', dust_con1_per[20])
print('retrieval dust: ', dust_thin[20])
print('------------------------------------')
print('CONDITION (2)')
print('dev', temp[0])
print('-10 K: ', dust_con2[0])
print('-10 K: ', dust_con2_per[0])
print('retrieval dust: ', dust_DS[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con2[20])
print('+10 K: ', dust_con2_per[20])
print('retrieval dust: ', dust_DS[20])
print('------------------------------------')
print('CONDITION (3)')
print('dev', temp[0])
print('-10 K: ', dust_con3[0])
print('-10 K: ', dust_con3_per[0])
print('retrieval dust: ', dust_GDS[0])
print('   ')
print('dev', temp[20])
print('+10 K: ', dust_con3[20])
print('+10 K: ', dust_con3_per[20])
print('retrieval dust: ', dust_GDS[20])
print('------------------------------------')

# %%
# Surface pressure deviation
# condition1を代入
sav_fname1 = "/Users/nyonn/Desktop/con1_SP.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[50]
dust_con1_per = (dust_con1 * 100) / dust_thin[50]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/con2_SP.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[50]
dust_con2_per = (dust_con2 * 100) / dust_DS[50]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/con3_SP.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
dust_con3 = dust_GDS - dust_GDS[50]
dust_con3_per = (dust_con3 * 100) / dust_GDS[50]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(
    111, title="Influence of the surface pressure uncertainties on the dust retrieval"
)
ax.set_xlabel("Surface pressure deviation (Pa)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation", fontsize=10)
ax.plot(surface_pressure, dust_con1, color="blue", label="(1) dust thin condition")
ax.plot(surface_pressure, dust_con2, color="red", label="(2) dust thick condition")
ax.plot(surface_pressure, dust_con3, color="black", label="(3) global dust storm period")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(
    111, title="Influence of the surface pressure uncertainties on the dust retrieval"
)
ax.set_xlabel("Surface pressure deviation (Pa)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation (%)", fontsize=10)
ax.plot(surface_pressure, dust_con1_per, color="blue", label="(1) dust thin condition")
ax.plot(surface_pressure, dust_con2_per, color="red", label="(2) dust thick condition")
ax.plot(surface_pressure, dust_con3_per, color="black", label="(3) global dust storm period")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

print('------------------------------------')
print('BASE CONDITION')
print('dev', surface_pressure[50])
print('retreival dust (1): ',dust_thin[50])
print('retreival dust (2): ',dust_DS[50])
print('retreival dust (3): ',dust_GDS[50])
print('------------------------------------')
print('CONDITION (1)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con1[0])
print('-50 Pa: ', dust_con1_per[0])
print('retrieval dust: ', dust_thin[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con1[100])
print('+50 Pa: ', dust_con1_per[100])
print('retrieval dust: ', dust_thin[100])
print('------------------------------------')
print('CONDITION (2)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con2[0])
print('-50 Pa: ', dust_con2_per[0])
print('retrieval dust: ', dust_DS[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con2[100])
print('+50 Pa: ', dust_con2_per[100])
print('retrieval dust: ', dust_DS[100])
print('------------------------------------')
print('CONDITION (3)')
print('dev', surface_pressure[0])
print('-50 Pa: ', dust_con3[0])
print('-50 Pa: ', dust_con3_per[0])
print('retrieval dust: ', dust_GDS[0])
print('   ')
print('dev', surface_pressure[100])
print('+50 Pa: ', dust_con3[100])
print('+50 Pa: ', dust_con3_per[100])
print('retrieval dust: ', dust_GDS[100])
print('------------------------------------')
# %%

# Dust paper fugure 2.2作成用のプログラム
Dust_list = ["Dust=0.00", "Dust=0.01", "Dust=0.02", "Dust=0.03", "Dust=0.04", "Dust=0.05", "Dust=0.06", "Dust=0.07", "Dust=0.08", "Dust=0.09", "Dust=0.1"]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_xlabel("Wavenumber [μm]", fontsize=10)
ax.set_ylabel("Radiance", fontsize=10)

for i in range(0, 11, 3):
    ARS = np.loadtxt(
        "/Users/nyonn/Desktop/pythoncode/output/old/output/loc1_dust" + str(i) + ".dat"
    )
    ARS_x = ARS[:, 0]
    ARS_x = ARS_x[::-1]
    ARS_wav = 1 / ARS_x
    ARS_x = (1 / ARS_x) * 10000
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]
    ARS_y = (ARS_y / (ARS_wav * ARS_wav)) * 1e-7

    ax.plot(ARS_x, ARS_y, label=Dust_list[i], zorder=i, lw=3)
    ax.scatter(ARS_x[5], ARS_y[5])

ax.set_xlim(2.65, 2.85)
ax.set_ylim(0, 0.02)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=8)