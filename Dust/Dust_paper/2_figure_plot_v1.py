# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio

# %%
# Dust paper fugure1 作成用のプログラム
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
    ARS = np.loadtxt("/Users/nyonn/Desktop//論文/retrieval dust/2章：Describe the method/data/OMEGA-like/Test_loc2_dust" + str(i) + ".dat")
    ARS_x = ARS[:, 0]
    ARS_x = ARS_x[::-1]
    ARS_wav = 1 / ARS_x
    ARS_x = (1 / ARS_x) * 10000
    ARS_y = ARS[:, 1]
    ARS_y = ARS_y[::-1]
    ARS_y = (ARS_y / (ARS_wav * ARS_wav)) * 1e-7

    ax.plot(ARS_x, ARS_y, label=Dust_legend[i], zorder=i, lw=3)
    #ax.scatter(ARS_x[5], ARS_y[5])

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper left", fontsize=10)
ax.axvline(x=ARS_x[5], color="black", linestyle="dashed")
ax.set_ylim(0, 0.75)



# %%
# Figure 2.1bのシフト量のtime variationをplotする
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
#sav_fname = "/Users/nyonn/Desktop/ORB4588_4.sav"
sav_fname = "/Users/nyonn/Desktop/ORB1420_1.sav"
sav_data = readsav(sav_fname)
longitude = sav_data["longi"]
latitude = sav_data["lati"]
new_radiance = sav_data["new_rad"]
dust_opacity = sav_data["dust_opacity"]
altitude = sav_data["ret_alt_dust"]

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Retrieval dust opacity at 2.77 μm")
c = ax.contourf(longitude, latitude, dust_opacity, cmap="jet")
fig.colorbar(c, ax=ax, orientation="vertical", label="Dust opacity")
plt.axis('equal')
plt.show()

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Radiance at 2.77 μm")
ax.set_xlim(95, 100)
c = ax.contourf(longitude, latitude, new_radiance, cmap="jet",vmax=0.18,vmin=0.0)
fig2.colorbar(c, ax=ax, orientation="vertical", label="Radiance")
#plt.axis('equal')
plt.show()

fig3 = plt.figure(dpi=500)
ax = fig3.add_subplot(111)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Altitude")
c = ax.contourf(longitude, latitude, altitude, cmap="jet")
fig3.colorbar(c, ax=ax, orientation="vertical", label="altitude")
plt.axis('equal')
plt.show()

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
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con1_SP.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[50]
dust_con1_per = (dust_con1 * 100) / dust_thin[50]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con2_SP.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[50]
dust_con2_per = (dust_con2 * 100) / dust_DS[50]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con3_SP.sav"
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
# radiance variation
# condition1を代入
data_dir = pjoin(dirname(sio.__file__), "tests", "data")

sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con1_rad_result.sav"
sav_data1 = readsav(sav_fname1)
surface_pressure = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[37]
dust_con1_per = (dust_con1 * 100) / dust_thin[37]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con2_rad_result.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[37]
dust_con2_per = (dust_con2 * 100) / dust_DS[37]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con3_rad_result.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
dust_con3 = dust_GDS - dust_GDS[37]
dust_con3_per = (dust_con3 * 100) / dust_GDS[37]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(
    111, title="Influence of the signal noise uncertainties on the dust retrieval"
)
ax.set_xlabel("Signal noize deviation (DN)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation", fontsize=10)
ax.plot(surface_pressure, dust_con1, color="blue", label="(1) dust thin condition")
ax.plot(surface_pressure, dust_con2, color="red", label="(2) dust thick condition")
ax.plot(surface_pressure, dust_con3, color="black", label="(3) global dust storm period")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(
    111, title="Influence of the signal noise uncertainties on the dust retrieval"
)
ax.set_xlabel("Signal noise deviation (DN)", fontsize=10)
ax.set_ylabel("Dust optical depth deviation (%)", fontsize=10)
ax.plot(surface_pressure, dust_con1_per, color="blue", label="(1) dust thin condition")
ax.plot(surface_pressure, dust_con2_per, color="red", label="(2) dust thick condition")
ax.plot(surface_pressure, dust_con3_per, color="black", label="(3) global dust storm period")
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
# %%
# Figure 2,7の3つのrandom errorの足し合わせを行ったものをここで評価
# condition 1
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con1_all_result.sav"
sav_data1 = readsav(sav_fname1)
dust_thin = sav_data1["dust_result"]
base_thin = dust_thin[10,50,37]

dust_histo_1 = dust_thin.flatten()
dust_histo_11 = dust_histo_1 - base_thin
stev1 = np.std(dust_histo_11)

# condition 2
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con2_all_result.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
base_DS = dust_DS[10,50,37]

dust_histo_2 = dust_DS.flatten()
dust_histo_21 = dust_histo_2 - base_DS
stev2 = np.std(dust_histo_21)

# condition 3
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con3_all_result.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
base_GDS = dust_GDS[10,50,37]

dust_histo_3 = dust_GDS.flatten()
dust_histo_31 = dust_histo_3 - base_GDS
stev3 = np.std(dust_histo_31)

fig1 = plt.figure(dpi=500)
ax = fig1.add_subplot(111, title="(a) dust thin condition")
ax.set_xlabel("Dust optical depth deviation", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.hist(dust_histo_11,bins=20,range=(-0.15,0.15),histtype='bar',color='blue',edgecolor="blue",alpha=0.3)
ax.axvline(x=stev1,lw=2,color='blue')
ax.axvline(x=-stev1,lw=2,color='blue')

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111, title="(b) dust thick condition")
ax.set_xlabel("Dust optical depth deviation", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.hist(dust_histo_21,bins=20,range=(-0.15,0.15),histtype='bar',color='red',edgecolor="red",alpha=0.3)
ax.axvline(x=stev2,lw=2,color='red')
ax.axvline(x=-stev2,lw=2,color='red')

fig3 = plt.figure(dpi=500)
ax = fig3.add_subplot(111, title="(c) global dust storm period")
ax.set_xlabel("Dust optical depth deviation", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.hist(dust_histo_31,bins=20,range=(-0.15,0.15),histtype='bar',color='black',edgecolor="black",alpha=0.3)
ax.axvline(x=stev3,lw=2,color='black')
ax.axvline(x=-stev3,lw=2,color='black')

# %%
