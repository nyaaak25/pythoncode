# %%
# dust retreival at 2.77のpaperで使用した図を作成するためのプログラム
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio

# %%
# ------------------------------------- Figure 2.1 -------------------------------------
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
# ------------------------------------- Figure 2.2 -------------------------------------
# Figure 2.2の放射輝度とダスト光学的厚さのプロット
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/2章：Describe the method/data/ORB1212_3.sav"
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
# ------------------------------------- Figure 2.3 -------------------------------------
# Figure 2.3を作成するプログラム
data_dir = pjoin(dirname(sio.__file__), "tests", "data")

# Temperature profile deviation
# condition1を代入
sav_fname1 = "/Users/nyonn/Desktop//論文/retrieval dust/2章：Describe the method/data/con1_TA.sav"
sav_data1 = readsav(sav_fname1)
temp = sav_data1["dev"]
dust_thin = sav_data1["dust_result"]
dust_con1 = dust_thin - dust_thin[10]
dust_con1_per = (dust_con1 * 100) / dust_thin[10]

# condition2を代入
sav_fname2 = "/Users/nyonn/Desktop//論文/retrieval dust/2章：Describe the method/data/con2_TA.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
dust_con2 = dust_DS - dust_DS[10]
dust_con2_per = (dust_con2 * 100) / dust_DS[10]

# condition3を代入
sav_fname3 = "/Users/nyonn/Desktop//論文/retrieval dust/2章：Describe the method/data/con3_TA.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
dust_con3 = dust_GDS - dust_GDS[10]
dust_con3_per = (dust_con3 * 100) / dust_GDS[10]

# absolute valueのplot
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Temperature deviation (K)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(temp, dust_con1, color="tan", label="(1) τ = 0.002",s=10)
ax.scatter(temp, dust_con2, color="chocolate", label="(2) τ = 0.25",s=10)
ax.scatter(temp, dust_con3, color="maroon", label="(3) τ = 0.7",s=10)
ax.plot(temp, dust_con1, color="tan", alpha=0.5)
ax.plot(temp, dust_con2, color="chocolate",alpha=0.5)
ax.plot(temp, dust_con3, color="maroon",alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=9)

# そのままのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Temperature deviation (K)", fontsize=12)
ax.set_ylabel("Dust optical depth", fontsize=12)
ax.scatter(temp, dust_thin, color="tan", label="(1) τ = 0.002",s=10)
ax.scatter(temp, dust_DS, color="chocolate", label="(2) τ = 0.25",s=10)
ax.scatter(temp, dust_GDS, color="maroon", label="(3) τ = 0.7",s=10)
ax.plot(temp, dust_thin, color="tan", alpha=0.5)
ax.plot(temp, dust_DS, color="chocolate",alpha=0.5)
ax.plot(temp, dust_GDS, color="maroon",alpha=0.5)
#ax.set_yscale('log')
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
# ------------------------------------- Figure 2.4 -------------------------------------
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
ax = fig.add_subplot(111)
ax.set_xlabel("Surface pressure deviation (Pa)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(surface_pressure, dust_con1, color="tan", label="(1) τ = 0.002",s=2)
ax.scatter(surface_pressure, dust_con2, color="chocolate", label="(2) τ = 0.25",s=2)
ax.scatter(surface_pressure, dust_con3, color="maroon", label="(3) τ = 0.7",s=2)
ax.plot(surface_pressure, dust_con1, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_con2, color="chocolate", alpha=0.5)
ax.plot(surface_pressure, dust_con3, color="maroon", alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Surface pressure deviation (Pa)", fontsize=10)
ax.set_ylabel("Dust optical depth", fontsize=10)
ax.scatter(surface_pressure, dust_thin, color="tan", label="(1) τ = 0.002",s=2)
ax.scatter(surface_pressure, dust_DS, color="chocolate", label="(2) τ = 0.25",s=2)
ax.scatter(surface_pressure, dust_GDS, color="maroon", label="(3) τ = 0.7",s=2)
ax.plot(surface_pressure, dust_thin, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_DS, color="chocolate",alpha=0.5)
ax.plot(surface_pressure, dust_GDS, color="maroon",alpha=0.5)
#ax.set_yscale('log')
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
# ------------------------------------- Figure 2.5 -------------------------------------
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
ax = fig.add_subplot(111)
ax.set_xlabel("Signal noize deviation (DN)", fontsize=12)
ax.set_ylabel("Dust optical depth deviation", fontsize=12)
ax.scatter(surface_pressure, dust_con1, color="tan", label="(1) τ = 0.002",s=2)
ax.scatter(surface_pressure, dust_con2, color="chocolate", label="(2) τ = 0.25",s=2)
ax.scatter(surface_pressure, dust_con3, color="maroon", label="(3) τ = 0.7",s=2)
ax.plot(surface_pressure, dust_con1, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_con2, color="chocolate", alpha=0.5)
ax.plot(surface_pressure, dust_con3, color="maroon", alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=9)

# Relative valueのplot
fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Signal noise deviation (DN)", fontsize=12)
ax.set_ylabel("Dust optical depth", fontsize=12)
ax.scatter(surface_pressure, dust_thin, color="tan", label="(1) τ = 0.002",s=2)
ax.scatter(surface_pressure, dust_DS, color="chocolate", label="(2) τ = 0.25",s=2)
ax.scatter(surface_pressure, dust_GDS, color="maroon", label="(3) τ = 0.7",s=2)
ax.plot(surface_pressure, dust_thin, color="tan", alpha=0.5)
ax.plot(surface_pressure, dust_DS, color="chocolate",alpha=0.5)
ax.plot(surface_pressure, dust_GDS, color="maroon",alpha=0.5)
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
# %%
# ------------------------------------- Figure 2.6 -------------------------------------
# Figure 2,7の3つのrandom errorの足し合わせを行ったものをここで評価
# condition 1
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con1_all_result.sav"
sav_data1 = readsav(sav_fname1)
dust_thin = sav_data1["dust_result"]
base_thin = dust_thin[10,50,37]

dust_histo_1 = dust_thin.flatten()
dust_histo_11 = dust_histo_1 #- base_thin
stev1 = np.std(dust_histo_11)

# condition 2
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con2_all_result.sav"
sav_data2 = readsav(sav_fname2)
dust_DS = sav_data2["dust_result"]
base_DS = dust_DS[10,50,37]

dust_histo_2 = dust_DS.flatten()
dust_histo_21 = dust_histo_2 #- base_DS
stev2 = np.std(dust_histo_21)

# condition 3
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname3 = "/Users/nyonn/Desktop/論文/retrieval dust/2章：Describe the method/data/con3_all_result.sav"
sav_data3 = readsav(sav_fname3)
dust_GDS = sav_data3["dust_result"]
base_GDS = dust_GDS[10,50,37]

dust_histo_3 = dust_GDS.flatten()
dust_histo_31 = dust_histo_3 #- base_GDS
stev3 = np.std(dust_histo_31)

fig1 = plt.figure(dpi=500)
ax = fig1.add_subplot(111)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_11,bins=20,histtype='bar',range=[-0.032,0.10],color='tan',edgecolor="tan",alpha=0.3)
ax.axvline(x=base_thin + stev1,lw=2,color='tan')
ax.axvline(x=base_thin-stev1,lw=2,color='tan')

fig2 = plt.figure(dpi=500)
ax = fig2.add_subplot(111)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_21,bins=20,histtype='bar',range=[0.15,0.37],color="chocolate",edgecolor="chocolate",alpha=0.3)
ax.axvline(x=base_DS + stev2,lw=2,color="chocolate")
ax.axvline(x=base_DS - stev2,lw=2,color="chocolate")

fig3 = plt.figure(dpi=500)
ax = fig3.add_subplot(111)
ax.set_xlabel("Dust optical depth", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.hist(dust_histo_31,bins=20,histtype='bar',range=[0.58,0.83],color="maroon",edgecolor="maroon",alpha=0.3)
ax.axvline(x=stev3 + base_GDS,lw=2,color="maroon")
ax.axvline(x=base_GDS - stev3,lw=2,color="maroon")

# %%
