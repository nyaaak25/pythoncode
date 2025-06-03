# %%
# --------------------------------------------------
# 2025.04.28 created by AKira Kazama
# Kazama et al, 2025の図を作成するためのプログラム
# revise version
# --------------------------------------------------
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from os.path import dirname, join as pjoin
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import glob

# %%
# ---------------------- Figure 1 (a) -----------------------------
# ORB0518_3
sav_fname1 = "/Users/nyonn/Desktop/論文/retrieval dust/old/2章：Describe the method/data/ORB0518_3.sav"
sav_data1 = readsav(sav_fname1)
wvl = sav_data1["wvl"]
jdat_1= sav_data1["jdat"]
wvl_L = wvl[128:255]
flux_my27 = jdat_1[155, 128:255, 121]
jdat_my27 = flux_my27 + 0
jdat_my27[jdat_my27 < 0.0000000001] = np.nan

# ORB3198_5
sav_fname2 = "/Users/nyonn/Desktop/論文/retrieval dust/old/2章：Describe the method/data/ORB3198_5.sav"
sav_data2 = readsav(sav_fname2)
jdat_2= sav_data2["jdat"]
flux_my28 = jdat_2[169, 128:255, 8]
jdat_my28 = flux_my28 + 0
jdat_my28[jdat_my28 < 0.0000000001] = np.nan

fig = plt.figure(dpi=300, figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_title("(a)", fontsize=26)
ax.set_xlabel("Wavelength [μm]", fontsize=22)
ax.set_ylabel("Radiance [W/m^2/sr]", fontsize=22)
ax.plot(wvl_L, jdat_my27, color="red", label="MY27: ORB0518_3")
ax.plot(wvl_L,jdat_my28, color="black", label="MY28: ORB3198_5")
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper right", fontsize=18)
ax.tick_params(axis='both', labelsize=20)

spectrum_data = np.stack((wvl_L, jdat_my27, jdat_my28), axis=-1)
spectrum_data = spectrum_data.T
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure1/spectrum_data.txt", spectrum_data.T)

fig2 = plt.figure(dpi=300, figsize=(6, 6))
ax = fig2.add_subplot(111)
ax.plot(wvl_L, jdat_my27, color="red", label="MY27: ORB0518_3")
ax.plot(wvl_L,jdat_my28, color="black", label="MY28: ORB3198_5")
ax.scatter(wvl_L[12], jdat_my27[12], color="red", s=80)
ax.scatter(wvl_L[12],jdat_my28[12], color="black", s=80)
ax.set_xlim(2.65, 2.85)
ax.set_xticks(np.arange(2.65, 2.85, 0.05))
ax.set_ylim(-0.01, 0.2)
ax.set_yticks(np.arange(0.01, 0.2, 0.05))
ax.tick_params(axis='both', labelsize=24, direction='in')

# %%
# ---------------------- Figure 1 (b) -----------------------------
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/old/2章：Describe the method/data/shift_variation.sav"
sav_data = readsav(sav_fname)
orbit = sav_data["file_number"]
shift_value = sav_data["shift_amount"]

fig = plt.figure(dpi=300, figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_title("(b)", fontsize=26)
ax.set_xlabel("Orbit number", fontsize=22)
ax.set_ylabel("Wavelength [μm]", fontsize=22)
ax.axhline(y=0.02, color="grey",zorder=1)
ax.axhline(y=-0.02, color="grey",zorder=2)
ax.scatter(orbit, shift_value, color="black",s=3,zorder=3)
ax.set_ylim(-0.075, 0.075)
ax.tick_params(axis='both', labelsize=20)

shift_data = np.stack((orbit, shift_value), axis=-1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure1/shift_variation.txt", shift_data)
# %%
# ------------------------ Figure 2 (b) -----------------------------
Dust_list  = 0.0 + np.arange(0, 1.6, 0.1)
Dust_legend = ["Dust=0.0", "Dust=0.1", "Dust=0.2", "Dust=0.3", "Dust=0.4", "Dust=0.5", "Dust=0.6", "Dust=0.7", "Dust=0.8", "Dust=0.9", "Dust=1.0", "Dust=1.1", "Dust=1.2", "Dust=1.3", "Dust=1.4", "Dust=1.5"]

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.set_xlabel("Wavenumber [μm]", fontsize=14)
ax.set_ylabel("Radiance [W/m2/sr/μm]", fontsize=14)

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
    ax.scatter(ARS_x, ARS_y, s=10)
    data = np.stack((ARS_x, ARS_y), axis=-1)
    np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure2/" + str(i) + ".txt", data)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="upper left", fontsize=14)
ax.axvline(x=ARS_x[5], color="black", linestyle="dashed")
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 0.75)
# %%
# ---------------------- Figure 4 -----------------------------
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
ax.set_xlabel("Longitude [deg]", fontsize=20)
ax.set_ylabel("Latitude [deg]", fontsize=20)
ax.set_title("(b)", fontsize=24)
c = ax.contourf(longitude, latitude, dust_opacity, cmap="jet",vmin=0)
ax.tick_params(axis='both', labelsize=16)
ax.set_yticks(np.arange(-20, 11, 10))

cbar = fig.colorbar(c, ax=ax, orientation="vertical")
cbar.set_label("Dust optical depth", fontsize=16)
#plt.axis('equal')
plt.show()

data_ret = np.stack((longitude, latitude, dust_opacity), axis=-1)
data_ret_2d = data_ret.reshape(-1, data_ret.shape[-1])
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure4/retrieval_dust.txt", data_ret_2d, fmt='%.6f')

fig2 = plt.figure(dpi=500)
axs = fig2.add_subplot(111)
axs.set_xlabel("Longitude [deg]", fontsize=20)
axs.set_ylabel("Latitude [deg]", fontsize=20)
axs.set_title("(a)", fontsize=24)
c = axs.contourf(longitude, latitude, new_radiance, cmap="jet",vmin=0)
axs.tick_params(axis='both', labelsize=16)
axs.set_yticks(np.arange(-20, 11, 10))

cbar2 = fig2.colorbar(c, ax=axs, orientation="vertical")
cbar2.set_label("Radiance  [W/m2/sr/μm]", fontsize=16)
#plt.axis('equal')
plt.show()
data_rad = np.stack((longitude, latitude, new_radiance), axis=-1)
data_rad_2d = data_rad.reshape(-1, data_rad.shape[-1])
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure4/raw_radiance.txt", data_rad_2d, fmt='%.6f')

# %%
# ---------------------- Figure 5 (a) -----------------------------
color_pallet = ["tan", "sandybrown", "peru", "chocolate", "maroon"]
label_tau = ["(1) τ = 0.3", "(2) τ = 1.2", "(3) τ = 2.7", "(4) τ = 3.9", "(5) τ = 6.6"]

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
ax.set_title("(a)", fontsize=24)
ax.set_xlabel("Temperature deviation (K)", fontsize=20)
ax.set_ylabel("Dust optical depth deviation", fontsize=17)
ax.scatter(temp, dust_con2, color=color_pallet[0], label=label_tau[0],s=10)
ax.scatter(temp, dust_con3, color=color_pallet[1], label=label_tau[1],s=10)
ax.scatter(temp, dust_con4, color=color_pallet[2], label=label_tau[2],s=10)
ax.scatter(temp, dust_con5, color=color_pallet[3], label=label_tau[3],s=10)
ax.scatter(temp, dust_con6, color=color_pallet[4], label=label_tau[4],s=10)
ax.plot(temp, dust_con2, color=color_pallet[0], alpha=0.5)
ax.plot(temp, dust_con3, color=color_pallet[1], alpha=0.5)
ax.plot(temp, dust_con4, color=color_pallet[2], alpha=0.5)
ax.plot(temp, dust_con5, color=color_pallet[3], alpha=0.5)
ax.plot(temp, dust_con6, color=color_pallet[4], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=13)
ax.tick_params(axis='both', labelsize=18)
ax.set_yticks(np.arange(-0.15, 0.16, 0.05))

data_t1 = np.stack([temp, dust_con2], axis=-1)
data_t2 = np.stack([temp, dust_con3], axis=-1)
data_t3 = np.stack([temp, dust_con4], axis=-1)
data_t4 = np.stack([temp, dust_con5], axis=-1)
data_t5 = np.stack([temp, dust_con6], axis=-1)

np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_t1.txt", data_t1)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_t2.txt", data_t2)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_t3.txt", data_t3)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_t4.txt", data_t4)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_t5.txt", data_t5)
# %%
# ---------------------- Figure 5 (b) -----------------------------
# Surface pressure deviation
color_pallet = ["black", "tan", "sandybrown", "peru", "chocolate", "maroon"]
label_tau = ["0", "(1) τ = 0.3", "(2) τ = 1.2", "(3) τ = 2.7", "(4) τ = 3.9", "(5) τ = 6.6"]

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
ax.set_title("(b)", fontsize=24)
ax.set_xlabel("Atmospheric pressure deviation (Pa)", fontsize=20)
ax.set_ylabel("Dust optical depth deviation", fontsize=17)
ax.scatter(surface_pressure, dust_con2, color=color_pallet[1], label=label_tau[1],s=2)
ax.scatter(surface_pressure, dust_con3, color=color_pallet[2], label=label_tau[2],s=2)
ax.scatter(surface_pressure, dust_con4, color=color_pallet[3], label=label_tau[3],s=2)
ax.scatter(surface_pressure, dust_con5, color=color_pallet[4], label=label_tau[4],s=2)
ax.scatter(surface_pressure, dust_con6, color=color_pallet[5], label=label_tau[5],s=2)
ax.plot(surface_pressure, dust_con2, color=color_pallet[1], alpha=0.5)
ax.plot(surface_pressure, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(surface_pressure, dust_con4, color=color_pallet[3], alpha=0.5)
ax.plot(surface_pressure, dust_con5, color=color_pallet[4], alpha=0.5)
ax.plot(surface_pressure, dust_con6, color=color_pallet[5], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=13)
ax.tick_params(axis='both', labelsize=18)
ax.set_yticks(np.arange(-0.5, 0.51, 0.25))

data_p1 = np.stack([surface_pressure, dust_con2], axis=-1)
data_p2 = np.stack([surface_pressure, dust_con3], axis=-1)
data_p3 = np.stack([surface_pressure, dust_con4], axis=-1)
data_p4 = np.stack([surface_pressure, dust_con5], axis=-1)
data_p5 = np.stack([surface_pressure, dust_con6], axis=-1)

np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_p1.txt", data_p1)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_p2.txt", data_p2)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_p3.txt", data_p3)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_p4.txt", data_p4)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_p5.txt", data_p5)
# %%
# ---------------------- Figure 5 (c) -----------------------------
# signal noise deviaition
color_pallet = ["black", "tan", "sandybrown", "peru", "chocolate", "maroon"]
label_tau = ["0", "(1) τ = 0.3", "(2) τ = 1.2", "(3) τ = 2.7", "(4) τ = 3.9", "(5) τ = 6.6"]

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
ax.set_title("(c)", fontsize=24)
ax.set_xlabel("Signal noise deviation (DN)", fontsize=20)
ax.set_ylabel("Dust optical depth deviation", fontsize=17)
ax.scatter(surface_pressure, dust_con2, color=color_pallet[1], label=label_tau[1],s=2)
ax.scatter(surface_pressure, dust_con3, color=color_pallet[2], label=label_tau[2],s=2)
ax.scatter(surface_pressure, dust_con4, color=color_pallet[3], label=label_tau[3],s=2)
ax.scatter(surface_pressure, dust_con5, color=color_pallet[4], label=label_tau[4],s=2)
ax.scatter(surface_pressure, dust_con6, color=color_pallet[5], label=label_tau[5],s=2)
ax.plot(surface_pressure, dust_con2, color=color_pallet[1], alpha=0.5)
ax.plot(surface_pressure, dust_con3, color=color_pallet[2], alpha=0.5)
ax.plot(surface_pressure, dust_con4, color=color_pallet[3], alpha=0.5)
ax.plot(surface_pressure, dust_con5, color=color_pallet[4], alpha=0.5)
ax.plot(surface_pressure, dust_con6, color=color_pallet[5], alpha=0.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower right", fontsize=13)
ax.tick_params(axis='both', labelsize=18)
ax.set_yticks(np.arange(-0.6, 0.61, 0.2))

data_s1 = np.stack([surface_pressure, dust_con2], axis=-1)
data_s2 = np.stack([surface_pressure, dust_con3], axis=-1)
data_s3 = np.stack([surface_pressure, dust_con4], axis=-1)
data_s4 = np.stack([surface_pressure, dust_con5], axis=-1)
data_s5 = np.stack([surface_pressure, dust_con6], axis=-1)

np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_s1.txt", data_s1)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_s2.txt", data_s2)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_s3.txt", data_s3)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_s4.txt", data_s4)
np.savetxt("/Users/nyonn/Desktop//論文/retrieval dust/Open-research/Figure5/data_s5.txt", data_s5)

# %%
# ---------------------- Figure 6 (a) -----------------------------
file_name = "277"
input_ls = [15, 165, 225, 315]
index = np.zeros(len(input_ls))
ret = [0.68194666, 0.91090359, 1.3454840, 1.4329228]

fig, ax2 = plt.subplots(dpi=300)
ax2.set_xlabel("Ls [deg]", fontsize=17)
ax2.set_ylabel("Retrieved dust optical depth", fontsize=17)
ax2.set_title("(a)", fontsize=24)

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/output/"+file_name+"/Tanguy_ls" + str(input_ls[i]) +"_v3.dat")
    rad = Ls15_profile[1]
    index[i] = rad
    ax2.scatter(input_ls[i], ret[i], label="tau = " + f"{ret[i]:.2f}")

ax2.legend(loc="upper left", fontsize=14)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_yticks(np.arange(0.6, 1.46, 0.17))
ax2.set_xticks(np.arange(10,320,60))

data_com = np.stack((input_ls, ret), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure6/retrieval_dust.txt", data_com)
# %%
# ---------------------- Figure 6 (b) -----------------------------
# scallingしたものをplotする
# scalling, ref_dop_ls315 = 1.1801242543377597
# Tanguy number densityをscallingして、Dust Number Densityをplotする
input_ls = [15,165,225,315]
file_name = "277"

fig, ax2 = plt.subplots(dpi=300)
ax2.set_xlabel("Dust Number Density [cm^-3]", fontsize=17)
ax2.set_ylabel("Altitude [km]", fontsize=17)
ax2.set_title("(b)", fontsize=24)

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(input_ls[i]) +"_v3.hc")
    alt = Ls15_profile[:,0]
    dust_number_density_cm = Ls15_profile[:,1]
    ax2.plot(dust_number_density_cm, alt, label="Ls=" + str(input_ls[i]))
    data_alt = np.stack((alt, dust_number_density_cm), axis=1)
    np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure6/dust_nd_ls" + str(input_ls[i]) +".txt", data_alt)
ax2.legend(loc="upper right", fontsize=14)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_yticks(np.arange(0, 81, 20))
ax2.set_xticks(np.arange(0,91,30))
# %%
# ---------------------- Figure 7 v1-1 -----------------------------
# もともとのdataを読み込む
sav_fname = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/evaluate_mathieu_points_v1_2.sav"
sav_data = readsav(sav_fname)

# plotのデータを読み込む
com_data = sav_data["corre_two"]

Akira_277_before = sav_data["mean_dust"] + 0 #com_data[0,:] + 0
Akira_277 = Akira_277_before * 1.12

Mathieu_slope = com_data[1,:] + 0

# 標準偏差のデータを読み込む
std_data_file = np.load("/Users/nyonn/Desktop/論文/retrieval dust/Sec-2/data/dust_std.npz")
std = std_data_file["dust_tau_std"]

Mathieu_slope[std == 0] = np.nan
Akira_277[std == 0] = np.nan
std[std == 0] = np.nan

valid_indices = ~np.isnan(Mathieu_slope) & ~np.isnan(Akira_277) & ~np.isinf(Mathieu_slope) & ~np.isinf(Akira_277)
Mathieu_slope = Mathieu_slope[valid_indices]
Akira_277 = Akira_277[valid_indices]
std = std[valid_indices]

# 相関をとる
coef = np.polyfit(Mathieu_slope, Akira_277,1,w=1/std)
appr = np.poly1d(coef)(Mathieu_slope)

# 一次関数y=ax+bの係数と誤差を求める
p,cov = np.polyfit(Mathieu_slope, Akira_277,1,w=1/std,cov=True)
a = p[0]
b = p[1]
sigma_a = np.sqrt(cov[0,0])
sigma_b = np.sqrt(cov[1,1])
print(sigma_a,sigma_b,a,b)

data = [np.min(Mathieu_slope), np.max(Mathieu_slope)]

# プロット
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.set_xlabel("Vincendon et al. 2009 retrievals [tau]", fontsize=14)
ax.set_ylabel("Our retrievals [tau]", fontsize=14)
ax.plot(Mathieu_slope, appr,  color = 'black', lw=1, zorder=1,label = "y = (2.1 ± 0.13)x + (-1.1 ± 0.18)")
ax.plot(data, data, color = 'black', lw=1, zorder=2, linestyle="dashed", label = "y = x")
ax.errorbar(Mathieu_slope, Akira_277, yerr = std, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
ax.legend(loc="upper left", fontsize=12)

# %%
# ---------------------- Figure 8 -----------------------------
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-3/data/dust_seasonal_rev.sav")
dust_color = all_data["dust_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(dust_color == 0)
dust_color = dust_color + 0
dust_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# color mapの設定
min_dust = 0.01
max_dust = 5.0
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

xticks = np.arange(0, 361, 60)
yticks = np.arange(-90, 91, 30)

# プロットをする
fig,axs = plt.subplots(3,1,dpi=300, figsize=(28, 20))
axs[0].set_title("MY27", fontsize=35)
axs[0].set_ylabel("Latitude [deg]", fontsize=35)
axs[0].set_ylim(-90, 90)
axs[0].set_yticks(np.arange(-90, 91, 30))
axs[0].set_xlim(0, 360)
axs[0].set_xticks(np.arange(0, 361, 60))
axs[0].tick_params(axis='both', labelsize=25)

my27_ind = derease_indices_all[1] - derease_indices_all[0]
my27_ls = np.zeros((my27_ind+1, 181))
my27_dust = np.zeros((my27_ind+1, 181))
my27_lat = np.zeros((my27_ind+1, 181))

for i in range(derease_indices_all[0] + 1, derease_indices_all[1], 1):
    color = sm.to_rgba(dust_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    axs[0].scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

    i27 = i - derease_indices_all[0]-1
    my27_ls[i27,:] = Ls_good[i]
    my27_dust[i27,:] = dust_good[:,0,i]
    my27_lat[i27,:] = lat_ind

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my27_ls.txt", my27_ls)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my27_dust.txt", my27_dust)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my27_lat.txt", my27_lat)

axs[1].set_title("MY28", fontsize=35)
axs[1].set_ylabel("Latitude [deg]", fontsize=35)
axs[1].set_ylim(-90, 90)
axs[1].set_yticks(np.arange(-90, 91, 30))
axs[1].set_xlim(0, 360)
axs[1].set_xticks(np.arange(0, 361, 60))
axs[1].tick_params(axis='both', labelsize=25)

my28_ls = np.zeros((derease_indices_all[2] - derease_indices_all[1]+1, 181))
my28_dust = np.zeros((derease_indices_all[2] - derease_indices_all[1]+1, 181))
my28_lat = np.zeros((derease_indices_all[2] - derease_indices_all[1]+1, 181))

for i in range(derease_indices_all[1] + 1, derease_indices_all[2], 1):
    color = sm.to_rgba(dust_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    axs[1].scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

    i28 = i - derease_indices_all[1]-1
    my28_ls[i28,:] = Ls_good[i]
    my28_dust[i28,:] = dust_good[:,0,i]
    my28_lat[i28,:] = lat_ind

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my28_ls.txt", my28_ls)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my28_dust.txt", my28_dust)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my28_lat.txt", my28_lat)

axs[2].set_title("MY29", fontsize=35)
axs[2].set_ylabel("Latitude [deg]", fontsize=35)
axs[2].set_xlabel("Solar Longitude [deg]", fontsize=35)
axs[2].set_ylim(-90, 90)
axs[2].set_yticks(np.arange(-90, 91, 30))
axs[2].set_xlim(0, 360)
axs[2].set_xticks(np.arange(0, 361, 60))
axs[2].tick_params(axis='both', labelsize=25)

my29_ls = np.zeros((count - derease_indices_all[2]+1, 181))
my29_dust = np.zeros((count - derease_indices_all[2]+1, 181))
my29_lat = np.zeros((count - derease_indices_all[2]+1, 181))

for i in range(derease_indices_all[2] + 1, count-1 ,1):
    color = sm.to_rgba(dust_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    axs[2].scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

    i29 = i - derease_indices_all[2]-1
    my29_ls[i29,:] = Ls_good[i]
    my29_dust[i29,:] = dust_good[:,0,i]
    my29_lat[i29,:] = lat_ind

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my29_ls.txt", my29_ls)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my29_dust.txt", my29_dust)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure8/my29_lat.txt", my29_lat)

cbar = plt.colorbar(sm,ax=axs.ravel().tolist(),orientation='vertical',aspect=90)
cbar.ax.tick_params(labelsize=30)
cbar.set_label("Dust optical depth at 2.0 μm", fontsize=30)

# %%
# ------------------------------------- Figure 10 -------------------------------------
# detection critriaについての図を作成する
# まずは生のリトリーバルデータをプロットする
# ------------------------------------- Figure 10a -------------------------------------
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/ORB1448_4-1.sav"
# 1448_4, 1339_1, 4482_3
sav_data = readsav(sav_fname)
longitude = sav_data["longi"]
latitude = sav_data["lati"]

dust_opacity = sav_data["dust_opacity"]
dust_opacity = dust_opacity + 0

ind = np.where(dust_opacity <= 0)
dust_opacity[ind] = np.nan
dust_opacity[:,80:96] = np.nan

# dust optical depth
min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [Deg]", fontsize=20)
ax.set_ylabel("Latitude [Deg]", fontsize=20)
im = ax.scatter(longitude, latitude, c=dust_opacity, cmap="jet", vmin=min_dust, vmax=max_dust)
cbar = plt.colorbar(im, orientation="vertical", extend='max')
cbar.ax.tick_params(labelsize=17)
cbar.set_label("Dust optical depth", fontsize=20)
ax.tick_params(axis='both', labelsize=17)

data_sav = np.stack((longitude, latitude, dust_opacity), axis=-1)
data_sav_2d = data_sav.reshape(-1, data_sav.shape[-1])
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure10/ORB1448_4-raw.txt", data_sav_2d)

# %%
# 1deg×1degマップを作成する
# ------------------------------------- Figure 10b -------------------------------------
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/ORB4482_3.sav"
sav_data = readsav(sav_fname)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [Deg]", fontsize=20)
ax.set_ylabel("Latitude [Deg]", fontsize=20)
im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
cbar = plt.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')
cbar.ax.tick_params(labelsize=17)
cbar.set_label("Dust optical depth", fontsize=20)
ax.tick_params(axis='both', labelsize=17)

data_bin = np.stack((lon, lat, med), axis=-1)
data_bin_2d = data_bin.reshape(-1, data_bin.shape[-1])
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure10/ORB1339_1-bin.txt", data_bin_2d)

# ------------------------------------- Figure 10c -------------------------------------
# %%
# detection/non-detection例を示す
sav_fname = "/Users/nyonn/Desktop//論文/retrieval dust/Sec-4/data/ORB1339_1.sav"
sav_data = readsav(sav_fname)

lat = sav_data["bin_lat"]
lon = sav_data["bin_lon"]
med = sav_data["bin_med"]

min_dust = 0.01
max_dust = 3.0

fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
ax.set_xlabel("Longitude [Deg]", fontsize=20)
ax.set_ylabel("Latitude [Deg]", fontsize=20)
# medの値が2.0以上のところは赤くプロットする
ind = np.where(med > 2.0)
ax.scatter(lon[ind], lat[ind], c="black",s=80)

im = ax.scatter(lon, lat, c=med, cmap="jet", vmin=min_dust, vmax=max_dust)
#ax.scatter(lon[ind], lat[ind], c="red")
cbar = plt.colorbar(im, orientation="vertical", label="Dust optical depth", extend='max')
cbar.ax.tick_params(labelsize=17)
cbar.set_label("Dust optical depth", fontsize=20)
ax.tick_params(axis='both', labelsize=17)

# %%
# ---------------------- Figure 11 a -----------------------------
# datac coverageをplotした図を作成する
# MY27のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 2.0
max_dust = 4.0
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY27_all_hist = Ls_good[derease_indices_all[0]+1:derease_indices_all[1]]
ax0.set_ylabel("Number of orbits", fontsize=22)
ax0.set_title("(a) MY27", fontsize=40)
ax0.set_xlim(0, 360)
ax0.set_xticks(np.arange(0, 361, 60))
ax0.set_yticks(np.arange(0, 126, 25))
ax0.tick_params(axis='both', labelsize=25)
ax0.hist(MY27_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_all_ls.txt", MY27_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=22)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)
ax1.set_yticks(np.arange(-90, 91, 45))
ax1.set_xticks(np.arange(0, 361, 60))
ax1.tick_params(axis='both', labelsize=25)

for i in range(derease_indices_all[0], derease_indices_all[1], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My27_all_localtime = local_good[:,0,derease_indices_all[0]+1:derease_indices_all[1]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_all_localtime.txt", My27_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=35)
ax2.set_ylabel("Latitude [Deg]", fontsize=22)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)
ax2.set_xticks(np.arange(0, 361, 60))
ax2.set_yticks(np.arange(-60, 61, 30))
ax2.tick_params(axis='both', labelsize=30)

my27_detect_lds = np.zeros((95))
my27_detect_lds_lc = np.zeros((95))
my27_detect_lds_ls = np.zeros((95))

for i in range(0,94,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)
    my27_detect_lds[i] = np.median(bin_med)
    my27_detect_lds_lc[i] = np.median(bin_lat)
    my27_detect_lds_ls[i] = Ls

my27_data_lds = np.stack((my27_detect_lds,my27_detect_lds_lc, my27_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY27_detect_lds.txt", my27_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Local time [h]", fontsize=25)


# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Dust optical depth", fontsize=25)
# %%
# %%
# ------------------------------------- Figure 11b -------------------------------------
# datac coverageをplotした図を作成する
# MY28のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 2.0
max_dust = 4.0
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY28_all_hist = Ls_good[derease_indices_all[1]+1:derease_indices_all[2]]
ax0.set_ylabel("Number of orbits", fontsize=22)
ax0.set_title("(b) MY28", fontsize=40)
ax0.set_xlim(0, 360)
ax0.set_xticks(np.arange(0, 361, 60))
ax0.set_yticks(np.arange(0, 126, 25))
ax0.tick_params(axis='both', labelsize=25)
ax0.hist(MY28_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_all_ls.txt", MY28_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=22)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)
ax1.set_yticks(np.arange(-90, 91, 45))
ax1.set_xticks(np.arange(0, 361, 60))
ax1.tick_params(axis='both', labelsize=25)

for i in range(derease_indices_all[1]+1, derease_indices_all[2], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My28_all_localtime = local_good[:,0,derease_indices_all[1]+1:derease_indices_all[2]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_all_localtime.txt", My28_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=35)
ax2.set_ylabel("Latitude [Deg]", fontsize=22)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)
ax2.set_xticks(np.arange(0, 361, 60))
ax2.set_yticks(np.arange(-60, 61, 30))
ax2.tick_params(axis='both', labelsize=30)

my28_detect_lds = np.zeros((35))
my28_detect_lds_lc = np.zeros((35))
my28_detect_lds_ls = np.zeros((35))

for i in range(95,130,1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

    my28_detect_lds[i-95] = np.median(bin_med)
    my28_detect_lds_lc[i-95] = np.median(bin_lat)
    my28_detect_lds_ls[i-95] = Ls

my28_data_lds = np.stack((my28_detect_lds,my28_detect_lds_lc,my28_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY28_detect_lds.txt", my28_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Local time [h]", fontsize=25)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Dust optical depth", fontsize=25)
# %%
# %%
# ------------------------------------- Figure 11c -------------------------------------
# datac coverageをplotした図を作成する
# MY29のデータをplotする

# 図の枠組みをここで定義する
# GridSpecを作成
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 0.02], height_ratios=[1, 1, 2])

# プロットを作成
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])

cax = fig.add_subplot(gs[1, 1])
cax2 = fig.add_subplot(gs[2, 1])

# カラーの定義
# local time
min_dust = 6
max_dust = 18
cmap = plt.get_cmap('jet')

norm = Normalize(vmin=min_dust, vmax=max_dust)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# dust optical depth
min_dust = 2
max_dust = 4
cmap = plt.get_cmap('jet')

norm2 = Normalize(vmin=min_dust, vmax=max_dust)
sm2 = ScalarMappable(cmap=cmap, norm=norm2)
sm2.set_array([])

# まずはdata coverageのデータを読み込む
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/figure4-1_3-1_detect.sav")
dust_color = all_data["dust_color"]
local_color = all_data["local_color"]
lat_ind = all_data["lat_ind"]
Ls_ind = all_data["Ls_ind"]

zero = np.where(local_color == 0)
local_color = local_color + 0
local_color[zero] = np.nan

ind_zero = np.where(Ls_ind == 0)
ind_good = np.where(Ls_ind > 0)
Ls_ind = Ls_ind + 0
Ls_ind[ind_zero] = np.nan

Ls_good = Ls_ind[ind_good]
dust_good = dust_color[:,ind_good]
local_good = local_color[:,ind_good]
count = np.size(Ls_good)
derease_indices_all = np.where(np.diff(Ls_good) < 0)[0]

# 全観測のヒストグラムを作成する
MY29_all_hist = Ls_good[derease_indices_all[2]+1:derease_indices_all[3]]
ax0.set_ylabel("Number of orbits", fontsize=22)
ax0.set_title("(c) MY29", fontsize=40)
ax0.set_xlim(0, 360)
ax0.set_xticks(np.arange(0, 361, 60))
ax0.set_yticks(np.arange(0, 126, 25))
ax0.tick_params(axis='both', labelsize=25)
ax0.hist(MY29_all_hist,bins=36,histtype='bar',color='blue',edgecolor='black')
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_all_ls.txt", MY29_all_hist)

# データカバレージのプロットをする
ax1.set_ylabel("Latitude [Deg]", fontsize=22)
ax1.set_ylim(-90, 90)
ax1.set_xlim(0, 360)
ax1.set_yticks(np.arange(-90, 91, 45))
ax1.set_xticks(np.arange(0, 361, 60))
ax1.tick_params(axis='both', labelsize=25)

for i in range(derease_indices_all[2]+1, derease_indices_all[3], 1):
    color = sm.to_rgba(local_good[:,0,i])
    LS_plt = np.repeat(Ls_good[i], len(lat_ind))
    ax1.scatter(LS_plt, lat_ind, c=color, cmap="viridis", s=1, zorder=3)

My29_all_localtime = local_good[:,0,derease_indices_all[2]+1:derease_indices_all[3]]
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_all_localtime.txt", My29_all_localtime)

# 検出されたLDSをplotする
path_work = "/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW-detect/"
# EW-detectのファイルを読み込む
file_pattern = path_work + '*.sav'

# Use glob to find files that match the pattern
files = glob.glob(file_pattern)
files = sorted(files)
count = len(files)

ax2.set_xlabel("Solar longitude [Deg]", fontsize=35)
ax2.set_ylabel("Latitude [Deg]", fontsize=22)
ax2.set_xlim(0, 360)
ax2.set_ylim(-65, 65)
ax2.set_xticks(np.arange(0, 361, 60))
ax2.set_yticks(np.arange(-60, 61, 30))
ax2.tick_params(axis='both', labelsize=30)

my29_detect_lds = np.zeros((15))
my29_detect_lds_lc = np.zeros((15))
my29_detect_lds_ls = np.zeros((15))

for i in range(131, count, 1):
    # ファイルを読み込む
    data = readsav(files[i])
    # ファイルからデータを取り出す
    bin_lat = data['bin_lat']
    bin_lon = data['bin_lon']
    bin_med = data['bin_med']
    Ls = data['LS']

    LDS_ind = np.where(bin_med > 2.0)
    bin_lat = bin_lat[LDS_ind]
    med_lat = np.median(bin_lat)
    bin_med = bin_med[LDS_ind]

    color = sm2.to_rgba(np.median(bin_med))
    ax2.scatter(Ls,med_lat, color=color)

    my29_detect_lds[i-131] = np.median(bin_med)
    my29_detect_lds_lc[i-131] = np.median(bin_lat)
    my29_detect_lds_ls[i-131] = Ls

my29_data_lds = np.stack((my29_detect_lds,my29_detect_lds_lc,my29_detect_lds_ls),axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/MY29_detect_lds.txt", my29_data_lds)

# カラーバーを作成
cbar = plt.colorbar(sm,cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Local time [h]", fontsize=25)

# カラーバーを作成
cbar = plt.colorbar(sm2,cax=cax2, orientation='vertical')
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Dust optical depth", fontsize=25)
# %%
# ------------------------------------- Figure 12a -------------------------------------
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']

data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/coverage.sav")
# データを取り出す
ALL_ls = data['Ls_ind']
ALL_lt = data['LT_ind']
ALL_lat = data['lat_ind']

# LS 0-180度における領域の時間軸ヒストグラムを作成する
all_spring_summer = np.where((ALL_ls >= 0) & (ALL_ls < 180))
obs_spring_summer = np.where((OBS_ls >= 0) & (OBS_ls < 180))

ALL_ls_ss = ALL_ls[all_spring_summer]
ALL_lt_ss = ALL_lt[all_spring_summer]
OBS_ls_ss = OBS_ls[obs_spring_summer]
OBS_lt_ss = OBS_lt[obs_spring_summer]

ls_obs_ss_hist = OBS_ls_ss.flatten()
ls_all_ss_hist = ALL_ls_ss.flatten()
lt_obs_ss_hist = OBS_lt_ss.flatten()
lt_all_ss_hist = ALL_lt_ss.flatten()

# からの配列を作成
obs_ss_hist = np.zeros(24)
all_ss_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    ind_ss_obs = np.where((OBS_lt_ss >= loop) & (OBS_lt_ss < loop+1))
    obs_ss_hist[loop] = len(OBS_lt_ss[ind_ss_obs])

    ind_ss_all = np.where((ALL_lt_ss >= loop) & (ALL_lt_ss < loop+1))
    all_ss_hist[loop] = len(ALL_lt_ss[ind_ss_all])

# all_ss_histがゼロでない場合のみ確率を計算する
pro_ss = np.zeros(24)
nonzero_indices = all_ss_hist > 0
pro_ss[nonzero_indices] = (obs_ss_hist[nonzero_indices] / all_ss_hist[nonzero_indices]) * 100

# エラーバーの計算（確率を0-1に変換）
er_bar = np.zeros(24)
p = pro_ss[nonzero_indices] / 100  # パーセンテージから確率に変換
n = all_ss_hist[nonzero_indices]
er_bar[nonzero_indices] = np.sqrt(p * (1 - p) / n) * 100  # 百分率に戻す

# 0から24までの数字の配列を作成する
time = np.arange(24)

# エラーバーにNaNが含まれていないか確認
print("pro_ss:", pro_ss)
print("er_bar:", er_bar)

# エラー処理としてゼロでないデータを表示
ind1 = np.where(obs_ss_hist > 0)
print('detection;', obs_ss_hist[ind1])
print('Observation', all_ss_hist[ind1])

# グラフを描画
fig, axs = plt.subplots(dpi=500, figsize=(5, 5))  # DPIとサイズを調整
axs.set_title("(a) Ls = 0°-180°", fontsize=22)
axs.set_xlabel("Local time [h]", fontsize=20)
axs.set_ylabel("Probability [%]", fontsize=20)
axs.tick_params(axis='both', labelsize=16)
#axs.bar(range(0, 24, 1), pro_ss, color='red', edgecolor='black', label='Probability')

# サンプル数が200以下の場合は、ビンの色を変更する
for i in range(0, 24, 1):
    if all_ss_hist[i] < 200:
        axs.bar(i, pro_ss[i], color='red', edgecolor='black', alpha=0.3)
    if all_ss_hist[i] > 200:
        axs.bar(i, pro_ss[i], color='red', edgecolor='black')

# エラーバーを描画（ゼロの部分はスキップ）
axs.errorbar(range(0, 24, 1), pro_ss, yerr=er_bar, color='black', ecolor='black', fmt='o', capsize=5, label='Error Bars')
axs.set_xlim(6, 18)
axs.set_ylim(0, 5)
axs.set_xticks(np.arange(6, 19, 2))
axs.legend(loc='upper right', fontsize=16)
plt.show()

ss_data = np.stack((time,all_ss_hist, obs_ss_hist), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_probability.txt", ss_data)

# %%
# ------------------------------------- Figure 12b -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ALL_EW_detect_0-180_lon_lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW_detect_0-180_lon_lat.sav")
OBS_info = OBS_data['detect_number']

# データを転置する
ALL_info = ALL_info.T
OBS_info = OBS_info.T

# 経度の配列を作成する
# 配列型は(181,361)で、緯度は-90から90まで、経度は0から360まで
longi = np.arange(361) * 1
Longi = np.tile(longi, (181, 1))

latitude = np.arange(181) * 1 - 90
Lati = np.tile(latitude, (361, 1))
Lati = np.transpose(Lati)

# color mapの設定
cmap = plt.get_cmap('Reds')
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=15))
sm.set_array([])

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("(b) Ls = 0°-180°", fontsize=20)
ax.set_xlabel("Longitude [Deg]", fontsize=18)
ax.set_ylabel("Latitude [Deg]", fontsize=18)
ax.set_xlim(-185, 195)
ax.set_ylim(-100, 100)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.tick_params(axis='both', labelsize=14)

# 15°×15°の格子点を作成する
smooth_all = np.zeros((13, 24))
smooth_obs = np.zeros((13, 24))

smooth_lati = np.arange(13)*15 - 90
smooth_lat = np.tile(smooth_lati, (25, 1))
smooth_lat = np.transpose(smooth_lat)

smooth_longi = np.arange(25)*15
smooth_long = np.tile(smooth_longi, (13, 1))

for i in range(0, 13, 1):
    for j in range(0, 24, 1):
        ind_smooth= np.where((Lati >= smooth_lati[i]) & (Lati < smooth_lati[i] + 15) & (Longi >= smooth_longi[j]) & (Longi < smooth_longi[j] + 15))

        smooth_all[i, j] = np.sum(ALL_info[ind_smooth])
        smooth_obs[i, j] = np.sum(OBS_info[ind_smooth])

# ALL_infoが0の場所と0でない場所を特定します
zero_indices = np.where(smooth_all == 0)
non_zero_indices = np.where(smooth_all != 0)

# longitudeの値を-180°から180°に変換する
smooth_long = np.where(smooth_long > 180, smooth_long - 360, smooth_long)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(smooth_long[zero_indices], smooth_lat[zero_indices], color="grey", alpha=0.2, s=100, marker='s')
RATIO_map = (smooth_obs[non_zero_indices] / smooth_all[non_zero_indices]) * 100
colors = sm.to_rgba(RATIO_map.flatten())
ax.scatter(smooth_long[non_zero_indices], smooth_lat[non_zero_indices], color=colors,marker='s', s=100)

# カラーバーを作成
cbar = plt.colorbar(sm)
cbar.set_label("Probability of detection [%]", fontsize=16)
cbar.ax.tick_params(labelsize=12)
plt.show()

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_all-observation-spatial.txt", smooth_all)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls0-180_observation-spatial.txt", smooth_obs)

# %%
# ------------------------------------- Figure 12 c -------------------------------------
# %%
# LDSが検出された季節におけるヒストグラムを作成する
data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/histo_info.sav")
# データを取り出す
OBS_ls = data['obs_Ls_ind']
OBS_lt = data['obs_lt_ind']
OBS_lat = data['obs_lat_ind']

data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/coverage.sav")
# データを取り出す
ALL_ls = data['Ls_ind']
ALL_lt = data['LT_ind']
ALL_lat = data['lat_ind']

# Ls 180 - 360°のヒストグラムを作成する
all_autumn_winter = np.where((ALL_ls >= 180) & (ALL_ls < 360))
obs_autumn_winter = np.where((OBS_ls >= 180) & (OBS_ls < 360))

ALL_ls_aw = ALL_ls[all_autumn_winter]
ALL_lt_aw = ALL_lt[all_autumn_winter]
OBS_ls_aw = OBS_ls[obs_autumn_winter]
OBS_lt_aw = OBS_lt[obs_autumn_winter]

ls_obs_aw_hist = OBS_ls_aw.flatten()
ls_all_aw_hist = ALL_ls_aw.flatten()
lt_obs_aw_hist = OBS_lt_aw.flatten()
lt_all_aw_hist = ALL_lt_aw.flatten()

# からの配列を作成
obs_aw_hist = np.zeros(24)
all_aw_hist = np.zeros(24)

# local timeでの検出率を求める
for loop in range(0,24,1):
    ind_aw_obs = np.where((OBS_lt_aw >= loop) & (OBS_lt_aw < loop+1))
    obs_aw_hist[loop] = len(OBS_lt_aw[ind_aw_obs])

    ind_aw_all = np.where((ALL_lt_aw >= loop) & (ALL_lt_aw < loop+1))
    all_aw_hist[loop] = len(ALL_lt_aw[ind_aw_all])

# all_aw_histがゼロでない場合のみ確率を計算する
pro_aw = np.zeros(24)
nonzero_indices = all_aw_hist > 0
pro_aw[nonzero_indices] = (obs_aw_hist[nonzero_indices] / all_aw_hist[nonzero_indices]) * 100

# エラーバーの計算（確率を0-1に変換）
er_bar = np.zeros(24)
p = pro_aw[nonzero_indices] / 100  # パーセンテージから確率に変換
n = all_aw_hist[nonzero_indices]
er_bar[nonzero_indices] = np.sqrt(p * (1 - p) / n) * 100  # 百分率に戻す

# エラーバーにNaNが含まれていないか確認
print("pro_aw:", pro_aw)
print("er_bar:", er_bar)

# エラー処理としてゼロでないデータを表示
ind1 = np.where(obs_aw_hist > 0)
print('detection;', obs_aw_hist[ind1])
print('Observation', all_aw_hist[ind1])

# 0から24の数字の配列を作成する
time = np.arange(24)

# グラフを描画
fig, axs = plt.subplots(dpi=800, figsize=(5, 5))  # DPIとサイズを調整
axs.set_title("(c) Ls = 180°-360°", fontsize=22)
axs.set_xlabel("Local time [h]", fontsize=20)
axs.set_ylabel("Probability [%]", fontsize=20)
 # x軸の目盛りを3時間ごとに設定
axs.tick_params(axis='both', labelsize=16)

# サンプル数が200以下の場合は、ビンの色を変更する
for i in range(0, 24, 1):
    if all_aw_hist[i] < 200:
        axs.bar(i, pro_aw[i], color='blue', edgecolor='black', alpha=0.3)
    if all_aw_hist[i] > 200:
        axs.bar(i, pro_aw[i], color='blue', edgecolor='black')

# エラーバーを描画（ゼロの部分はスキップ）
axs.errorbar(range(0, 24, 1), pro_aw, yerr=er_bar, color='black', ecolor='black', fmt='o', capsize=5, label='Error Bars')

axs.set_xlim(6, 18)
axs.set_ylim(0, 15)
axs.set_xticks(np.arange(6, 19, 2))  # x軸の目盛りを3時間ごとに設定
axs.legend(loc='upper right', fontsize=16)
plt.show()

aw_data = np.stack((time, all_aw_hist, obs_aw_hist), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_probability.txt", aw_data)

# %%    
# ------------------------------------- Figure 12 d -------------------------------------
# MY27-29のLDSの分布を作成する
all_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/ALL_EW_detect_180-360_lon_lat.sav")
ALL_info = all_data['detect_number']

# observation dataを読み込む
OBS_data = readsav("/Users/nyonn/Desktop/論文/retrieval dust/Sec-4/data/EW_detect_180-360_lon_lat.sav")
OBS_info = OBS_data['detect_number']

# データを転置する
ALL_info = ALL_info.T
OBS_info = OBS_info.T

# 経度の配列を作成する
# 配列型は(181,361)で、緯度は-90から90まで、経度は0から360まで
longi = np.arange(361) * 1
Longi = np.tile(longi, (181, 1))

latitude = np.arange(181) * 1 - 90
Lati = np.tile(latitude, (361, 1))
Lati = np.transpose(Lati)

# color mapの設定
cmap = plt.get_cmap('Blues')
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=25))
sm.set_array([])

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("(d) Ls = 180°-360°", fontsize=20)
ax.set_xlabel("Longitude [Deg]", fontsize=18)
ax.set_ylabel("Latitude [Deg]", fontsize=18)
ax.set_xlim(-185, 195)
ax.set_ylim(-100, 100)
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.tick_params(axis='both', labelsize=14)

# 15°×15°の格子点を作成する
smooth_all = np.zeros((13, 24))
smooth_obs = np.zeros((13, 24))

smooth_lati = np.arange(13)*15 - 90
smooth_lat = np.tile(smooth_lati, (25, 1))
smooth_lat = np.transpose(smooth_lat)

smooth_longi = np.arange(25)*15
smooth_long = np.tile(smooth_longi, (13, 1))

for i in range(0, 13, 1):
    for j in range(0, 24, 1):
        ind_smooth= np.where((Lati >= smooth_lati[i]) & (Lati < smooth_lati[i] + 15) & (Longi >= smooth_longi[j]) & (Longi < smooth_longi[j] + 15))

        smooth_all[i, j] = np.sum(ALL_info[ind_smooth])
        smooth_obs[i, j] = np.sum(OBS_info[ind_smooth])

# ALL_infoが0の場所と0でない場所を特定します
zero_indices = np.where(smooth_all == 0)
non_zero_indices = np.where(smooth_all != 0)

# longitudeの値を-180°から180°に変換する
smooth_long = np.where(smooth_long > 180, smooth_long - 360, smooth_long)

# ALL_infoが0の場所をグレーでプロットします
ax.scatter(smooth_long[zero_indices], smooth_lat[zero_indices], color="grey", alpha=0.2, s=100, marker='s')
RATIO_map = (smooth_obs[non_zero_indices] / smooth_all[non_zero_indices]) * 100
colors = sm.to_rgba(RATIO_map.flatten())
ax.scatter(smooth_long[non_zero_indices], smooth_lat[non_zero_indices], color=colors,marker='s', s=100)

np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/smooth_longitude.txt", smooth_long)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/smooth_latitude.txt", smooth_lat)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_all-observation-spatial.txt", smooth_all)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/ls180-360_detection-spatial.txt", smooth_obs)

# カラーバーを作成
cbar = plt.colorbar(sm)
cbar.set_label("Probability of detection [%]", fontsize=16)
cbar.ax.tick_params(labelsize=12)
plt.show()
# %%
