
# %%
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama
# Dust_scale_v1.py;; 計算で出てきた高度依存性を確認するためのプログラム

import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["0.35", "3.5", "0.035"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function")
ax.set_ylabel("Altitude [km]")
ax.set_ylim(0, 75)

for loop in range(0, 3, 1):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    #ax.scatter(normarize, altitude, label=Dust_list[loop],s=10)
    ax.plot(normarize, altitude, label=Dust_list[loop])

ax.legend()
ax.grid()

# %%

# %%
Original = np.loadtxt("/Users/nyonn/Desktop/Test_loc2_dust1.dat")
wavelengh = Original[:,0]
radiance = Original[:,1]

wavelengh = wavelengh[::-1]
wav = 1/wavelengh
wavelengh = (1 / wavelengh) * 10000
radiance = radiance[::-1]
radiance = (radiance / (wav**2)) * 1e-7

modified = np.loadtxt("/Users/nyonn/Desktop/H_Test_loc2_dust1.dat")
modified_radiance = modified[:,1]
modified_radiance = modified_radiance[::-1]
modified_radiance = (modified_radiance / (wav**2)) * 1e-7

base_rad = modified_radiance[5]

HD0 = np.loadtxt("/Users/nyonn/Desktop/D0/0_rad.dat")
wave0 = HD0[:,0]
wave0 = wave0[::-1]
wav0 = 1/wave0
wave0 = (1 / wave0) * 10000

rad0 = HD0[:,1]
rad0 = rad0[::-1]
rad0 = (rad0 / (wav0**2)) * 1e-7

HD1 = np.loadtxt("/Users/nyonn/Desktop/1_rad.dat")
wave1 = HD1[:,0]
wave1 = wave1[::-1]
wav1 = 1/wave1
wave1 = (1 / wave1) * 10000

rad1 = HD1[:,1]
rad1 = rad1[::-1]
rad1 = (rad1 / (wav1**2)) * 1e-7

fig = plt.figure()
plt.ylabel("Radiance [W/m2/sr/um]")
plt.xlabel("Wavelength [um]")
plt.plot(wavelengh, radiance, label="Original", color="black", linestyle="solid")
plt.plot(wavelengh, modified_radiance, label="modified", linestyle="dashed")
plt.legend()
plt.show()
# %%
import glob

# Base radiance
modified = np.loadtxt("/Users/nyonn/Desktop/H_Test_loc2_dust1.dat")
modified_radiance = modified[:,1]
modified_radiance = modified_radiance[::-1]
modified_radiance = (modified_radiance / (wav**2)) * 1e-7
base_rad = modified_radiance[5]

# Dust profileを変えたときのもの
file_list = glob.glob("/Users/nyonn/Desktop/D0/*.dat")

# Baseの高度をここに入れる
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/dust_ret/1d.hc")
altitude = HD_base[:,0]
new_opacity = np.zeros(len(altitude))

for i in range(len(file_list)):
    HD = np.loadtxt(file_list[i])
    wave = HD[0]
    wav = 1/wave
    wave = (1 / wave) * 10000

    rad = HD[1]
    rad = (rad / (wav**2)) * 1e-7
    print(rad, wave)
    new_opacity[i] = rad - base_rad

# %%