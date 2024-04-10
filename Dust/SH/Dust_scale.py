
# %%
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama

import numpy as np
import matplotlib.pyplot as plt

HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/dust_ret/1d.hc")
altitude = HD_base[:,0]
density = HD_base[:,1]
new_density = np.zeros(len(density))

fig = plt.figure()
plt.plot(density, altitude, label="Original", color="black", linestyle="solid")
plt.ylabel("Altitude [km]")
plt.xlabel("Density [km-1]")
plt.title("Density profile of dust")

for i in range(len(density)):
#for i in range(10,11):
    new_density = density
    new_density[i] = density[i] + 1
    plt.plot(new_density, altitude, label="Scaled", linestyle="dashed")
    new_HD = np.column_stack((altitude, new_density))
    np.savetxt("/Users/nyonn/Desktop/pythoncode/dust_ret/HD/" +str(i)+ "d.hc", new_HD)

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
