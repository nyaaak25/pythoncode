
# ------------------------------------------
# Tanguy profileを確認するためのプログラム
# MMR→VMRに変換
# 2024/09/10 Tue 10:53 by Akira Kazama
# ------------------------------------------

# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import readsav
import scipy.io as sio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# %%
# ダウトの消散係数情報を取得する
Dust_exp = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/dust_2500_8500.aero")
# 波数3601のデータを取得
Dust_ext = Dust_exp[1101,1] # 2.77 μm
Dust_ext = Dust_ext * 1e-3

# inputするLsを指定
input_ls = 165

# Tanguy profileを読み込む
Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/Tanguy/dust_profiles_ls" + str(input_ls) +".txt",skiprows=1)
alt = Ls15_profile[:,0]
# kmに変換
alt = alt / 1000
dust_MMR = Ls15_profile[:,1]

# Density sav fileを読み込む
sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/Density/density_" + str(input_ls) +".sav"
sav_data = readsav(sav_fname)
Dens = sav_data["Density"]

# MMRからダスト質量密度に変換
dust_dens = dust_MMR * Dens

# さらにNumber densityに変換
# 1 dust particle = 1.0e-14 kg
dust_number_density = dust_dens / 1.0e-14

# m→cmに変換
dust_number_density_cm = dust_number_density * 1.0e-6

# 光学的厚さをLs 315と同様になるように調整
# 2.362860339534096はLs 315の光学的厚さ
# 0.3761356087608193はLs 15の光学的厚さ
# 1.0213256841549032はLs 225の光学的厚さ
# 0.585345536875849はLs 165の光学的厚さ

#dust_number_density_cm = dust_number_density_cm * (2.362860339534096 / 0.3761356087608193)
#dust_number_density_cm = dust_number_density_cm * (2.362860339534096 / 1.0213256841549032)
dust_number_density_cm = dust_number_density_cm * (2.362860339534096 / 0.585345536875849)

# dataを保存する
save_data = np.column_stack((alt, dust_number_density_cm))
np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output_tanguy/Tanguy_ls" + str(input_ls) +".hc", save_data)

# 積分をして各層を足し合わせていく
optical_thickness = 0
for i in range(len(alt)):
    optical_thickness += Dust_ext * dust_number_density_cm[i]
print("Dust Optical Thickness:", optical_thickness)

# plot
fig, ax = plt.subplots()
ax.scatter(dust_number_density_cm, alt, label="Dust Number Density", color="black")

# %%
# Tanguyからのprofileだけをplotする
input_ls = [15,165,225,315]

fig, ax = plt.subplots()
ax.set_xlabel("Dust Mass Mixing Ratio")
ax.set_ylabel("Altitude [km]")

fig, ax2 = plt.subplots()
ax2.set_xlabel("Dust Number Density [cm^-3]")
ax2.set_ylabel("Altitude [km]")

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/Tanguy/dust_profiles_ls" + str(input_ls[i]) +".txt",skiprows=1)
    alt = Ls15_profile[:,0]
    # kmに変換
    alt = alt / 1000

    dust_MMR = Ls15_profile[:,1]
    ax.plot(dust_MMR, alt, label="Ls" + str(input_ls[i]))

    # Density sav fileを読み込む
    sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/Density/density_" + str(input_ls[i]) +".sav"
    sav_data = readsav(sav_fname)
    Dens = sav_data["Density"]

    # MMRからダスト質量密度に変換
    dust_dens = dust_MMR * Dens

    # さらにNumber densityに変換
    # 1 dust particle = 1.0e-14 kg
    dust_number_density = dust_dens / 1.0e-14

    # m→cmに変換
    dust_number_density_cm = dust_number_density * 1.0e-6
    ax2.plot(dust_number_density_cm, alt, label="Ls" + str(input_ls[i]))

ax.legend()
ax2.legend()
# %%
# scallingしたものをplotする
input_ls = [15,165,225,315]

fig, ax2 = plt.subplots()
ax2.set_xlabel("Dust Number Density [cm^-3]")
ax2.set_ylabel("Altitude [km]")

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output_tanguy/Tanguy_ls" + str(input_ls[i]) +".hc")
    alt = Ls15_profile[:,0]
    dust_number_density_cm = Ls15_profile[:,1]
    ax2.plot(dust_number_density_cm, alt, label="Ls" + str(input_ls[i]))

ax2.legend()
# %%
# output fileをplotする
input_ls = [15, 165, 225, 315]

fig, ax = plt.subplots()
ax.set_xlabel("Ls [deg]")
ax.set_ylabel("Radiance [W/m^2/sr]")

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output_tanguy/Tanguy_ls" + str(input_ls[i]) +"_2um.dat")
    rad = Ls15_profile[1]
    ax.scatter(input_ls[i], rad)
    print("Ls:", input_ls[i], "Radiance:", rad)
# %%
s