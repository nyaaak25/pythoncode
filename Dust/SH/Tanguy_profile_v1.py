
# ------------------------------------------
# Tanguy profileを確認するためのプログラム
# MMR→VMRに変換
# 2024/09/10 Tue 10:53 by Akira Kazama

# 2024.10.15 updated by AKira Kazama
# パラメーターを設定すれば諸々変更される仕様に変更
# 主に2.77 μm付近のデータのみに対応可能
# ------------------------------------------

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy.integrate import simps

# ------- set a parameter -------
# 波長をμmで指定する
input_wav = 2.77 # μm
input_ls = 225 # 15, 165, 225, 315
file_name = "277"
# ref_dop_ls315 = 1.7536012421816796 #test for ダスト感度テスト:: uniform condiitonを比較
ref_dop_ls315 = 1.1801242543377597 # for 277

# ------- calculate  -------
# そこから波数を計算する
input_wav_cm = 1/input_wav * 1e4 # cm-1

# ダウトの消散係数情報を取得する
Dust_exp = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/dust_2500-25000_mathieu.aero")
wv_ref = Dust_exp[:,0]
ext_ref = Dust_exp[:,1]

# input_wav_cmの値に補間を行う
ext_interp = np.interp(input_wav_cm, wv_ref, ext_ref)

Dust_ext = ext_interp*1e-3 # μmをcmに変換
# μm → mに直したのちにkmに変換
# もともとのunitはμm2, 数密度はcm-3

# Tanguy profileを読み込む
Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Tanguy/dust_profiles_ls" + str(input_ls) +".txt",skiprows=1)
alt = Ls15_profile[:,0]
# kmに変換
alt = alt / 1000
dust_MMR = Ls15_profile[:,1]

# Density sav fileを読み込む
sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Density/density_" + str(input_ls) +".sav"
sav_data = readsav(sav_fname)
Dens = sav_data["Density"]

# MMRからダスト質量密度に変換
dust_dens = dust_MMR * Dens

# さらにNumber densityに変換
# 1 dust particle = 1.0e-14 kg
dust_number_density = dust_dens / 1.0e-14

# m→cmに変換
dust_number_density_cm = dust_number_density * 1.0e-6

# 積分をして各層を足し合わせていく
optical_thickness = 0
for i in range(len(alt)):
    optical_thickness += Dust_ext * dust_number_density_cm[i] 
print("Dust Optical Thickness:", optical_thickness)

integral_value = simps(Dust_ext * dust_number_density_cm, alt)
print("Dust_ext と dust_number_density_cm の積分値:", integral_value)

dust_number_density_cm = dust_number_density_cm * (ref_dop_ls315 / integral_value)

# 積分して再計算する
# 積分をして各層を足し合わせていく
optical_thickness = 0
for i in range(len(alt)):
    optical_thickness += Dust_ext * dust_number_density_cm[i]
print("Dust Optical Thickness:", optical_thickness)

integral_value = simps(Dust_ext * dust_number_density_cm, alt)
print("Dust_ext と dust_number_density_cm の積分値:", integral_value)

# dataを保存する
save_data = np.column_stack((alt, dust_number_density_cm))
np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(input_ls) +"_v3.hc", save_data)

# plot
fig, ax = plt.subplots()
ax.scatter(dust_number_density_cm, alt, label="Dust Number Density", color="black")
# %%
# ---------------------------------------------------------
# Tanguy profileを読み込んで、Dust MMRとNumber densityをplotする
# Tanguyからのprofileだけをplotする
input_ls = [15,165,225,315]

fig, ax = plt.subplots()
ax.set_xlabel("Dust Mass Mixing Ratio")
ax.set_ylabel("Altitude [km]")

fig, ax2 = plt.subplots()
ax2.set_xlabel("Dust Number Density [cm^-3]")
ax2.set_ylabel("Altitude [km]")

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Tanguy/dust_profiles_ls" + str(input_ls[i]) +".txt",skiprows=1)
    alt = Ls15_profile[:,0]
    # kmに変換
    alt = alt / 1000

    dust_MMR = Ls15_profile[:,1]
    ax.plot(dust_MMR, alt, label="Ls" + str(input_ls[i]))

    # Density sav fileを読み込む
    sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Density/density_" + str(input_ls[i]) +".sav"
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
# ---------------------------------------------------------
# scallingしたものをplotする
# Tanguy number densityをscallingして、Dust Number Densityをplotする
input_ls = [15,165,225,315]
file_name = "277"

fig, ax2 = plt.subplots(dpi=300)
ax2.set_xlabel("Dust Number Density [cm^-3]", fontsize=14)
ax2.set_ylabel("Altitude [km]", fontsize=14)
ax2.set_title("(b) Assumed dust vertical profile", fontsize=14)

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(input_ls[i]) +"_v3.hc")
    alt = Ls15_profile[:,0]
    dust_number_density_cm = Ls15_profile[:,1]
    ax2.plot(dust_number_density_cm, alt, label="Ls" + str(input_ls[i]))
    data_alt = np.stack((alt, dust_number_density_cm), axis=1)
    np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure6/dust_nd_ls" + str(input_ls[i]) +".txt", data_alt)


ax2.legend()
# %%
# ---------------------------------------------------------
# DISORTを使って計算して結果をplotする
# output fileをplotする
input_ls = [15, 165, 225, 315]
index = np.zeros(len(input_ls))
ret = [0.68194666, 0.91090359, 1.3454840, 1.4329228]

fig, ax = plt.subplots(dpi=200)
ax.set_xlabel("Ls [deg]")
ax.set_ylabel("Radiance [W/m^2/sr]")
ax.set_title("Calculated radiance at 2.77 μm")

fig, ax2 = plt.subplots(dpi=300)
ax2.set_xlabel("Ls [deg]", fontsize=14)
ax2.set_ylabel("Retrieved dust optical depth", fontsize=14)
ax2.set_title("(a) Retrieved dust optical depth from 2.77 μm", fontsize=14)

for i in range(len(input_ls)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/output/"+file_name+"/Tanguy_ls" + str(input_ls[i]) +"_v3.dat")
    rad = Ls15_profile[1]
    ax.scatter(input_ls[i], rad, label="Ls" + str(input_ls[i]))
    print("Ls:", input_ls[i], "Radiance:", rad)
    index[i] = rad
    ax2.scatter(input_ls[i], ret[i], label="Ls" + str(input_ls[i])+" , tau = " + f"{ret[i]:.2f}")

ax.legend(loc="lower right")
ax2.legend(loc="lower right")

data_com = np.stack((input_ls, index), axis=1)
np.savetxt("/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure6/radiance.txt", data_com)
# %%