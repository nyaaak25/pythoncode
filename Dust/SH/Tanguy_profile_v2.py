
# ------------------------------------------
# Tanguy profileを確認するためのプログラム
# MMR→VMRに変換
# 2024/09/10 Tue 10:53 by Akira Kazama

# 2024.10.15 updated by AKira Kazama
# パラメーターを設定すれば諸々変更される仕様に変更
# 波長だけ設定すれば全てを計算してくれる仕様を追加
# 2.0 μm付近のOMEGAの波長データを持ってきて、それに対応するファイルを作成する
# ------------------------------------------

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# ------- set a parameter -------
# OMEGAの波長を取得する
OMEGA_wav = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Soft/OMEGA_v/OMEGA_channnel_cm.dat')
wvn = OMEGA_wav
wav  = (1/ OMEGA_wav)*1e4

# wavを3桁の数字に変換する
wav_ind = np.round(wav,2)

# wav_indを四捨五入して、3桁の整数にする
wav_ind_after = np.round(wav_ind*100,0)

for i in range(len(wav_ind_after)):
    input_wav = wav_ind[i]
    file_name = str(int(wav_ind_after[i]))

    # ------- calculate  -------
    # そこから波数を計算する
    input_wav_cm = 1/input_wav * 1e4 # cm-1
    reference_ls = 315

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
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Tanguy/dust_profiles_ls" + str(reference_ls) +".txt",skiprows=1)
    alt = Ls15_profile[:,0]
    # kmに変換
    alt = alt / 1000
    dust_MMR = Ls15_profile[:,1]

    # Density sav fileを読み込む
    sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Density/density_" + str(reference_ls) +".sav"
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

    ref_dop_ls315  = optical_thickness

    # ここからLSを替えたものを計算する
    Ls_ind = [15, 165, 225, 315]

    for loop in range(len(Ls_ind)):
        Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Tanguy/dust_profiles_ls" + str(Ls_ind[loop]) +".txt",skiprows=1)
        alt = Ls15_profile[:,0]
        # kmに変換
        alt = alt / 1000
        dust_MMR = Ls15_profile[:,1]

        # Density sav fileを読み込む
        sav_fname = "/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/Density/density_" + str(Ls_ind[loop]) +".sav"
        sav_data = readsav(sav_fname)
        Dens = sav_data["Density"]

        # MMRからダスト質量密度に変換
        dust_dens = dust_MMR * Dens

        # さらにNumber densityに変換
        # 1 dust particle = 1.0e-14 kg
        dust_number_density = dust_dens / 1.0e-14

        # m→cmに変換
        dust_number_density_cm = dust_number_density * 1.0e-6
        optical_thickness = 0
        for i in range(len(alt)):
            optical_thickness += Dust_ext * dust_number_density_cm[i]
        print("Dust Optical Thickness:", optical_thickness)

        dust_number_density_cm = dust_number_density_cm * (ref_dop_ls315 / optical_thickness)
        optical_thickness = 0
        for i in range(len(alt)):
            optical_thickness += Dust_ext * dust_number_density_cm[i]
        print("Dust Optical Thickness:", optical_thickness)

        # dataを保存する
        save_data = np.column_stack((alt, dust_number_density_cm))
        #np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(Ls_ind[loop]) +".hc", save_data)

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# ------ input information -------
input_ls = [15,165,225,315]

# ------- set a parameter -------
# OMEGAの波長を取得する
OMEGA_wav = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Soft/OMEGA_v/OMEGA_channnel_cm.dat')
wvn = OMEGA_wav
wav  = (1/ OMEGA_wav)*1e4

# wavを3桁の数字に変換する
wav_ind = np.round(wav,2)

# wav_indを四捨五入して、3桁の整数にする
wav_ind_after = np.round(wav_ind*100,0)

for i in range(len(wav_ind_after)):
    input_wav = wav_ind[i]
    file_name = str(int(wav_ind_after[i]))

    fig, ax2 = plt.subplots()
    ax2.set_xlabel("Dust Number Density [cm^-3]")
    ax2.set_ylabel("Altitude [km]")
    ax2.set_title("Wavelength: " + str(input_wav) + " μm")

    for i in range(len(input_ls)):
        Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(input_ls[i]) +".hc")
        alt = Ls15_profile[:,0]
        dust_number_density_cm = Ls15_profile[:,1]
        ax2.plot(dust_number_density_cm, alt, label="Ls" + str(input_ls[i]))

    ax2.legend()
# %%
# ---------------------------------------------------------
# DISORTを使って計算して結果をplotする
# output fileをplotする
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

input_ls = [15, 165, 225, 315]

fig, ax = plt.subplots()
ax.set_xlabel("Wavelength [μm]")
ax.set_ylabel("Radiance [W/m^2/sr]")

fig, ax2 = plt.subplots()
ax2.set_xlabel("Ls [deg]")
ax2.set_ylabel("Radiance [W/m^2/sr]")

fig, ax3 = plt.subplots()
ax3.set_xlabel("Wavelength [μm]")
ax3.set_ylabel("Radiance [W/m^2/sr]")

# ------- set a parameter -------
# OMEGAの波長を取得する
OMEGA_wav = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Soft/OMEGA_v/OMEGA_channnel_cm.dat')
wvn = OMEGA_wav
wav  = (1/ OMEGA_wav)*1e4

# wavを3桁の数字に変換する
wav_ind = np.round(wav,2)

# wav_indを四捨五入して、3桁の整数にする
wav_ind_after = np.round(wav_ind*100,0)

# 新しいradを入れるための箱を作成
# 各Lsごとに計算をradを行っている
rad_all = np.zeros(len(wav_ind_after))
wvl_all = np.zeros(len(wav_ind_after))

for loop in range(len(input_ls)):
    for i in range(len(wav_ind_after)):
        input_wav = wav_ind[i]
        file_name = str(int(wav_ind_after[i]))
        Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/output/"+file_name+"/Tanguy_ls" + str(input_ls[loop]) +".dat")
        rad = Ls15_profile[1]
        wvl_rad = Ls15_profile[0]

        rad_all[i] = rad
        wvl_all[i] = wvl_rad
        
    # 波長と放射輝度をcmからumに変換する
    wvl_all_um = 1/wvl_all * 1e4
    wvl_all_um = wvl_all_um[::-1]

    rad_all_um = (rad_all/((1/wvl_all)**2))* 1e-7
    rad_all_um = rad_all_um[::-1]

    POLY_x = [wvl_all_um[0], wvl_all_um[3], wvl_all_um[5], wvl_all_um[23], wvl_all_um[24], wvl_all_um[25]]
    POLY_y = [rad_all_um[0], rad_all_um[3], rad_all_um[5], rad_all_um[23], rad_all_um[24], rad_all_um[25]]
    a, b = np.polyfit(POLY_x, POLY_y, 1)

    cont0 = b + a * wvl_all_um
    rad_calc = rad_all_um / cont0

    ax.plot(wvl_all_um, rad_all_um, label="Ls" + str(input_ls[loop]))
    #ax.scatter(wvl_all_um, rad_all_um, s=7)
    ax.scatter(POLY_x, POLY_y, s=30)
    ax.legend()

    ax3.plot(wvl_all_um, rad_calc, label="Ls" + str(input_ls[loop]))
    ax3.scatter(wvl_all_um, rad_calc, s=7)
    ax3.legend()

    data_save = np.stack((wvl_all_um, rad_calc), axis=-1)
    np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/2umband_" + str(input_ls[loop]) +".txt", data_save)

    rad_total = np.sum(rad_calc[8:21])
    #ax2.scatter(input_ls[loop], rad_total,label="Ls" + str(input_ls[loop]))
    ax2.scatter(input_ls[loop], rad_calc[15], label="Ls" + str(input_ls[loop]))
    #ax2.scatter(input_ls[loop], rad_total,label="amount of CO2 absorption", color="black")
    #ax2.scatter(input_ls[loop], rad_calc[15], label = '2.01 um', color="red")


    # rad_calc[6]とrad_calc[17]に直線を引く
    ax3.axvline(x=wvl_all_um[8], color="black", linestyle="--")
    ax3.axvline(x=wvl_all_um[21], color="black", linestyle="--")
    ax2.legend()



# %%