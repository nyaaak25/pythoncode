
# ------------------------------------------
# 各高度に感度のあるダストのプロファイルを作成する
# 2025/03/14 Fri 14:34:00 created by AKira Kazama

# Tanguy profileのLs15°データを元に、各高度に感度のあるダストのプロファイルを作成する
# sigmaで広がり、amlituteで強さ、h_peakでピークの高度を設定する
# その後、各高度でのoptical thicknessを計算し、reference optical depthに合わせる
# 波長が1.8から2.2.umの間で、各高度に感度のあるダストのプロファイルを作成する

# ------------------------------------------

# %%
import numpy as np
import matplotlib.pyplot as plt

# ------- set a parameter -------
ref_dop = 2.0 # for 277 # reference optical depthを決める

# 増加させる高度ははここで設定
h_num = 10 # 刻み数
h_min = 10  # 最小高度 (km)
h_max = 70  # 最大高度 (km)

plot_on = True # プロットを行うかどうか
# -------------------------------

# -------- base information --------
# 高度の範囲 (0〜80km)
h = np.linspace(0, 80, 41)  # 高度 (km)
H = 10  # スケールハイト (km)

# ------- getting information about wavelength at OMEGA -------
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
    input_wav_cm = 1/input_wav * 1e4 # cm-1

    # ダストの消散係数情報を取得する
    Dust_exp = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/dust_2500-25000_mathieu.aero")
    wv_ref = Dust_exp[:,0]
    ext_ref = Dust_exp[:,1]

    # input_wav_cmの値に補間を行う
    ext_interp = np.interp(input_wav_cm, wv_ref, ext_ref)
    Dust_ext = ext_interp*1e-3 # μmをcmに変換
    # μm → mに直したのちにkmに変換
    # もともとのunitはμm2, 数密度はcm-3

    # detached layer likeのプロファイルを作成する
    h_peak = np.arange(h_min, h_max, h_num)  # 増加させる高度 (km)
    sigma = 3     # 増加の広がり (km)
    # 各高度で増加の強さを揃える
    #amplitude = 0.2  # 増加の強さ
    # 各高度でamplitudeを変化させる
    amplitude_per = 2.0 # 増加量

    if plot_on:
        plt.figure(figsize=(6, 4))
        plt.plot(np.exp(-h / H)*7, h, linestyle="dashed", label="Original Profile")
        plt.xlabel("Dust Density (arbitrary units)")
        plt.ylabel("Altitude (km)")
        np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/0_v1.hc", np.column_stack((h, np.exp(-h / H)*7)))
        
    """
    # 元のプロファイルを保存する: 0.hc
    dust_profile = np.exp(-h / H)*7
    save_data = np.column_stack((h, dust_profile))
    np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/0.hc", save_data)
    """

    # h_peak分のプロファイルを重ねてplotする
    for peak in h_peak:
        dust_profile = 0
        dust_profile = np.exp(-h / H)
        ind = np.where(h == peak)[0][0] # peakのindexを取得
        amplitude = amplitude_per * dust_profile[ind] # 増加の強さ
        # 増加の強さを設定する
        dust_profile += amplitude * np.exp(-((h - peak) ** 2) / (2 * sigma ** 2))

        if plot_on:
            plt.plot(dust_profile*7, h, label="Modified Dust Profile")

        # 積分をして各層を足し合わせていく
        dust_profile_fix = dust_profile*7
        optical_thickness = 0

        for k in range(len(h)):
            optical_thickness += Dust_ext * dust_profile_fix[k]
        #print("Dust Optical Thickness:", optical_thickness)

        dust_profile_fin = dust_profile_fix * (ref_dop / optical_thickness)
        optical_thickness = 0

        for j in range(len(h)):
            optical_thickness += Dust_ext * dust_profile_fin[j]
        #print("Dust Optical Thickness:", optical_thickness)

        save_data = np.column_stack((h, dust_profile_fin))
        np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/" + str(peak) +"_v1.hc", save_data)


if plot_on:
    plt.legend()
    plt.title("Dust Vertical Profile with Enhancement")
    plt.show()
# %%
# ---------------------------------------------------------
# 波長が2.77 μmのときのprofile input fileを作成する

import numpy as np
import matplotlib.pyplot as plt

# ------- set a parameter -------
ref_dop = 2.0 # for 277 # reference optical depthを決める

# 増加させる高度ははここで設定
h_num = 10 # 刻み数
h_min = 0  # 最小高度 (km)
h_max = 70  # 最大高度 (km)

plot_on = True # プロットを行うかどうか
# -------------------------------

# -------- base information --------
# 高度の範囲 (0〜80km)
h = np.linspace(0, 80, 41)  # 高度 (km)
H = 11  # スケールハイト (km)

# ------- getting information about wavelength at OMEGA -------
input_wav = 2.77 # 波長 (μm)
file_name = str(int(input_wav*100))
input_wav_cm = 1/input_wav * 1e4 # cm-1

# ダストの消散係数情報を取得する
Dust_exp = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/dust_2500-25000_mathieu.aero")
wv_ref = Dust_exp[:,0]
ext_ref = Dust_exp[:,1]

# input_wav_cmの値に補間を行う
ext_interp = np.interp(input_wav_cm, wv_ref, ext_ref)
Dust_ext = ext_interp*1e-3 # μmをcmに変換
# μm → mに直したのちにkmに変換
# もともとのunitはμm2, 数密度はcm-3

# detached layer likeのプロファイルを作成する
h_peak = np.arange(h_min, h_max, h_num)  # 増加させる高度 (km)
sigma = 3     # 増加の広がり (km)
# 各高度で増加の強さを揃える
#amplitude = 0.2  # 増加の強さ

# 各高度でamplitudeを変化させる
amplitude_per = 2.0 # 増加量

if plot_on:
    plt.figure(figsize=(6, 4))
    plt.plot(np.exp(-h / H)*7, h, linestyle="dashed", label="Original Profile")
    plt.xlabel("Dust Density (arbitrary units)")
    plt.ylabel("Altitude (km)")
    
"""
# 元のプロファイルを保存する: 0.hc
dust_profile = np.exp(-h / H)*7
save_data = np.column_stack((h, dust_profile))
np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/0.hc", save_data)
"""

# h_peak分のプロファイルを重ねてplotする
for peak in h_peak:
    dust_profile = 0
    dust_profile = np.exp(-h / H)
    ind = np.where(h == peak)[0][0] # peakのindexを取得
    amplitude = amplitude_per * dust_profile[ind] # 増加の強さ
    dust_profile += amplitude * np.exp(-((h - peak) ** 2) / (2 * sigma ** 2))

    if plot_on:
        plt.plot(dust_profile*7, h, label="Modified Dust Profile")

    # 積分をして各層を足し合わせていく
    dust_profile_fix = dust_profile*7
    optical_thickness = 0

    for k in range(len(h)):
        optical_thickness += Dust_ext * dust_profile_fix[k]
    print("Dust Optical Thickness:", optical_thickness)

    dust_profile_fin = dust_profile_fix * (ref_dop / optical_thickness)
    optical_thickness = 0

    for j in range(len(h)):
        optical_thickness += Dust_ext * dust_profile_fin[j]
    print("Dust Optical Thickness:", optical_thickness)

    save_data = np.column_stack((h, dust_profile_fin))
    np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/" + str(peak) +"_v2.hc", save_data)

if plot_on:
    plt.legend()
    plt.title("Dust Vertical Profile with Enhancement")
    plt.show()

# %%
# ---------------------------------------------------------
# DISORTを使って計算して結果をplotする
# output fileをplotする
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# ------- set a parameter -------
# 増加させる高度ははここで設定
h_num = 10 # 刻み数
h_min = 10  # 最小高度 (km)
h_max = 70  # 最大高度 (km)

h_peak = np.arange(h_min, h_max, h_num)
# ---------------------------------

fig, ax = plt.subplots()
ax.set_xlabel("Wavelength [μm]")
ax.set_ylabel("Radiance [W/m^2/sr]")

fig, ax2 = plt.subplots()
ax2.set_xlabel("enhanced altitude [km]")
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

for loop in range(len(h_peak)):
    for i in range(len(wav_ind_after)):
        input_wav = wav_ind[i]
        file_name = str(int(wav_ind_after[i]))
        Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/detached-layer/"+file_name+"/"+ str(h_peak[loop]) +".dat")
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

    ax.plot(wvl_all_um, rad_all_um, label="altitude" + str(h_peak[loop]))
    #ax.scatter(wvl_all_um, rad_all_um, s=7)
    ax.scatter(POLY_x, POLY_y, s=30)
    ax.legend()

    ax3.plot(wvl_all_um, rad_calc, label="altitude" + str(h_peak[loop]))
    ax3.scatter(wvl_all_um, rad_calc, s=7)
    ax3.legend()

    data_save = np.stack((wvl_all_um, rad_calc), axis=-1)
    #np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/2umband_" + str(input_ls[loop]) +".txt", data_save)

    rad_total = np.sum(rad_calc[8:21])
    #ax2.scatter(input_ls[loop], rad_total,label="Ls" + str(input_ls[loop]))
    ax2.scatter(h_peak[loop], rad_calc[15], label="enhancement altitude" + str(h_peak[loop]))
    #ax2.scatter(input_ls[loop], rad_total,label="amount of CO2 absorption", color="black")
    #ax2.scatter(input_ls[loop], rad_calc[15], label = '2.01 um', color="red")


    # rad_calc[6]とrad_calc[17]に直線を引く
    ax3.axvline(x=wvl_all_um[8], color="black", linestyle="--")
    ax3.axvline(x=wvl_all_um[21], color="black", linestyle="--")
    ax2.legend()
# %%
# ---------------------------------------------------------
# DISORTを使って計算して結果をplotする
# output fileをplotする
import numpy as np
import matplotlib.pyplot as plt

# ------- set a parameter -------
# 増加させる高度ははここで設定
h_num = 10 # 刻み数
h_min = 0  # 最小高度 (km)
h_max = 60  # 最大高度 (km)

h_peak = np.arange(h_min, h_max, h_num)

OMEGA_wav = 2.77 # 波長 (μm)
file_name = str(int(OMEGA_wav*100))

# ----------- plot --------------------
fig, ax = plt.subplots()
ax.set_xlabel("enhanced altitude [km]")
ax.set_ylabel("Radiance [W/m^2/sr]")

# peakごとのradをplotする
for loop in range(len(h_peak)):
    Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/detached-layer/"+file_name+"/"+ str(h_peak[loop]) +"_v1.dat")
    rad = Ls15_profile[1]
    wvl_rad = Ls15_profile[0]

    # 波長と放射輝度をcmからumに変換する
    wvl_um = 1/wvl_rad * 1e4
    rad_um = (rad/((1/wvl_rad)**2))* 1e-7
    print('altitude:', h_peak[loop], 'rad:', rad_um)

    ax.scatter(h_peak[loop], rad_um, label="altitude" + str(h_peak[loop]))
    ax.legend(loc="lower right", fontsize=8)
# %%
