# %%
# --------------------------------------------------
# 2024.04.12 created by AKira Kazama
# ダストの光学的厚さを計算する関数
# 与えられた数密度からダストの光学的厚さを計算します。

# 2024.10.15 updated by AKira Kazama
# ダストの光学的厚さを計算する関数
# 与えられた数密度からダストの光学的厚さを計算します。
# 波長を指定すると、補間をしてその波長での消散係数を取得します。
# --------------------------------------------------
# %%
import numpy as np
import matplotlib.pyplot as plt

# ------- set a parameter -------
# 波長をμmで指定する
input_wav = 2.77 # μm
ratio = 1
filename = "277_1d"  # 波長+何倍かを示している。1なら等倍、2なら10倍, 3なら100倍

# -------- base information --------
# 高度の範囲 (0〜80km)
h = np.linspace(0, 80, 41)  # 高度 (km)
H = 10  # スケールハイト (km)

dust_profie = np.exp(-h / H)*7
# ------- calculate  -------
# そこから波数を計算する
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

# 数密度データを読み込む
# reference data
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/input-hc/1d.hc")
altitude = HD_base[:,0]
ref_dust_number_density_profile = HD_base[:,1]

# 数密度データを読み込む
ground_nd = ref_dust_number_density_profile[0] * ratio
dust_number_density_profile = np.zeros(len(altitude))
for i in range(len(altitude)):
    dust_number_density_profile[i] = ground_nd * np.exp(-altitude[i]/10) # scale height = 10 km

# 積分をして各層を足し合わせていく
optical_thickness = 0
for i in range(len(dust_number_density_profile)):
    optical_thickness += Dust_ext * dust_number_density_profile[i]

print("Dust Optical Thickness:", optical_thickness)
plt.scatter(dust_number_density_profile, altitude, label="Dust Number Density", color="black", linestyle="solid")
plt.ylim(0,80)

save_data = np.column_stack((altitude, dust_number_density_profile))
#np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/input-hc/"+filename+".hc", save_data)

input_ls = [165]
file_name = "277"

Ls15_profile = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-dustprofile/hc/"+file_name+"/Tanguy_ls" + str(input_ls[0]) +"_v2.hc")
alt = Ls15_profile[:,0]
dust_number_density_cm = Ls15_profile[:,1]


# plot
fig, ax = plt.subplots()
#ax.scatter(dust_number_density_cm, alt, label="Tanguy profile", color="black")
ax.scatter(dust_profie,h, label="fixed altitude", color="black")
ax.scatter(dust_number_density_profile, altitude, label="input condition", color="red",s=8)
ax.set_xlabel("Dust Number Density [cm-3]")
ax.set_ylabel("Altitude [km]")
ax.legend()
# タイトルにinput_lsを表示
ax.set_title("input_ls = " + str(input_ls[0]))

# %%