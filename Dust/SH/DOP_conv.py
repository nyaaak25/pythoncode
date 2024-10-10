# %%
# --------------------------------------------------
# 2024.04.12 created by AKira Kazama
# ダストの光学的厚さを計算する関数
# 与えられた数密度からダストの光学的厚さを計算します。
# --------------------------------------------------
# %%
import numpy as np
import matplotlib.pyplot as plt

# ダウトの消散係数情報を取得する
Dust_exp = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/dust_2500_8500.aero")
# 波数3601のデータを取得
#Dust_ext = Dust_exp[1101,1] # 2.77 μm
#Dust_ext = Dust_exp[2472,1] # 2.01 μm
#Dust_ext = Dust_exp[2961,1] # 1.82 μm
Dust_ext = Dust_exp[2063,1] # 2.19 μm

# μmをcmに変換
Dust_ext = Dust_ext * 1e-3
# μm → mに直したのちにkmに変換
# もともとのunitはμm2, 数密度はcm-3

# 数密度データを読み込む
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/1d.hc")
altitude = HD_base[:,0]

# 0から80 kmまで5km刻みでデータを作成
# altitude = np.linspace(0, 80, 17)

# 1d.hcのDust density profile
# dust_number_density_profile = HD_base[:,1]

# 数密度データを読み込む
ground_nd = 8
dust_number_density_profile = np.zeros(len(altitude))
for i in range(len(altitude)):
    dust_number_density_profile[i] = ground_nd * np.exp(-altitude[i]/10) # scale height = 10 km

# 積分をして各層を足し合わせていく
optical_thickness = 0
for i in range(len(dust_number_density_profile)):
    optical_thickness += Dust_ext * dust_number_density_profile[i]

print("Dust Optical Thickness:", optical_thickness)
plt.scatter(dust_number_density_profile, altitude, label="Dust Number Density", color="black", linestyle="solid")

# 2d.hcは1d.hcの約10倍のDust opacityで作成
# 3d.hcは1d.hcの約1/10倍のDust opacityで作成
save_data = np.column_stack((altitude, dust_number_density_profile))
np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/1dd_after.hc", save_data)

# %%