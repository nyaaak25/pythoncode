
# ダストの光学的厚さを計算する関数

# %%
import numpy as np

def dust_optical_thickness(dust_number_density_profile, particle_radius, wavelength, dz):
    # 定数
    pi = np.pi
    k_mie = 1e-6  # ミー散乱の係数（適切な値を使用してください）
    k_rayleigh = 0  # レイリー散乱の係数（適切な値を使用してください）
    
    # 光学的厚さの初期化
    optical_thickness = 0.0
    
    # 鉛直方向に積分して光学的厚さを計算
    for dust_number_density in dust_number_density_profile:
        # ミー散乱による光学的厚さの計算
        dust_cross_section_mie = pi * particle_radius**2  # ダスト粒子の断面積（単純化のため球状の粒子としています）
        scattering_cross_section_mie = k_mie * dust_cross_section_mie  # ミー散乱による散乱断面積
        
        # レイリー散乱による光学的厚さの計算
        scattering_cross_section_rayleigh = k_rayleigh / wavelength**4  # レイリー散乱による散乱断面積
        
        # 1ステップの光学的厚さを加算
        optical_thickness += (scattering_cross_section_mie + scattering_cross_section_rayleigh) * dust_number_density * dz
    
    return optical_thickness


# ダスト数密度の鉛直プロファイル（例としてリストで与える）
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/1d.hc")
dust_number_density_profile = HD_base[:,1]

# ダスト粒子の半径（単位はμm）
particle_radius = 2  # 例として0.1μmを設定します。適切な値を使用してください。

# 光の波長（単位はμm）
wavelength = 2.7  # 例として0.5μmを設定します。適切な値を使用してください。

# 鉛直方向のステップサイズ（dz）（単位は任意の長さ、例えばcmなど）
dz = 2.0  # 例として1.0を設定します。適切な値を使用してください。

# ダスト光学的厚さの計算
optical_thickness = dust_optical_thickness(dust_number_density_profile, particle_radius, wavelength, dz)

print("Dust Optical Thickness:", optical_thickness)
# %%
