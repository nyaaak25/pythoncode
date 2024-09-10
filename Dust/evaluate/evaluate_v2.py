# %%
# ダストサイズを変えたことによる影響を確認する
# ダストの粒子サイズを計算させる

# MER SITEにおけるyannと2.77 μmの比較
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt

# read the result
# a=１から3のときの放射輝度値
dust_a1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_a1_rad.dat')
rad_a1 = dust_a1[1]
dust_a2 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_a2_rad.dat')
rad_a2 = dust_a2[1]
dust_a3 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_a3_rad.dat')
rad_a3 = dust_a3[1]

rad_array = [rad_a1, rad_a2, rad_a3]
number_array = [1, 2, 3]

plt.scatter(number_array, rad_array)

# %%
# Reffの計算を行う(a=1,2,3のとき)
a1_data = readsav("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/dust_imformation_a1.sav")
a1_radius = a1_data["r"]
a1_Nr = a1_data["N"]

a2_data = readsav("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/dust_imformation_a2.sav")
a2_radius = a2_data["r"]
a2_Nr = a2_data["N"]

a3_data = readsav("/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/dust_imformation_a3.sav")
a3_radius = a3_data["r"]
a3_Nr = a3_data["N"]

# 積分を行う
# Armin+09も(9)式を用いる
# G = ∫ π * r^2 * n(r) * dr
# Reff = 1/G * ∫ π * r^3 * n(r) * dr
# Veff = 1/(G * Reff^2) * ∫ (r-reff)^2 * π * r^2 * n(r) * dr

G_a1 = np.sum(np.pi * a1_radius**2 * a1_Nr)
a1_reff = np.sum(np.pi * a1_radius**3 * a1_Nr) / G_a1
a1_veff = np.sum((a1_radius - a1_reff)**2 * np.pi * a1_radius**2 * a1_Nr) / (G_a1 * a1_reff**2)

G_a2 = np.sum(np.pi * a2_radius**2 * a2_Nr)
a2_reff = np.sum(np.pi * a2_radius**3 * a2_Nr) / G_a2
a2_veff = np.sum((a2_radius - a2_reff)**2 * np.pi * a2_radius**2 * a2_Nr) / (G_a2 * a2_reff**2)

G_a3 = np.sum(np.pi * a3_radius**2 * a3_Nr)
a3_reff = np.sum(np.pi * a3_radius**3 * a3_Nr) / G_a3
a3_veff = np.sum((a3_radius - a3_reff)**2 * np.pi * a3_radius**2 * a3_Nr) / (G_a3 * a3_reff**2)

print('Reff, ', a1_reff, a2_reff, a3_reff)
print('Veff, ', a1_veff, a2_veff, a3_veff)
# %%
# read the result
# SSAを変化させたときの放射輝度値
dust_a1 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_mat_rad.dat')
rad_a1 = dust_a1[1]
dust_a2 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_mat1_rad.dat')
rad_a2 = dust_a2[1]
dust_a3 = np.loadtxt('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/imf-size/ORG_mat2_rad.dat')
rad_a3 = dust_a3[1]

rad_array = [rad_a1, rad_a2, rad_a3]
number_array = [1, 2, 3]

plt.scatter(number_array, rad_array)
# %%
