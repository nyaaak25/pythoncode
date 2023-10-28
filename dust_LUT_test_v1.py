"""

ARSによって生成されたfileからLUT gridを決めるためのテストを行う
Created on Tue Aug 1 08:48:00 2023
@author: A.Kazama kazama@pparc.gp.tohoku.ac.jp

ver. 1.0: 角度依存性を確認するためのプログラム

ver. 2.0: ダスト依存性について確認する

"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin

# .sav fileを読み込む
# OMEGAの観測データと生成された放射輝度スペクトルを比較する際に使用する
data_dir = pjoin(dirname(sio.__file__), "tests", "data")
sav_fname = "dust_ret/ORB0923_3.sav"
sav_data = readsav(sav_fname)
wvl = sav_data["wvl"]

# 波長範囲はOMEGAのL channelを使用する
# その中でも2.66から2.9 μmに着目
wvl = wvl[128:255]  # L channel
CO2 = np.where((wvl > 2.66) & (wvl < 2.9))
wvl = wvl[CO2]

jdat = sav_data["jdat"]
jdat = jdat[:, 128:255, :]

# 太陽輝度スペクトルを読み込む
specmars = np.loadtxt("/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat")
dmars = sav_data["dmars"]
specmars = specmars / dmars / dmars
specmars = specmars[128:255]
specmars = specmars[CO2]

# 放射輝度だけで議論する場合
# flux = jdat[71, CO2, 1]

# I/Fで議論する場合
flux = jdat[71, CO2, 1] / specmars

# ここで2.77 μmにおける角度依存性を確かめるための配列を作成する
AA = np.zeros(10)

# 各スペクトルのgridを決める
Dust_list = [
    "Dust=0.00",
    "Dust=0.01",
    "Dust=0.02",
    "Dust=0.03",
    "Dust=0.04",
    "Dust=0.05",
    "Dust=0.06",
    "Dust=0.07",
    "Dust=0.08",
    "Dust=0.09",
    "Dust=0.1",
]
Pa_list = [
    "Pa=100",
    "Pa=120",
    "Pa=140",
    "Pa=160",
    "Pa=180",
    "Pa=200",
    "Pa=220",
    "Pa=240",
    "Pa=260",
    "Pa=280",
    "Pa=300",
    "Pa=320",
    "Pa=340",
    "Pa=360",
    "Pa=380",
    "Pa=400",
    "Pa=420",
    "Pa=440",
    "Pa=460",
    "Pa=480",
    "Pa=500",
    "Pa=520",
    "Pa=540",
    "Pa=560",
    "Pa=580",
    "Pa=600",
    "Pa=620",
    "Pa=640",
    "Pa=660",
    "Pa=680",
    "Pa=700",
]
CO2_list = [
    "mix=0.9532",
    "mix=0.9432",
    "mix=0.9332",
    "mix=0.9232",
    "mix=0.9132",
    "mix=0.9032",
    "mix=0.8932",
    "mix=0.8832",
    "mix=0.8732",
    "mix=0.8632",
    "mix=0.8532",
    "mix=0.8432",
]
TA_list = ["TA=135", "TA=160", "TA=213", "TA=260", "TA=285"]
TB_list = ["TB=80", "TB=146", "TB=200"]
Albedo_list = [
    "Albedo=0.05",
    "Albedo=0.15",
    "Albedo=0.25",
    "Albedo=0.35",
    "Albedo=0.45",
    "Albedo=0.55",
    "Albedo=0.65",
    "Albedo=0.75",
    "Albedo=0.85",
    "Albedo=0.95",
]


"""
# v0: それぞれのパラメータだけを降ってテストをした場合のgrid
SZA_list = ['SZA=0', 'SZA=15', 'SZA=30', 'SZA=45', 'SZA=60',
            'SZA=75']
EA_list = ['EA=0', 'EA=15', 'EA=30', 'EA=45', 'EA=60',
           'EA=75']
PA_list = ['PA=0', 'PA=45', 'PA=90', 'PA=135', 'PA=180',
           'PA=225']
"""

# v1: SZA, EA, PAをみんな降ってどのようになるかを確認する
SZA_list = ["SZA=0", "SZA=15", "SZA=30", "SZA=45"]
EA_list = ["EA=0", "EA=15", "EA=30", "EA=45"]
PA_list = ["PA=0", "PA=45", "PA=90", "PA=135"]

SZA = [0, 15, 30, 45]
# SZA test用
# SZA = ['SZA=00.0', 'SZA=0', 'SZA=0.1', 'SZA=1.0']
EA = [0, 5, 15, 30]
PA = [0, 45, 90, 135]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title="2.7um CO2 absorption band")
ax.grid(c="lightgray", zorder=1)
ax.set_xlabel("Wavenumber [μm]", fontsize=10)

# Albedo = np.zeros(10)
CO2_mix = np.zeros(10)

for i in range(0, 10, 1):
    # Albedo[i] = 0.05 + 0.1 * i
    CO2_mix[i] = 0.9532 - 0.01 * i

    """

    各ファイルの計算詳細: old/output

    # _rad: 純粋な計算結果
    # _new_rad: 純粋な計算結果を確かめるためにもう一度計算をしてみた

    # _SZA1_rad: SZA=1, 15, 30, 45の計算結果
    # _SZA01_rad: SZA=0.1, 15, 30, 45の計算結果

    # _1-3_rad: 1.5 to 3.5 μmのガス抜き計算 (SZA=0,15,30,45)
    # _1-3_d0_rad: 1.5 to 3.5 μmのガス+エアロゾル抜き計算 (SZA=0,15,30,45)
    # → ARS_y[221] equl 2.77 μm

    # _gas0_rad: ガス抜き計算結果 (SZA=0,15,30,45)
    # _aero0_rad: Number_of_aerosol = 0, エアロゾル0のときの結果 (SZA=00.0,0,0.1,1)

    # _full_rad: 完璧なDISORT(3 2 0)で計算を行う (SZA=00.0,0,0.1,1)
    # _full2_rad: 完璧なDISORTで(3 2 0)で計算を行う (SZA=0,15,30,45)

    # _d0_rad: ダストを抜きで計算を行う (SZA=00.0,0,0.1,1)
    # d001_rad: ダストを0.01にして計算を行う (SZA=00.0,0,0.1,1)


    各ファイルの詳細: output/
    # _albedo_: アルベドを0.05から0.95まで変化
    # _CO2mix_: CO2混合比を0.95から0.85まで変化
    # _SZA_EA_PA: それぞれの角度を変化させて角度依存性を確認
    # _dust_: ダストを0から0.2まで変化

    """

    ARS = np.loadtxt(
        "/Users/nyonn/Desktop/pythoncode/output/loc1_CO2mix " + str(i) + "_320.dat"
    )
    ARS_x = ARS[0]
    ARS_wav = 1 / ARS_x
    ARS_x = (1 / ARS_x) * 10000
    ARS_y = ARS[1]
    ARS_y = (ARS_y / (ARS_wav * ARS_wav)) * 1e-7
    ARS_y = ARS_y / specmars[5]
    AA[i] = ARS_y

    ax.scatter(ARS_x, ARS_y, label=CO2_list[i], zorder=i, s=10)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc="lower left", fontsize=5)

# %%
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title="dependent of Albedo")
ax.grid(c="lightgray", zorder=1)
ax.set_xlabel("CO2 mixing ratio", fontsize=10)
ax.plot(CO2_mix[::-1], AA, lw=1)
