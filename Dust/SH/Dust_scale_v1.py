
# %%
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama
# Dust_scale_v1.py;; 計算で出てきた高度依存性を確認するためのプログラム

import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["0.35", "3.5", "0.035"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function")
ax.set_ylabel("Altitude [km]")
ax.set_ylim(10, 40)
ax.set_xlim(0, 0.1)

for loop in range(1, 3, 1):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        #new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    #normarize = new_opacity / np.max(new_opacity)
    ax.scatter(new_opacity, altitude, label=Dust_list[loop])
    #ax.scatter(normarize, altitude, label=Dust_list[loop],s=10)
    #ax.plot(normarize, altitude, label=Dust_list[loop])

ax.legend()
ax.grid()


# %%
import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["0.35", "3.5", "0.035"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function")
ax.set_ylabel("Altitude [km]")

for loop in range(0, 3, 1):
    # 2 μmの暫定的な結果を出力するためのプログラム
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_DD" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]

    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000
    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    # 1.8 μmの暫定的な結果を出力するためのプログラム
    ORG_base_con1 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/Before_ORG_DD" +str(loop)+ "_rad.dat")
    con1_wav = ORG_base_con1[0]
    ORG_wav1 = 1 / con1_wav
    con1_wav = (1 / con1_wav) * 10000

    ORG_rad_con1 = ORG_base_con1[1]
    ORG_rad_con1 = (ORG_rad_con1 / (ORG_wav1**2)) * 1e-7
    # baseの高度をここに入れる
    HD_base_con1 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd_before.hc")
    orginal_con1 = HD_base_con1[:,1]

    # 2.2 μmの暫定的な結果を出力するためのプログラム
    ORG_base_con2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/After_ORG_DD" +str(loop)+ "_rad.dat")
    con2_wav = ORG_base_con2[0]
    ORG_wav2 = 1 / con2_wav
    con2_wav = (1 / con2_wav) * 10000
    ORG_rad_con2 = ORG_base_con2[1]
    ORG_rad_con2 = (ORG_rad_con2 / (ORG_wav2**2)) * 1e-7

    # baseの高度をここに入れる
    HD_base_con2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd_after.hc")
    orginal_con2 = HD_base_con2[:,1]

    # 2 μmの結果を2点を使ってreferectanceを計算する
    x_wav = np.array([con1_wav, con2_wav])
    y_rad = np.array([ORG_rad_con1, ORG_rad_con2])
    a_cont,b_cont = np.polyfit(x_wav, y_rad, 1)
    cont = b_cont + a_cont * ORG_wave
    ref_cont = 1 - (ORG_rad/cont)

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        ref2_cont = 0
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad_1 = (rad / (ORG_wav**2)) * 1e-7

        HD_2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "_before/" + str(i) + "_rad.dat")
        rad2 = HD_2[1]
        rad_2 = (rad2 / (ORG_wav1**2)) * 1e-7

        HD_3 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "_after/" + str(i) + "_rad.dat")
        rad3 = HD_3[1]
        rad_3 = (rad3 / (ORG_wav2**2)) * 1e-7

        # referectanceを計算する
        #xxx = np.array([con1_wav, ORG_wave, con2_wav])
        #yyy = np.array([rad_2, rad_1, rad_3])

        x_alt = np.array([con1_wav, con2_wav])
        y_alt = np.array([rad_2, rad_3])

        a_alt, b_alt = np.polyfit(x_alt, y_alt, 1)
        cont2 = b_alt + a_alt * ORG_wave
        ref2_cont = 1 - (rad_1/cont2)

        new_opacity[i] = (ref2_cont - ref_cont) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.min(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3

    ax.plot(normarize, altitude, label=Dust_list[loop])

ax.legend()
ax.grid()


# %%
# ----------------------------- Figure 3.5  -----------------------------

import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["0.35", "3.5", "0.035"]
color_list = ["blue", "orange", "green"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function")
ax.set_ylabel("Altitude [km]")
#ax.set_ylim(0, 100)
#ax.set_xlim(-0.1, 1.0)

# 2.7 μmのプロット
for loop in range(0, 3, 2):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3
    ax.plot(normarize, altitude, label=Dust_list[loop], color=color_list[loop])

# 2.01 μmのプロット
for loop in range(0, 3, 2):
    # 2 μmの暫定的な結果を出力するためのプログラム
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_DD" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]

    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000
    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        ref2_cont = 0
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad_1 = (rad / (ORG_wav**2)) * 1e-7

        new_opacity[i] = (rad_1 - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3

    ax.plot(normarize, altitude,color=color_list[loop], linestyle="--")

ax.legend()
ax.grid()


# %%
# ----------------------------- Figure 3.5  -----------------------------

import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["0.35", "3.5", "0.035"]
color_list = ["black", "orange", "green"]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function", fontsize=16)
ax.set_ylabel("Altitude [km]", fontsize=16)
ax.set_ylim(0, 70)
ax.set_xlim(0, 1.0)

# 2.7 μmのプロット
for loop in range(0, 1, 1):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3
    #ax.plot(normarize, altitude, label=Dust_list[loop], color=color_list[loop])
    ax.plot(normarize, altitude, label="2.7 μm", color=color_list[loop])

# 2.01 μmのプロット
for loop in range(0, 1, 1):
    # 2 μmの暫定的な結果を出力するためのプログラム
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_DD" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]

    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000
    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    # 1.8 μmの暫定的な結果を出力するためのプログラム
    ORG_base_con1 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/Before_ORG_DD" +str(loop)+ "_rad.dat")
    con1_wav = ORG_base_con1[0]
    ORG_wav1 = 1 / con1_wav
    con1_wav = (1 / con1_wav) * 10000

    ORG_rad_con1 = ORG_base_con1[1]
    ORG_rad_con1 = (ORG_rad_con1 / (ORG_wav1**2)) * 1e-7
    # baseの高度をここに入れる
    HD_base_con1 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd_before.hc")
    orginal_con1 = HD_base_con1[:,1]

    # 2.2 μmの暫定的な結果を出力するためのプログラム
    ORG_base_con2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/After_ORG_DD" +str(loop)+ "_rad.dat")
    con2_wav = ORG_base_con2[0]
    ORG_wav2 = 1 / con2_wav
    con2_wav = (1 / con2_wav) * 10000
    ORG_rad_con2 = ORG_base_con2[1]
    ORG_rad_con2 = (ORG_rad_con2 / (ORG_wav2**2)) * 1e-7

    # baseの高度をここに入れる
    HD_base_con2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "dd_after.hc")
    orginal_con2 = HD_base_con2[:,1]

    # 2 μmの結果を2点を使ってreferectanceを計算する
    x_wav = np.array([con1_wav, con2_wav])
    y_rad = np.array([ORG_rad_con1, ORG_rad_con2])
    a_cont,b_cont = np.polyfit(x_wav, y_rad, 1)
    cont = b_cont + a_cont * ORG_wave
    ref_cont = 1 - (ORG_rad/cont)

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        ref2_cont = 0
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad_1 = (rad / (ORG_wav**2)) * 1e-7

        HD_2 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "_before/" + str(i) + "_rad.dat")
        rad2 = HD_2[1]
        rad_2 = (rad2 / (ORG_wav1**2)) * 1e-7

        HD_3 = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/DD" +str(loop)+ "_after/" + str(i) + "_rad.dat")
        rad3 = HD_3[1]
        rad_3 = (rad3 / (ORG_wav2**2)) * 1e-7

        # referectanceを計算する
        #xxx = np.array([con1_wav, ORG_wave, con2_wav])
        #yyy = np.array([rad_2, rad_1, rad_3])

        x_alt = np.array([con1_wav, con2_wav])
        y_alt = np.array([rad_2, rad_3])

        a_alt, b_alt = np.polyfit(x_alt, y_alt, 1)
        cont2 = b_alt + a_alt * ORG_wave
        ref2_cont = 1 - (rad_1/cont2)

        new_opacity[i] = (ref2_cont - ref_cont) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.min(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3

    ax.plot(normarize, altitude,color=color_list[loop], linestyle="--",label="2.01 μm")

ax.legend()
#ax.grid()

# %%
# -------------------- Figure 3.6 --------------------

import numpy as np
import matplotlib.pyplot as plt

Dust_list = ["τ=0.35", "τ=3.5", "τ=0.035"]
color_list = ["black", "orange", "green"]
linestyle = ["-", "--", "-."]

fig = plt.figure(dpi=800)
ax = fig.add_subplot(111)
ax.set_title("Dust Weighting Function", fontsize=16)
ax.set_ylabel("Altitude [km]", fontsize=16)
ax.set_ylim(0, 70)
ax.set_xlim(0, 1.05)

# 2.7 μmのプロット
for loop in range(0, 3, 1):
    ORG_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/ORG/ORG_D" +str(loop)+ "_rad.dat")
    ORG_wave = ORG_base[0]
    ORG_wav = 1 / ORG_wave
    ORG_wave = (1 / ORG_wave) * 10000

    ORG_rad = ORG_base[1]
    ORG_rad = (ORG_rad / (ORG_wav**2)) * 1e-7

    # Baseの高度をここに入れる
    HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/input_hc/" +str(loop+1)+ "d.hc")
    altitude = HD_base[:,0]
    orginal = HD_base[:,1]

    new_opacity = np.zeros(len(altitude))

    # Dust profileを変えたときのもの
    for i in range(0,60,1):
        HD = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/output/D" +str(loop)+ "/" + str(i) + "_rad.dat")
        rad = HD[1]
        rad = (rad / (ORG_wav**2)) * 1e-7
        new_opacity[i] = (rad - ORG_rad) * (orginal[i] /np.max(orginal))

    normarize = new_opacity / np.max(new_opacity)
    # normarizeをsmoothingする
    for i in range(1, len(normarize)-1, 1):
        normarize[i] = (normarize[i-1] + normarize[i] + normarize[i+1]) / 3
    ax.plot(normarize, altitude, label=Dust_list[loop], color='black', linestyle=linestyle[loop])


ax.legend()
#ax.grid()
# %%