# -----------------------------------------------------------
# Open research用の.txtファイルのヘッダーを作成するプログラム
# 2025/02/03 Mon 10:22:00
# Created by Akira Kazama
# -----------------------------------------------------------

# %%
# Figure 1-1
# path
path_fig1 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure1/"

# 書き込むヘッダーを定義
header = "Orbit number, Shift variation\n"

with open(path_fig1 + "shift_variation.txt", "r", encoding="utf-8") as f:
    data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig1 + "shift_variation.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 1-2
# 書き込むヘッダーを定義
header = "wavelength, MY27:ORB0518_3, MY28:ORB3198_5\n"

with open(path_fig1 + "spectrum_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig1 + "spectrum_data.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 2-1
# path
path_fig2 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure2/"

# 書き込むヘッダーを定義
header = "wavelength, Radiance\n"

# dust = 0.0
with open(path_fig2 + "dust-00.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig2 + "dust-00.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust = 0.5
with open(path_fig2 + "dust-05.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig2 + "dust-05.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust = 1.0
with open(path_fig2 + "dust-10.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig2 + "dust-10.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust = 1.5
with open(path_fig2 + "dust-15.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig2 + "dust-15.txt", "w", encoding="utf-8") as f:
    f.write(header + data)
# %%
# Figure 3
# path
path_fig3 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure3/"

# 書き込むヘッダーを定義
header = "Weighting Function, altitude [km]\n"

# clear-sky_condition
with open(path_fig3 + "clear-sky_condition.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig3 + "clear-sky_condition.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust-storm_condition
with open(path_fig3 + "dust-storm_condition.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig3 + "dust-storm_condition.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# strong-dust_condition
with open(path_fig3 + "strong-dust_condition.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig3 + "strong-dust_condition.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 4-1
# path
path_fig4 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure4/"

# 書き込むヘッダーを定義
header = "Longitude[deg], Latitude[deg], Dust opacity\n"

# retrieval_dust
with open(path_fig4 + "retrieval_dust.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig4 + "retrieval_dust.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 4-2
# 書き込むヘッダーを定義
header = "Longitude[deg], Latitude[deg], Radiance\n"

# raw_radiance
with open(path_fig4 + "raw_radiance.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig4 + "raw_radiance.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 5-1
# path
path_fig5 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure5/"

# 書き込むヘッダーを定義
header = "Temperature deviation [K], Dust optical depth deviation\n"

# data_t1.txtからdata_t5.txtまで全てに適応
for i in range(1, 6):
    with open(path_fig5 + "data_t" + str(i) + ".txt", "r", encoding="utf-8") as f:
          data = f.read()

    # ヘッダーを追加して新しい内容を書き込む
    with open(path_fig5 + "data_t" + str(i) + ".txt", "w", encoding="utf-8") as f:
        f.write(header + data)

# %%
# Figure 5-2
# 書き込むヘッダーを定義
header = "Atmospheric pressure deviation [Pa], Dust optical depth deviation\n"

# data_p1.txtからdata_p5.txtまで全てに適応
for i in range(1, 6):
    with open(path_fig5 + "data_p" + str(i) + ".txt", "r", encoding="utf-8") as f:
          data = f.read()

    # ヘッダーを追加して新しい内容を書き込む
    with open(path_fig5 + "data_p" + str(i) + ".txt", "w", encoding="utf-8") as f:
        f.write(header + data)

# %%
# Figure 5-3
# 書き込むヘッダーを定義
header = "Instrument noise deviation[DN], Dust optical depth deviation\n"

# data_s1.txtからdata_s5.txtまで全てに適応
for i in range(1, 6):
    with open(path_fig5 + "data_s" + str(i) + ".txt", "r", encoding="utf-8") as f:
          data = f.read()

    # ヘッダーを追加して新しい内容を書き込む
    with open(path_fig5 + "data_s" + str(i) + ".txt", "w", encoding="utf-8") as f:
        f.write(header + data)
# %%
# Figure 6-1
# path
path_fig6 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure6/"

# 書き込むヘッダーを定義
header = "Altitude [km], Dust number density [cm^-3]\n"

# dust_nd_ls15
with open(path_fig6 + "dust_nd_ls15.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig6 + "dust_nd_ls15.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust_nd_ls165
with open(path_fig6 + "dust_nd_ls165.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig6 + "dust_nd_ls165.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust_nd_ls225
with open(path_fig6 + "dust_nd_ls225.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig6 + "dust_nd_ls225.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# dust_nd_ls315
with open(path_fig6 + "dust_nd_ls315.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig6 + "dust_nd_ls315.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 6-2
# 書き込むヘッダーを定義
header = "Solar longitude [deg], Radiance\n"

# radiance.txt
with open(path_fig6 + "radiance.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig6 + "radiance.txt", "w", encoding="utf-8") as f:
    f.write(header + data)
# %%
# Figure 7
# path
path_fig7 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure7/"

# 書き込むヘッダーを定義
header = "Vincendon retrievals, Our retrievals, y=ax+b\n"

# data_comparison
with open(path_fig7 + "data_comparison.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig7 + "data_comparison.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 9
# path
path_fig9 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure9/"

# 書き込むヘッダーを定義
header = "Longitude[deg], Latitude[deg], Dust optical depth\n"

# directory内の全てのファイルに適応
import os
for file in os.listdir(path_fig9):
    if file.endswith(".txt"):
        with open(path_fig9 + file, "r", encoding="utf-8") as f:
            data = f.read()

        # ヘッダーを追加して新しい内容を書き込む
        with open(path_fig9 + file, "w", encoding="utf-8") as f:
            f.write(header + data)

# %%
# Figure 10
# path
path_fig10 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure10/"

# 書き込むヘッダーを定義
header = "Longitude[deg], Latitude[deg], Dust optical depth\n"

# directory内の全てのファイルに適応
for file in os.listdir(path_fig10):
    if file.endswith(".txt"):
        with open(path_fig10 + file, "r", encoding="utf-8") as f:
            data = f.read()

        # ヘッダーを追加して新しい内容を書き込む
        with open(path_fig10 + file, "w", encoding="utf-8") as f:
            f.write(header + data)

# %%
# Figure 11
# path
path_fig11 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure11/"

# 書き込むヘッダーを定義
header = "Dust optical depth, Latitude [deg], Solar longitude [deg]\n"

# MY27_detect_lds
with open(path_fig11 + "MY27_detect_lds.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig11 + "MY27_detect_lds.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# MY28_detect_lds
with open(path_fig11 + "MY28_detect_lds.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig11 + "MY28_detect_lds.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# MY29_detect_lds
with open(path_fig11 + "MY29_detect_lds.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig11 + "MY29_detect_lds.txt", "w", encoding="utf-8") as f:
    f.write(header + data)
# %%
# Figure 12
# path
path_fig12 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure12/"

# 書き込むヘッダーを定義
header = "Local time [h], Number of all observations, Number of detections\n"

# ls0-180_probability
with open(path_fig12 + "ls0-180_probability.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig12 + "ls0-180_probability.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# ls180-360_probability
with open(path_fig12 + "ls180-360_probability.txt", "r", encoding="utf-8") as f:
          data = f.read()

# ヘッダーを追加して新しい内容を書き込む
with open(path_fig12 + "ls180-360_probability.txt", "w", encoding="utf-8") as f:
    f.write(header + data)

# %%
# Figure 13
# path
path_fig13 = "/Users/nyonn/Desktop/論文/retrieval dust/Open-research/Figure13/"

# 書き込むヘッダーを定義
header = "Longitude[deg], Latitude[deg], Dust optical depth\n"

# directory内の全てのファイルに適応
for file in os.listdir(path_fig13):
    if file.endswith(".txt"):
        with open(path_fig13 + file, "r", encoding="utf-8") as f:
            data = f.read()

        # ヘッダーを追加して新しい内容を書き込む
        with open(path_fig13 + file, "w", encoding="utf-8") as f:
            f.write(header + data)

# %%
