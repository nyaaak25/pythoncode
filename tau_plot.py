# %%
# -*- coding: utf-8 -*-
"""
計算された光学的厚みから透過率を計算し、plotをする
"""

# import
import numpy as np
import matplotlib.pyplot as plt

# 波数
ν_txt = np.loadtxt('4445-5656_0.01step.txt')
ν = ν_txt[10000:111100, 0]

v_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
v = v_txt[:, 0]

# 光学的厚み
# τ_txt = np.loadtxt('new_4971-4976_kinji_2.txt')
τ_txt = np.loadtxt('4545-5556_0.01step_cutoff.txt')
# τ_txt = np.loadtxt('4545-5556_0.01step_new.txt')
τ = τ_txt[:, 1]  # * 7e24
# τ_txt = np.loadtxt('4545-5556_Test.txt')
# τ  = τ_txt[426:431, 1]

# 光学的厚み
tau_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
tau = tau_txt[:, 1]

tau_txt1 = np.loadtxt('4545-5556_0.01step_cutoff_80.txt')
tau2 = tau_txt1[:, 1]

tau_txt2 = np.loadtxt('4445-5656_0.01step.txt')
tau3 = tau_txt2[10000:111100, 1]

tau_txt4 = np.loadtxt('4545-5556_0.01step_cutoff_50.txt')
tau4 = tau_txt4[:, 1]
# tau = tau_txt[10000:111100, 1]

# 透過率
A = np.exp(-tau)

# 佐藤さんスペクトル(encordingの　cp932/utf-8)
# Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum.csv", encoding="cp932", sep=",")
# Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum_sumtau.csv", encoding="utf-8", sep=",")
# Sato1 = pd.read_csv(filepath_or_buffer="Satosan-kensyou.csv",encoding = "cp932", sep = ",")

# Sato_spectrum.csvのファイルopen、4973.86のシングルライン検証用
# Satoν=Sato['波数(cm-1)']
# Satosingleτ=Sato['光学的厚さ']
# SatosingleAA=Sato['透過率T=exp(-tau)']

# Sato_spectrum_sum.csvファイル openの際のもの
# Satoν=Sato['波数']
# Satoallτ=Sato['光学的厚さ(all)']
# SatoallAA=Sato['透過率 (all)']
# Sato21τ=Sato['光学的厚さ (only 21)']
# Sato21AA=Sato['透過率 (only 21)']

# Sato_spectrum_sum.csvファイル openの際のもの
# Satov = Sato1['波数']
# Satoτ_python = Sato1['光学的厚さpython']
# Satoτ_for = Sato1['光学的厚さfor']
# Sato_gosa = Sato1['誤差']


# %%
# ---------------グラフ作成----------------------------------------------
# データ読み込み&定義
# DDD=((τ-Satoτ_for[0:2001])/Satoτ_for[0:2001])*100  #(風間ー佐藤さん)*100/佐藤さん
# DDD = (τ-tau)*100/tau     #normalaizeされているよ
DDD = tau3-tau
CCC = tau3-tau2
BBB = tau3-τ
AAA = tau3-tau4
XXX = ((np.exp(-tau3)-np.exp(-τ))/np.exp(-tau3))*100
# AAA = ((A-Sato21AA)/Sato21AA)*100 #(風間ー佐藤さん)*100/佐藤さん
# AAA = A-Sato21AA

x1 = v
x2 = τ_txt[:, 0]
y1 = tau

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
# ax.plot(x1, τ/200000, color='blue', label="Cut-off")
ax.plot(x1, XXX, color='blue', label="cutoff 120")
# ax.plot(x2, BBB, color='green', label="cutoff 100")
# ax.plot(x1, CCC, color='red', label="cutoff 80")
# ax.plot(x1, AAA, color='orange', label="cutoff 50")
# ax.set_xlim(4958.05, 4958.15)
# ax.set_xlim(4681.3, 4681.35)
ax.set_xlim(4800, 5100)
# ax.set_ylim(1e-5, 1.8e-5)
ax.set_ylim(-0.1, 0.1)
# ax.set_yscale('log')
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
# ax.set_ylabel('error (%)', fontsize=14)
ax.set_ylabel('Defference[%]', fontsize=14)
# ax.set_ylabel('Transmittance', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

# ax.plot(Satoν,Satoallτ,color='r')
# ax.plot(Satoν,Sato21τ,color='green',label="Sato-san")
# ax.plot(Satoν,SatoallAA,color='r')
# ax.plot(Satoν,Sato21AA,color='green',label="Sato-san")
# ax.plot(Satov,Satoτ_for,color='green',label="Sato-san")

# 凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='upper right', fontsize=6)
plt.show()

# %%
