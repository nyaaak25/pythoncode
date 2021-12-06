# %%
# -*- coding: utf-8 -*-
"""
計算された光学的厚みから透過率を計算し、plotをする
"""

#import
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

#波数
ν_txt = np.loadtxt('new_4971-4976_kinji_2.txt')
ν=ν_txt[2000:4001,0]

#光学的厚み
τ_txt = np.loadtxt('new_4971-4976_kinji_2.txt')
τ=τ_txt[2000:4001,1]

#光学的厚み
tau_txt = np.loadtxt('new_4971-4976_voigt_test.txt')
tau =tau_txt[2000:4001,1]

#透過率
A = np.exp(-τ)

#佐藤さんスペクトル(encordingの　cp932/utf-8)
#Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum.csv", encoding="cp932", sep=",")
#Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum_sumtau.csv", encoding="utf-8", sep=",")
Sato1 = pd.read_csv(filepath_or_buffer="Satosan-kensyou.csv", encoding="cp932", sep=",")

#Sato_spectrum.csvのファイルopen、4973.86のシングルライン検証用
#Satoν=Sato['波数(cm-1)']
#Satosingleτ=Sato['光学的厚さ']
#SatosingleAA=Sato['透過率T=exp(-tau)']

#Sato_spectrum_sum.csvファイル openの際のもの
#Satoν=Sato['波数']
#Satoallτ=Sato['光学的厚さ(all)']
#SatoallAA=Sato['透過率 (all)']
#Sato21τ=Sato['光学的厚さ (only 21)']
#Sato21AA=Sato['透過率 (only 21)']

#Sato_spectrum_sum.csvファイル openの際のもの
Satov=Sato1['波数']
Satoτ_python=Sato1['光学的厚さpython']
Satoτ_for=Sato1['光学的厚さfor']
Sato_gosa=Sato1['誤差']


# %%
#---------------グラフ作成----------------------------------------------
#データ読み込み&定義
#DDD=((τ-Satoτ_for[0:2001])/Satoτ_for[0:2001])*100  #(風間ー佐藤さん)*100/佐藤さん
DDD = (τ-tau)*100/tau
#AAA = ((A-Sato21AA)/Sato21AA)*100 #(風間ー佐藤さん)*100/佐藤さん
#AAA = A-Sato21AA

x1 = ν
y1 = DDD

fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
ax.plot(x1, y1, color='red',label="Voigt-Kinji")
#ax.plot(x1, τ, color='b',label="Kinji")
#ax.plot(x1, tau, color='red',label="Voigt")
ax.set_xlim(4973,4975)
#ax.set_yscale('log') 
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
#ax.set_ylabel('error (%)', fontsize=14)
#ax.set_ylabel('Optical Depth', fontsize=14)
#ax.set_ylabel('Transmittance', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

#ax.plot(Satoν,Satoallτ,color='r')
#ax.plot(Satoν,Sato21τ,color='green',label="Sato-san")
#ax.plot(Satoν,SatoallAA,color='r')
#ax.plot(Satoν,Sato21AA,color='green',label="Sato-san")
#ax.plot(Satov,Satoτ_for,color='green',label="Sato-san")

#凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='upper right', fontsize=8)
plt.show()

# %%
