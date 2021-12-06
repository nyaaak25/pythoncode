# %%
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:32:00 2021

@author: Suzuki

ver.2
Created on Wed Sep 1 11:21:00 2021

@author Kazama
S
"""

"""
吸収係数と高度のプロット
"""

#hapi.pyファイルを読み込んで、HITRAN line-by-lineからパラメータを持ってくる作業

from hapi import *
import numpy as np
import plotly

# スペクトルの範囲
spectrum_begin = 4973.86 # cm^-1
spectrum_end = 4973.87
#4973 --> 大体2um

# HITRAN のパラメータ
name = 'CO2' 
moleculeID = 2
isotopologueID = 1

db_begin('data')
fetch(name, moleculeID, isotopologueID, spectrum_begin, spectrum_end)

# 取得したテーブルから、必要な情報を取得
vij,Sij,E,gammaair,gammaself,nair,δair = getColumns(name, ['nu', 'sw','elower','gamma_air','gamma_self','n_air','delta_air']) 
print('nu=',vij,'Sij=',Sij,'γself=',gammaself,'γair=',gammaair,'E"=',E,'nair=',nair,'δair=',δair)

"ここからスペクトル計算"
#インポート
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

#file open＆定数定義
k = 1.380649E-16 #erg K-1
c = 2.9979E+10 #cm s-1
h = 6.62607015E-27 #erg s
c2 = (h*c)/k #cm K (hc / k)
Na = 6.02214129E+23
M = 43.98983 #g molcules-1 CO2

#温度
T_txt = np.loadtxt('Temp_pres_Kazama.dat')
T=T_txt[:,1]
Tref = 296 #K

#圧力
P_txt = np.loadtxt('Temp_pres_Kazama.dat')
P=(P_txt[:,2])*10 #Pa⇒Barye

#Q(T)
Q_CO2=pd.read_csv(filepath_or_buffer="CO2_Q.csv", encoding="cp932", sep=",")
QT=Q_CO2['Q']
Qref=286.09

#Number Density
mixCO2=0.95
R=8.31E+7  #(g*cm^2*s^2)/(mol*K)
Pself=mixCO2*P
#nd=((Pself*Na)/(R*T))
nd = P / (k * T)

#佐藤さんスペクトル
Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum.csv", encoding="cp932", sep=",")
Satoτ=Sato['光学的厚さ']
SatoAA=Sato['透過率T=exp(-tau)']
#print(Satoτ)

#lnはいらない
#波数幅
ν = np.zeros(2001)
for i in range(2001):
    ν[i] = 4973.0000+0.001*i

#カラム計算量、鉛直積分量
#CLM = np.zeros(1)
#for i in range(len(nd)):
#    CLM += nd[i]*2E+5 #molecules cm-3
#DU = CLM/2.69E+16 #DU
#DU1 = DU.astype(np.float16)
#DU2 = DU1.astype(np.str)
#print(DU2)

#(1)ドップラー幅νD(T)
νD = (vij/c)*((2*k*T*Na)/M)**(1/2) #cm-1
print('ドップラー幅',νD)

#(2)ローレンツ幅νL(p,T) 
mixCO2 = 0.9532
Pself = mixCO2*P #分圧
Pref = 1013250  #分圧 /atm
νL = ((Tref/T)**nair)*(gammaair*(P-Pself)/Pref +gammaself*Pself/Pref) #cm-1
#################################################
#HITRANのgammaairとgammaselfは[cm-1 atm-1]で与えられているので
#(gammaair*(P-self)の単位は[cm-1 atm-1] * [hPa]
#となります。1 atmは1013.25 hPaなので102行目の右辺を1013.25で割らないといけませんね。
#################################################
print('ローレンツ幅',νL)

#nyo=np.stack([νD,νL],1)
#np.savetxt('doppler_lorentzian.dat',nyo)

#################################################
# Voigt functionの部分(116-134行目)は私と作り方が違うので見ていません
#################################################
#(3)x
x = np.zeros((len(νD), len(ν)))
vijs = np.zeros(len(νD))
for i in range(len(νD)):
    vijs[i] = vij + ((δair*P[i])/Pref)
    for j in range(len(ν)):
        x[i,j] = (ν[j]-vijs[i])/νD[i]
print(x)
#(4)y 
y = νL/νD

print(y)
# %%
#(5)Voigt function f(ν,p,T)
iy1, err1 = (np.zeros((len(νD), len(ν))), np.zeros((len(νD), len(ν))))
la = np.zeros((len(νD), len(ν)))
f = np.zeros((len(νD), len(ν)))
for i in range(len(νD)):
    for j in range(len(ν)):
        la = lambda t: ((np.e**(-t**2))/((x[i,j]-t)**2+y[i]**2))
        iy1[i,j], err1[i,j] = integrate.quad(la, -np.inf, np.inf)
        f[i,j] = (y[i]/np.pi)*iy1[i,j]/νD[i]/np.sqrt(np.pi) #cm
#(6)吸収線強度S(T)
#################################################
# Q(Tref)/Q(T)が入っていません　温度によって1からずれます
#################################################
m = 1
S = Sij*(Qref/QT)*((np.exp((-c2*E)/T))/(np.exp((-c2*E)/Tref)))*((1.0-np.exp((-c2*vij)/T))/(1-np.exp((-c2*vij)/Tref))) #cm-1/(molecule-1 cm2)
print("S",S)

#(7)吸収係数σ(z, ν)
σ = np.zeros((len(νD), len(ν)))
for i in range(len(νD)):
    for j in range(len(ν)):
        σ[i,j] = S[i]*f[i,j]

# %%

#x1 = ν
#y1 = AAA

fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
#for i in range(10):
#    ax.scatter(σ[:,i], np.arange(0,62,2), color='red')
ax.scatter(σ[:,100], np.arange(0,62,2), color='red')
ax.scatter(σ[:,300], np.arange(0,62,2), color='blue')
ax.scatter(σ[:,500], np.arange(0,62,2), color='green')
ax.scatter(σ[:,700], np.arange(0,62,2), color='yellow')
ax.scatter(σ[:,1000], np.arange(0,62,2), color='purple')
#ax.set_xlim(4973,4975) 
ax.set_xlabel('absorption [$cm^{-1}$]', fontsize=14)
ax.set_ylabel('altitude (km)', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)


#凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='upper right', fontsize=8)
plt.show()
# %%
