# %%
# -*- coding: utf-8 -*-
"""
光学的厚みを計算させるプログラム

Created on Sun Apr 20 15:32:00 2021
@author: Suzuki

ver.2 マルチスペクトルをVoigt functionをベタに計算させて導出プログラムを作成
Created on Wed Sep 1 11:21:00 2021
@author：A.Kazama

ver.2.1：Voigt functonに多項式近似を導入
created on Sat Oct 16 21:00:00 2021
@author：A.Kazama

"""

# hapi.pyファイルを読み込んで、HITRAN line-by-lineからパラメータを持ってくる作業
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from hapi import *
import numpy as np
import plotly

# スペクトルの範囲：変更するところ
spectrum_begin = 4971  # cm^-1
spectrum_end = 4976

# 4973 --> 大体2um
# 1.8 umから2.2umまでは4545cm-1から5556cm-1

# HITRAN のパラメータ
name = 'CO2'
moleculeID = 2
isotopologueID = 1

# CO2同位体は一個のみ

db_begin('data')
fetch(name, moleculeID, isotopologueID, spectrum_begin, spectrum_end)

# 取得したテーブルから、必要な情報を取得
vij, Sij, E, gammaair, gammaself, nair, δair = getColumns(
    name, ['nu', 'sw', 'elower', 'gamma_air', 'gamma_self', 'n_air', 'delta_air'])
print('nu=', vij, 'Sij=', Sij, 'γself=', gammaself, 'γair=',
      gammaair, 'E"=', E, 'nair=', nair, 'δair=', δair)

# HITRANから引っ張ってきた吸収線データをテキストファイルに落とし込む
#savearray = np.array([vij,Sij,gammaself,gammaair,E,nair,δair])
# np.savetxt('savearray1.txt',savearray.T)


"ここからスペクトル計算"

# インポート

# CGS単位系
# file open＆定数定義
k = 1.380649E-16  # erg K-1
c = 2.9979E+10  # cm s-1
h = 6.62607015E-27  # erg s
c2 = (h*c)/k  # cm K (hc / k)
Na = 6.02214129E+23
#ln = np.log(2)
M = 43.98983  # g molcules-1 CO2

# 温度
T_txt = np.loadtxt('Temp_pres_Kazama.dat')
T = T_txt[:, 1]
Tref = 296  # K

# 圧力
P_txt = np.loadtxt('Temp_pres_Kazama.dat')
P = (P_txt[:, 2])*10  # Pa⇒Barye

# Q(T)　温度によって変更、ここのInputfileの計算は別プログラム(Qcaluculation.py)
Q_CO2 = pd.read_csv(filepath_or_buffer="CO2_Q.csv", encoding="cp932", sep=",")
QT = Q_CO2['Q']
Qref = 286.09

# Number Density
mixCO2 = 0.95
R = 8.31E+7  # (g*cm^2*s^2)/(mol*K)
Pself = mixCO2*P
# nd=((Pself*Na)/(R*T))
nd = P / (k * T)

# 検証用の佐藤さんスペクトルをImport
Sato = pd.read_csv(filepath_or_buffer="Sato_spectrum.csv",
                   encoding="cp932", sep=",")
Satoτ = Sato['光学的厚さ']
SatoAA = Sato['透過率T=exp(-tau)']
# print(Satoτ)

# 波数幅：計算する波数を決定　変更するパラメータ
ν = np.zeros(5001)
for i in range(5001):
    ν[i] = 4971.0000+(0.001*i)

# 1.8 cm-1から2.2m-1までは4545cm-1から5556cm-1

print('波数', ν)

# カラム計算量、鉛直積分量
#CLM = np.zeros(1)
# for i in range(len(nd)):
#    CLM += nd[i]*2E+5 #molecules cm-3
# DU = CLM/2.69E+16 #DU
#DU1 = DU.astype(np.float16)
#DU2 = DU1.astype(np.str)
# print(DU2)

# (1)ドップラー幅νD(T)
νD = []
for i in range(len(vij)):
    νD.append((vij[i]/c)*((2*k*T*Na)/M)**(1/2))  # cm-1
print('ドップラー幅', νD)

# %%
# (2)ローレンツ幅νL(p,T)
mixCO2 = 0.953
Pself = mixCO2*P  # 分圧
Pref = 1013250  # hPa(1atm) --> Ba
νL = []
for i in range(len(vij)):
    νL.append(((Tref/T)**nair[i])*(gammaair[i] *
              (P-Pself)/Pref+gammaself[i]*Pself/Pref))  # cm-1

print('ローレンツ幅', νL)

# nyo=np.stack([νD,νL],1)
# np.savetxt('doppler_lorentzian.dat',nyo)
# %%
# (3)x
x = np.zeros((len(vij), len(T), len(ν)))
vijs = np.zeros((len(vij), len(T)))
for i in range(len(vij)):
    for j in range(len(T)):
        vijs[i, j] = vij[i] + ((δair[i]*P[j]/Pref))
        for k in range(len(ν)):
            x[i, j, k] = ((ν[k]-vijs[i][j])/νD[i][j])

print('x', x)

# (4)y
y = np.zeros((len(vij), len(T)))
for i in range(len(vij)):
    y[i, :] = (νL[i]/νD[i])

print('y', y)

# %%
# (5)Voigt function f(ν,p,T)
# ベタ計算
iy1, err1 = (np.zeros((len(vij), len(T), len(ν))),
             np.zeros((len(vij), len(T), len(ν))))
la = np.zeros((len(vij), len(T), len(ν)))
f = np.zeros((len(vij), len(T), len(ν)))
for i in range(len(vij)):
    for j in range(len(T)):
        for k in range(len(ν)):
            def la(t): return ((np.e**(-t**2))/((x[i][j, k]-t)**2+y[i][j]**2))
            iy1[i, j, k], err1[i, j, k] = integrate.quad(la, -5, 5)
            f[i, j, k] = (y[i][j]/np.pi)*iy1[i][j, k] / \
                νD[i][j]/np.sqrt(np.pi)  # cm^-1

print('Voigt function', f)


# %%
# (6)吸収線強度S(T)
m = 1
S = np.zeros((len(vij), len(T)))
for i in range(len(vij)):
    S[i, :] = (Sij[i]*(Qref/QT)*((np.exp((-c2*E[i])/T))/(np.exp((-c2*E[i])/Tref))) *
               ((1.0-np.exp((-c2*vij[i])/T))/(1-np.exp((-c2*vij[i])/Tref))))  # cm-1/(molecule-1 cm2)
print('S', S)
# %%
# (7)吸収係数σ(z, ν)
σ = np.zeros((len(vij), len(T), len(ν)))
for i in range(len(ν)):
    σ[:, :, i] = S * f[:, :, i]

# for i in range(len(vij)):
#    for j in range(len(T)):
#        for k in range(len(ν)):
#            σ[i,j,k] = S[i][j] * K[i,j][k]
#    σ[i,j,k] = S[i][j]*f[i][j,k] #molecule-1 cm2 Voigt functionを使用して計算


print('吸収係数', σ)

# フォークト線形
# fff=f[i][j,k]/np.sqrt(np.pi)/νD[i][j]
# print('フォークト線形',fff)

# %%
# (8-1)光学的厚みτ(ν)：シングルライン計算
# q = USSA_q #cm-3
#lb = np.zeros((len(vij),len(T),len(ν)))
τ = np.zeros((len(vij), len(ν)))
for i in range(len(vij)):
    for k in range(len(ν)):
        for j in range(len(T)):
            if j == 0:
                τ[i, k] = 0.5 * σ[i, j, k]*nd[j]*mixCO2*(2*10**5)
            elif j == len(T)-1:
                τ[i, k] += 0.5 * σ[i, j, k] * nd[j] * mixCO2 * (2*10**5)
            else:
                τ[i, k] += 0.5 * σ[i, j, k]*nd[j] * mixCO2 * (4*10**5)

            # lb[i,j,k] = σ[i][j,k]*nd[j]*(2*10**5)
        # 適当な温度と適当な温度を使って線の広がりを見る
#                τ[i,k] += lb[i][j,k] #無次元

# 光学的厚みの足し合わせは、台形近似をして積分を行っている

# (8-2)：マルチライン増やす(ラインバイライン計算)
sumτ = np.sum(τ, axis=0)

print('光学的厚み', sumτ)

τ_v = np.stack([ν, sumτ], 1)
np.savetxt('new_4971-4976_voigt_test.txt', τ_v, fmt='%.10e')

# (9)各高度の吸収線形
#A = np.exp(-τ[i])
# print(τ)

""""
#(10)装置関数
Twn = np.zeros((len(ν)))
Twd = np.zeros((len(ν)))
Tw = np.zeros((len(ν)))
#W = np.loadtxt("")

width = 35

W1 = np.zeros(width+1)
W2 = np.zeros(width+1)
for l in range(0,width):
    W1[l] = (width-l)/width
W2 = W1[::-1] #装置関数を逆順にする

W = np.concatenate([W2, W1]) 

W = np.delete(W, 35) #1が2回あるので1回分削除


for j in range(len(ν)):
    for e in range(len(W)):
        if  j+e-width < len(ν):
            Twn[j] += A[j+e-width]*W[e]
            Twd[j] += W[e]
            Tw[j] = Twn[j]/Twd[j]
print(Tw) 

#データフレーム作成
df = pd.DataFrame(ν, columns=['nu'])
df['A'] = A
df['Tw'] = Tw

dfs = pd.DataFrame(τ)

"""

# ---------------グラフ作成----------------------------------------------
# データ読み込み&定義
x1 = ν
y1 = sumτ
#y2 = df1['A']

fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
ax.plot(x1, y1, color='b')
ax.set_xlim(4973, 4975)
ax.set_yscale('log')
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

# ax.plot((Sato['波数(cm-1)']),Satoτ,color='r')

# 凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=14)
plt.show()

# %%
