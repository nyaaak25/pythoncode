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

#hapi.pyファイルを読み込んで、HITRAN line-by-lineからパラメータを持ってくる作業
from hapi import *

# スペクトルの範囲：変更するところ
spectrum_begin = 4545 # cm^-1
spectrum_end = 5556

#4973 --> 大体2um
#1.8 umから2.2umまでは4545cm-1から5556cm-1

# HITRAN のパラメータ
name = 'CO2' 
moleculeID = 2
isotopologueID = 1

#CO2同位体は一個のみ

db_begin('data')
fetch(name, moleculeID, isotopologueID, spectrum_begin, spectrum_end)

# 取得したテーブルから、必要な情報を取得
vij,Sij,E,gammaair,gammaself,nair,δair = getColumns(name, ['nu', 'sw','elower','gamma_air','gamma_self','n_air','delta_air']) 
print('nu=',vij,'Sij=',Sij,'γself=',gammaself,'γair=',gammaair,'E"=',E,'nair=',nair,'δair=',δair)

#HITRANから引っ張ってきた吸収線データをテキストファイルに落とし込む
#savearray = np.array([vij,Sij,gammaself,gammaair,E,nair,δair])
#np.savetxt('savearray1.txt',savearray.T)


"ここからスペクトル計算"
# %%
#インポート
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

#Hitrandata = np.loadtxt('4971-4976_hitrandata.txt')
#vij = Hitrandata[:,0]
#Sij = Hitrandata[:,1]
#gammaair = Hitrandata[:,3]
#gammaself = Hitrandata[:,2]
#E = Hitrandata[:,4]
#nair = Hitrandata[:,5]
#δair = Hitrandata[:,6]

#CGS単位系
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

#Q(T)　温度によって変更、ここのInputfileの計算は別プログラム(Qcaluculation.py)
Q_CO2=pd.read_csv(filepath_or_buffer="CO2_Q.csv", encoding="cp932", sep=",")
QT=Q_CO2['Q']
Qref=286.09

#Number Density
mixCO2=0.95
R=8.31E+7  #(g*cm^2*s^2)/(mol*K)
Pself=mixCO2*P
#nd=((Pself*Na)/(R*T))
nd = P / (k * T)

#波数幅：計算する波数を決定　変更するパラメータ
#最後まで使う
ν = np.zeros(1011)
for i in range(1011):
    ν[i] = 4545.0000+(1.00*i)

#1.8 cm-1から2.2m-1までは4545cm-1から5556cm-1

print('波数',ν)

#(1)ドップラー幅νD(T)
#Voigt functionの計算まで使用
νD=[]
for i in range(len(vij)):
    νD.append((vij[i]/c)*((2*k*T*Na)/M)**(1/2)) #cm-1
print('ドップラー幅',νD)

#%%
#(2)ローレンツ幅νL(p,T) 
mixCO2 = 0.9532
Pself = mixCO2*P #分圧
Pref = 1013250 #hPa(1atm) --> Ba

νL=[]
for i in range(len(vij)):
    νL.append(((Tref/T)**nair[i])*(gammaair[i]*(P-Pself)/Pref+gammaself[i]*Pself/Pref)) #cm-1

print('ローレンツ幅',νL)

#gammerairとgammerself、nairを削除
del gammaair,gammaself,nair

#nyo=np.stack([νD,νL],1)
#np.savetxt('doppler_lorentzian.dat',nyo)
# %%
#(3)x
x = np.zeros((len(vij), len(T), len(ν)))
vijs =np.zeros((len(vij), len(T)))
for i in range(len(vij)):
    for j in range(len(T)):
        vijs[i,j] = vij[i] + ((δair[i]*P[j]/Pref))
        for k in range(len(ν)):
            x[i,j,k] = ((ν[k]-vijs[i][j])/νD[i][j])

print('x',x)

#vijsとδairを削除
del vijs,δair

#(4)y 
y = np.zeros((len(vij),len(T)))
for i in range(len(vij)):
    y[i,:] = (νL[i]/νD[i])

print('y',y)

#νLを削除
del νL

# %%
#(5)Voigt function f(ν,p,T)
#近似式導入 [M.Kuntz 1997 etal.]
#|x|+yの計算
xy = np.zeros((len(vij),len(T),len(ν)))
for i in range(len(ν)):
    xy[:,:,i] = abs(x[:,:,i])+y

#-y + 0.195|x|-0.176の計算
x_y = np.zeros((len(vij),len(T),len(ν)))
for i in range(len(ν)):
    x_y[:,:,i] = -y + 0.195*abs(x[:,:,i]) - 0.176
print(x_y)

# %%
K1 = np.zeros((len(vij),len(T),len(ν)))
K2 = np.zeros((len(vij),len(T),len(ν)))
K3 = np.zeros((len(vij),len(T),len(ν)))
K4 = np.zeros((len(vij),len(T),len(ν)))

for i in range(len(ν)):
    #Region1 　|x|+y >15 の領域
    a1 = 0.2820948*y + 0.5641896*y**3
    b1 = 0.5641896*y
    a2 = 0.5 + y**2 + y**4
    b2 = -1 + 2*y**2

    K1[:,:,i] = (a1 + b1*x[:,:,i]**2)/(a2+b2*x[:,:,i]**2+x[:,:,i]**4)
    print('第一近似',i)

    del a1,a2,b1,b2

    #Region2 　5.5 < |x|+y < 15の領域
    a3 = 1.05786*y + 4.65456*y**3 + 3.10304*y**5 + 0.56419*y**7
    b3 = 2.962*y + 0.56419*y**3 + 1.69257*y**5
    c3 = 1.69257*y**3 - 2.53885*y
    d3 = 0.56419*y
    a4 = 0.5625 + 4.5*y**2 + 10.5*y**4 + 6*y**6 + y**8
    b4 = -4.5 + 9*y**2 + 6*y**4 + 4*y**6
    c4 = 10.5 -6*y**2 +6*y**4
    d4 = -6 + 4*y**2

    K2[:,:,i] = (a3 + b3*x[:,:,i]**2 + c3*x[:,:,i]**4 + d3*x[:,:,i]**6) /(a4 + b4*x[:,:,i]**2 + c4*x[:,:,i]**4 + d4*x[:,:,i]**6 + x[:,:,i]**8)
    print('第二近似',i)

    del a3,b3,c3,d3,a4,b4,c4,d4

    #Region3　|x|+y < 5.5 and y > 0.195|x|-0.176
    a5 = 272.102 + 973.778*y + 1629.76*y**2 + 1678.33*y**3 + 1174.8*y**4 + 581.746*y**5 + 204.501*y**6 + 49.5213*y**7 + 7.55895*y**8 + 0.564224*y**9
    b5 = -60.5644 -2.34403*y + 220.843*y**2 + 336.364*y**3 + 247.198*y**4 + 100.705*y**5 + 22.6778*y**6 + 2.25689*y**7
    c5 = 4.58029 + 18.546*y + 42.5683*y**2 + 52.8454*y**3 + 22.6798*y**4 + 3.38534*y**5
    d5 = -0.128922 + 1.66203*y + 7.56186*y**2 + 2.25689*y**3
    e5 = 0.000971457 + 0.564224*y
    a6 = 272.102 + 1280.83*y + 2802.87*y**2 + 3764.97*y**3 + 3447.63*y**4 + 2256.98*y**5 + 1074.41*y**6 + 369.199*y**7 + 88.2674*y**8 + 13.3988*y**9 + y**10
    b6 = 211.678 + 902.306*y + 1758.34*y**2 + 2037.31*y**3 + 1549.68*y**4 + 793.427*y**5 + 266.299*y**6 + 53.5952*y**7 + 5*y**8
    c6 = 78.866 + 308.186*y + 497.302*y**2 + 479.258*y**3 + 269.292*y**4 + 80.3928*y**5 + 10*y**6
    d6 = 22.0353 + 55.0293*y + 92.7568*y**2 + 53.5952*y**3 + 10*y**4
    e6 = 1.49645+ 13.3988*y + 5*y**2

    K3[:,:,i] = (a5 + b5*x[:,:,i]**2 + c5*x[:,:,i]**4 + d5*x[:,:,i]**6 + e5*x[:,:,i]**8 )/(a6+b6*x[:,:,i]**2+c6*x[:,:,i]**4 + d6*x[:,:,i]**6 + e6*x[:,:,i]**8 + x[:,:,i]**10)
    print('第三近似',i)

    del a5,a6,b5,b6,c5,c6,d5,d6,e5,e6

    #Region4　|x|+y < 5.5 and y < 0.195|x|-0.176
    a7 = 1.16028e9*y - 9.86604e8*y**3 + 4.56662e8*y**5 - 1.53575e8*y**7 + 4.08168e7*y**9 - 9.69463e6*y**11 + 1.6841e6*y**13 - 320772*y**15 + 40649.2*y**17 - 5860.68*y**19 + 571.687*y**21 - 72.9359*y**23 + 2.35944*y**25 - 0.56419*y**27
    b7 = -5.60505e8*y - 9.85386e8*y**3 + 8.06985e8*y**5 - 2.91876e8*y**7 + 8.64829e7*y**9 - 7.72359e6*y**11 + 3.59915e6*y**13 - 234417*y**15 + 45251.3*y**17 - 2269.19*y**19 - 234.143*y**21 + 23.0312*y**23 - 7.33447*y**25
    c7 = -6.51523e8*y + 2.47157e8*y**3 + 2.94262e8*y**5 - 2.04467e8*y**7 + 2.29302e7*y**9 - 2.3818e7*y**11 + 576054*y**13 + 98079.1*y**15 - 25338.3*y**17 + 1097.77*y**19 + 97.6203*y**21 - 44.0068*y**23
    d7 = -2.63894e8*y + 2.70167e8*y**3 - 9.96224e7*y**5 - 4.15013e7*y**7 + 3.83112e7*y**9 + 2.2404e6*y**11 - 303569*y**13 - 66431.2*y**15 + 8381.97*y**17 + 228.563*y**19 - 161.358*y**21
    e7 = -6.31771e7*y + 1.40677e8*y**3 + 5.56965e6*y**5 + 2.46201e7*y**7 + 468142*y**9 - 1.003e6*y**11 - 66212.1*y**13 + 23507.6*y**15 + 296.38*y**17 - 403.396*y**19
    f7 = -1.69846e7*y + 4.07382e6*y**3 - 3.32896e7*y**5 - 1.93114e6*y**7 - 934717*y**9 + 8820.94*y**11 + 37544.8*y**13 + 125.591*y**15 - 726.113*y**17
    g7 = -1.23165e6*y + 7.52883e6*y**3 - 900010*y**5 - 186682*y**7 + 79902.5*y**9 + 37371.9*y**11 - 260.198*y**13 - 968.15*y**15
    h7 = -610622*y + 86407.6*y**3 + 153468*y**5 + 72520.9*y**7 + 23137.1*y**9 - 571.645*y**11 - 968.15*y**13
    o7 = -23586.5*y + 49883.8*y**3 + 26538.5*y**5 + 8073.15*y**7 - 575.164*y**9 - 726.113*y**11
    p7 = -8009.1*y + 2198.86*y**3 + 953.655*y**5 - 352.467*y**7 - 403.396*y**9
    q7 = -622.056*y - 271.202*y**3 - 134.792*y**5 - 161.358*y**7
    r7 = -77.0535*y - 29.7896*y**3 - 44.0068*y**5
    s7 = -2.92264*y - 7.33447*y**3
    t7 = -0.56419*y
    a8 = 1.02827e9 - 1.5599e9*y**2 + 1.17022e9*y**4 - 5.79099e8*y**6 + 2.11107e8*y**8 - 6.11148e7*y**10 + 1.44647e7*y**12 - 2.85721e6*y**14 + 483737*y**16 - 70946.1*y**18 + 9504.65*y**20 - 955.194*y**22 + 126.532*y**24- 3.68288*y**26 + y**28
    b8 = 1.5599e9 - 2.28855e9*y**2 + 1.66421e9*y**4 - 7.53828e8*y**6 + 2.89676e8*y**8 - 7.01358e7*y**10 + 1.39465e7*y**12 - 2.84954e6*y**14 + 498334*y**16 - 55600*y**18 + 3058.26*y**20 + 533.254*y**22 - 40.5117*y**24 + 14*y**26
    c8 = 1.17022e9 - 1.66421e9*y**2 + 1.06002e9*y**4 - 6.60078e8*y**6 + 6.33496e7*y**8 - 4.60396e7*y**10 + 1.4841e7*y**12 - 1.06352e6*y**14 - 217801*y**16 + 48153.3*y**18 - 1500.17*y**20- 198.876*y**22+ 91*y**24
    d8 = 5.79099e8 - 7.53828e8*y**2 + 6.60078e8*y**4 + 5.40367e7*y**6 + 1.99846e8*y**8 - 6.87656e6*y**10- 6.89002e6*y**12 + 280428*y**14 + 161461*y**16 - 16493.7*y**18 - 567.164*y**20 + 364*y**22
    e8 = 2.11107e8 - 2.89676e8*y**2 + 6.33496e7*y**4 - 1.99846e8*y**6 - 5.01017e7*y**8 - 5.25722e6*y**10 + 1.9547e6*y**12 + 240373*y**14 - 55582*y**16 - 1012.79*y**18+ 1001*y**20
    f8 = 6.11148e7 - 7.01358e7*y**2 + 4.60396e7*y**4 - 6.87656e6*y**6 + 5.25722e6*y**8 + 3.04316e6*y**10 + 123052*y**12 - 106663*y**14 - 1093.82*y**16 + 2002*y**18
    g8 = 1.44647e7 - 1.39465e7*y**2 + 1.4841e7*y**4 + 6.89002e6*y**6 + 1.9547e6*y**8 - 123052*y**10 - 131337*y**12 - 486.14*y**14 + 3003*y**16
    h8 = 2.85721e6 - 2.84954e6*y**2 + 1.06352e6*y**4 + 280428*y**6 - 240373*y**8 - 106663*y**10 + 486.14*y**12 + 3432*y**14
    o8 = 483737 - 498334*y**2 - 217801*y**4 - 161461*y**6 - 55582*y**8 + 1093.82*y**10 + 3003*y**12
    p8 = 70946.1 - 55600*y**2 -48153.3*y**4 - 16493.7*y**6 + 1012.79*y**8 + 2002*y**10
    q8 = 9504.65 - 3058.26*y**2 - 1500.17*y**4 + 567.164*y**6 + 1001*y**8
    r8 = 955.194 + 533.254*y**2 + 198.876*y**4 + 364*y**6
    s8 = 126.532 + 40.5117*y**2 + 91*y**4
    t8 = 3.68288 + 14*y**2

    K4[:,:,i] = (np.exp(y**2) / np.exp(x[:,:,i]**2)) * np.cos(2*x[:,:,i]*y) - ((a7 + b7*x[:,:,i]**2 + c7*x[:,:,i]**4 + d7*x[:,:,i]**6 + e7*x[:,:,i]**8 + f7*x[:,:,i]**10 + g7*x[:,:,i]**12 + h7*x[:,:,i]**14 + o7*x[:,:,i]**16 + p7*x[:,:,i]**18 + q7*x[:,:,i]**20 + r7*x[:,:,i]**22 + s7*x[:,:,i]**24 + t7*x[:,:,i]**26 )/(a8+b8*x[:,:,i]**2+c8*(x[:,:,i])**4 + d8*x[:,:,i]**6 + e8*x[:,:,i]**8 +f8*x[:,:,i]**10 + g8*x[:,:,i]**12 + h8*x[:,:,i]**14 + o8*x[:,:,i]**16 + p8*x[:,:,i]**18 + q8*x[:,:,i]**20 + r8*x[:,:,i]**22 + s8*x[:,:,i]**24 + t8*x[:,:,i]**26 + x[:,:,i]**28))
    print('第4近似',i)

    del a7,a8,b7,b8,c7,c8,d7,d8,e7,e8,f7,f8,g7,g8,h7,h8,o7,o8,p7,p8,q7,q8,r7,r8,s7,s8,t7,t8

#xとyを削除
del x,y

#Region1〜Region4を満たす要素番号場所を検索
C1 = np.where(xy > 15)
C2 = np.where((xy < 15) & (xy >5.5))
C3 = np.where((xy <5.5) & (0 > x_y))
C4 = np.where((xy <5.5) & (0 < x_y))

#xyとx_yを削除
del xy,x_y

# %%
#K1~K4の要素番号にC1のlistを代入
#初期化
KK1 = np.zeros(K1.shape)
KK2 = np.zeros(K2.shape)
KK3 = np.zeros(K3.shape)
KK4 = np.zeros(K4.shape)

#具体的な値を代入
KK1[C1] = K1[C1]
KK2[C2] = K2[C2]
KK3[C3] = K3[C3]
KK4[C4] = K4[C4]

#多項式近似ができたVoigt functionの式 K
K = KK1 + KK2 + KK3 + KK4
print("K",K)
#KK1~KK4,K1~K4を削除
del KK1,KK2,KK3,KK4,K1,K2,K3,K4

for i in range(len(ν)):
    K[:,:,i] = K[:,:,i]/νD/np.sqrt(np.pi)

#νDとx,yを削除
del νD

# %%
#(6)吸収線強度S(T)
m = 1
S = np.zeros((len(vij),len(T)))
for i in range(len(vij)):
    S[i,:] = (Sij[i]*(Qref/QT)*((np.exp((-c2*E[i])/T))/(np.exp((-c2*E[i])/Tref)))*((1.0-np.exp((-c2*vij[i])/T))/(1-np.exp((-c2*vij[i])/Tref)))) #cm-1/(molecule-1 cm2)
print('S',S)

#EとSijを削除
del Sij,E
# %%
#(7)吸収係数σ(z, ν)
σ = np.zeros((len(vij),len(T), len(ν)))
for i in range(len(ν)):
    σ[:,:,i] = S * K[:,:,i] #molecule-1 cm2 Voigt functionを使用して計算

print('吸収係数',σ)

#SとKを削除
del S,K
# %%
#(8-1)光学的厚みτ(ν)：シングルライン計算
τ = np.zeros((len(vij),len(ν)))
for i in range(len(vij)):
    for k in range(len(ν)):
        for j in range(len(T)):
            if j==0:
                τ[i,k]= 0.5 * σ[i,j,k]*nd[j]*mixCO2*(2e5)
            elif j== len(T)-1:
                τ[i,k] += 0.5 * σ[i,j,k] * nd[j] * mixCO2 * (2e5)
            else:
                τ[i,k]+= 0.5 * σ[i,j,k]*nd[j] *mixCO2* (4e5)
 
        #適当な温度と適当な温度を使って線の広がりを見る
#光学的厚みの足し合わせは、台形近似をして積分を行っている

#(8-2)：マルチライン増やす(ラインバイライン計算)
sumτ=np.sum(τ,axis=0)

print('光学的厚み',sumτ)

τ_v=np.stack([ν,sumτ],1)
np.savetxt('new4545-5556_kinji.txt',τ_v,fmt ='%.10e')

#(9)各高度の吸収線形 
#A = np.exp(-τ[i])
#print(τ)

#---------------グラフ作成----------------------------------------------
#データ読み込み&定義
x1 = ν
y1 = sumτ

fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
ax.plot(x1, y1, color='b')
ax.set_xlim(4973,4975)
ax.set_yscale('log') 
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

#凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=14)
plt.show()
# %%
