# %%
# インポート
import matplotlib.pylab as plt
from pyrsistent import v
from scipy.io import readsav
import scipy.io as sio
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
import time
from numba import jit, f8

# %%
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = 'work_ORB0920_3.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())
lati = sav_data['lati']
longi = sav_data['longi']

pressure = sav_data['pressure']
pressure[pressure <= 0] = np.nan

fig = plt.figure(figsize=(6, 13), dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.set_xlim(271, 278)
ax.set_ylim(50, 61)
im = ax.scatter(longi, lati, c=pressure, s=2)
fig.colorbar(im, orientation='horizontal')

# %%
# linfitさせて吸収の深さを計算する
ARS = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP12_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x = ARS[:, 0]
ARS_x = (1/ARS_x)*10000
ARS_x = ARS_x[::-1]
ARS_y = ARS[:, 1]
ARS_y = ARS_y[::-1]

# 最小二乗法でfitting
POLY_x = [ARS_x[0], ARS_x[3], ARS_x[5], ARS_x[23], ARS_x[24], ARS_x[25]]
POLY_y = [ARS_y[0], ARS_y[3], ARS_y[5], ARS_y[23], ARS_y[24], ARS_y[25]]
a, b = np.polyfit(POLY_x, POLY_y, 1)

cont0 = b + a*ARS_x
y_calc = 1 - ARS_y/cont0
y_total = np.sum(y_calc[6:17])

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.set_xlabel('Wavenumber [μm]', fontsize=10)
ax.plot(ARS_x, ARS_y, color='blue', label="fitting", zorder=2)
ax.plot(ARS_x, cont0, color='red', label="Observation", zorder=1)
ax.axvline(x=ARS_x[3])
ax.axvline(x=ARS_x[20])
# 9:19が1.94~2.09μm
# 9:13は1.94〜1.99μm
# 3:20が1.85~2.10μm
# 9:16が1.93~2.04μm

# IDLで計算されたものとの整合性を取るテスト
# %%

data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = '/Users/nyonn/IDLWorkspace/Default/savfile/ORB0313_4.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())
wvl = sav_data['wvl']

specmars = np.loadtxt(
    '/Users/nyonn/IDLWorkspace/Default/profile/specsol_0403.dat')
dmars = sav_data['dmars']
specmars = specmars/dmars/dmars

CO2 = np.where((wvl > 1.81) & (wvl < 2.19))
specmars = specmars[CO2]
wvl = wvl[CO2]

jdat = sav_data['jdat']
# %%
nwvl = len(wvl)
io = len(jdat[1, 1, :])
ip = len(jdat[:, 1, 1])
flux = np.zeros((io, nwvl, ip))

for i in range(io):
    for o in range(ip):
        # flux[i, :, o] = jdat[o, CO2, i]/specmars
        flux[i, :, o] = jdat[o, CO2, i]

longi = sav_data['longi']
lati = sav_data['lati']
maxlongi = np.max(longi)
minlongi = np.min(longi)
maxlati = np.max(lati)
minlati = np.min(lati)

x = [wvl[0], wvl[1], wvl[2], wvl[23], wvl[24], wvl[25]]
y = [flux[0, 0, 0], flux[0, 1, 0], flux[0, 2, 0],
     flux[0, 23, 0], flux[0, 24, 0], flux[0, 25, 0]]
a, b = np.polyfit(x, y, 1)
cont = b + a*wvl

radiance = jdat[0, :, 0]

"""
radiance = [3.04291914e-01, 3.01133918e-01, 2.99194302e-01, 3.01644108e-01,
       2.93011375e-01, 2.85192736e-01, 2.91648333e-01, 2.87336526e-01,
       2.97353790e-01, 2.67734043e-01, 1.59216543e-01, 1.92891786e-01,
       2.73729659e-01, 1.86171653e-01, 7.47282105e-02, 1.98996083e-01,
       np.nan, 1.82028307e-01, 1.88777015e-01, 2.46611916e-01,
       2.74026413e-01, 2.91733514e-01, 2.99147348e-01, 3.06648120e-01,
       2.97765150e-01, 2.98931135e-01, np.nan]
"""

radiance = [6.0231223e+00, 5.8426933e+00, 5.6736460e+00, 5.5487056e+00,
            5.1142383e+00, 4.8854246e+00, 4.9333162e+00, 4.7842565e+00,
            4.7933760e+00, 4.1179385e+00, 2.4188302e+00, 2.8826861e+00,
            3.9918549e+00, 2.6615460e+00, 1.0469103e+00, 2.7060721e+00,
            np.nan, 2.3625975e+00, 2.3858469e+00, 3.0465832e+00,
            3.2991776e+00, 3.4420812e+00, 3.4416115e+00, 3.4568799e+00,
            3.2720873e+00, 3.1409810e+00, np.nan]

width = 1 - (radiance/cont)
band = np.where((wvl >= 1.94) & (wvl <= 1.99))
obs_spec = np.nansum(width[band])

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)
ax.plot(wvl, cont, color='red', label="fitting", zorder=2)
ax.plot(wvl, radiance, color='blue', label="Observation", zorder=1)
ax.scatter(x, y)
ax.axvline(x=wvl[9])
ax.axvline(x=wvl[12])

# where_xyzでくじけてやめた


# %%
# Look Up Tabel用のprofileを読み込み
def filesearch(dir):
    # 指定されたディレクトリ内の全てのファイルを取得
    path_list = glob.glob("LookUpTable_HTP/*.txt")
    name_list = []                          # ファイル名の空リストを定義
    ext_list = []                           # 拡張子の空リストを定義
    out_list = []                           # 保存パスの空リストを定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in path_list:
        file = os.path.basename(i)          # 拡張子ありファイル名を取得
        name, ext = os.path.splitext(file)  # 拡張子なしファイル名と拡張子を取得
        name_list.append(name)              # 拡張子なしファイル名をリスト化
        ext_list.append(ext)                # 拡張子をリスト化
        out_list.append(os.path.join(
            *[dir, name + '_resize' + ext]))  # 保存パスを作成
    return path_list, name_list, ext_list, out_list
# %%
# 気圧と温度プロファイルを導出する


# parameter指定 MKS単位系
g = 3.72  # m s-1
# R = 8.314  # E+7  # (g*cm^2*s^2)/(mol*K)
R = 192
T1 = [160, 213, 260]  # K
T2 = [80, 146, 200]  # K
Surface_pressure = [50, 150, 180, 215, 257, 308,
                    369, 442, 529, 633, 758, 907, 1096, 1300, 1500]  # Pa

# hight profile
Hight_km = np.arange(0, 62, 2)  # km
Hight = Hight_km * 1000  # m

# Scale Hight
SH = (R*T1[1])/g

# pressure profile
pre = Surface_pressure[9] * np.exp(-Hight/SH)

# Temp profile
H1 = SH*0.1
H2 = SH*4.0

a = (T1[1]-T2[1])/(H1-H2)
b = T1[1]-a*H1

Temp = np.zeros(len(Hight))
for i in range(len(Hight)):
    if Hight[i] <= H2:
        Temp[i] = a * Hight[i] + b
    else:
        Temp[i] = T2[1]

print(Temp)
# %%

# 波数
v_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
v1 = v_txt[:, 0]  # cm-1
cm_v = (1/v1)*10000
wav = cm_v[::-1]  # um

# 光学的厚み→放射強度
tau_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
tau1 = tau_txt[:, 1]
tau = tau1[::-1]
sza_theta = 18.986036
I0 = np.exp(-tau/np.cos(sza_theta))
Iobs = I0 * np.exp(-tau)

OMEGAcenter_list = [1.8143300, 1.8284900, 1.8426300, 1.8567700, 1.8708900, 1.8850000, 1.8990901, 1.9131700, 1.9272400, 1.9412900, 1.9553300, 1.9693500, 1.9833500,
                    1.9973400, 2.0113201, 2.0252800, 2.0392201, 2.0531399, 2.0670500, 2.0809400, 2.0948100, 2.1086600, 2.1224899, 2.1363101, 2.1501000, 2.1638801, 2.1776299]

OMEGAchannel = np.zeros(len(OMEGAcenter_list))


# def GAUSSIAN_func():
#        C1 = np.where((wav < mu + 0.013) & (mu - 0.013 < wav))
#    new_wav = wav[C1]
#    fx = (1/np.sqrt(2*np.pi*(sig**2)))*np.exp(-((new_wav-mu)**2)/(2*sig**2))

for k in range(len(OMEGAcenter_list)):
    # OMEGAの中心波長aについての GAUSSIANを定義
    mu = OMEGAcenter_list[k]
    sig = 6.5e-3

    # wav 1 ~ wav n までのガウシアンの値を求める
    C1 = np.where((wav <= mu + 0.013) & (mu - 0.013 < wav))
    new_wav = wav[C1]
    new_I = Iobs[C1]
    average_I = np.average(new_I)

    OMEGAchannel[k] = average_I

tau_v = np.stack([OMEGAcenter_list, OMEGAchannel], 1)
np.savetxt('0.01_BOXinstrumentfunction.txt', tau_v, fmt='%.10e')

plt.plot(OMEGAcenter_list, OMEGAchannel)
plt.show()

# %%
"""
for k in range(len(OMEGAcenter_list)):
    # OMEGAの中心波長aについての GAUSSIANを定義
    mu = OMEGAcenter_list[k]
    sig = 6.5e-3

    # wav 1 ~ wav n までのガウシアンの値を求める
    C1 = np.where((wav <= mu + 0.013) & (mu - 0.013 < wav))
    new_wav = wav[C1]
    GAUSSIAN_func = (1/np.sqrt(2*np.pi*(sig**2))) * \
        np.exp(-((new_wav-mu)**2)/(2*sig**2))

    # ガウシアンとスペクトルをかけ合わせる
    new_I = Iobs[C1]
    multiple = GAUSSIAN_func * new_I

    # 畳み込みの台形近似で積分を行う
    S_conv = 0
    for i in range(len(new_wav)-1):
        S_conv += ((multiple[i]+multiple[i+1])*(new_wav[i+1]-new_wav[i]))/2

    # ガウシアンの台形近似で積分を行う
    S_gauss = 0
    for i in range(len(new_wav)-1):
        S_gauss += ((GAUSSIAN_func[i]+GAUSSIAN_func[i+1])
                    * (new_wav[i+1]-new_wav[i]))/2

    # OMEGA channelに落とし込み
    OMEGAchannel[k] = S_conv/S_gauss
    # OMEGAchannelに足し合わせるのではなく、OMEGAchannel[k]にS_conv/S_gaussを格納する
    # print(OMEGAchannel)

tau_v = np.stack([OMEGAcenter_list, OMEGAchannel], 1)
np.savetxt('test_OMEGAinstrumentfunction.txt', tau_v, fmt='%.10e')

plt.plot(OMEGAcenter_list, OMEGAchannel)
plt.show()
"""
# %%


def GAUSSIAN(sig, mu, x):
    f_x = (1/np.sqrt(2*np.pi*sig))*np.exp(-((x-mu)**2)/(2*sig**2))

    return f_x


def main():
    spectral = np.zeros((len(OMEGAcenter_list)))
    sig = 6.5e-3
    # 波数
    v_txt = np.loadtxt('4545-5556_0.001step_cutoff_120.txt')
    v1 = v_txt[:, 0]  # cm-1
    cm_v = (1/v1)*10000
    v = cm_v[::-1]  # um

    # 光学的厚み
    tau_txt = np.loadtxt('4545-5556_0.001step_cutoff_120.txt')
    x = tau_txt[:, 1]

    for k in range(len(OMEGAcenter_list)):
        mu = OMEGAcenter_list[k]
        f_xk = GAUSSIAN(sig, mu, x)

        spectral += f_xk

    # データセーブ
    tau_v = np.stack([OMEGAcenter_list, spectral], 1)
    np.savetxt('test_OMEGAinstrumentfunction.txt', tau_v, fmt='%.10e')


if __name__ == '__main__':
    main()


# %%
"""
# Create the initial function. I model a spike
# as an arbitrarily narrow Gaussian
mu = 1.0  # Centroid
sig = 0.001  # Width
original_pdf = stats.norm(mu, sig)

# 波数
v_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
v1 = v_txt[:, 0]
cm_v = (1/v1)*10000
v = cm_v[::-1]

# 光学的厚み
tau_txt = np.loadtxt('4545-5556_0.01step_cutoff_120.txt')
cut120 = tau_txt[:, 1]

# x = np.linspace(0.0, 2.0, 1000)
# y = original_pdf.pdf(x)
# plt.plot(v, cut120, label='original')


# Create the ``smearing" function to convolve with the
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and
# width of 0.5

# mu = 1.8001601のとき
mu_conv = 1.8001601  # Centroid
sigma_conv = 20  # Width
convolving_term = stats.norm(mu_conv, sigma_conv)

# 中心波長からOMEGA channel分を持ってきている
C1 = np.where((v < mu_conv + 0.013) & (mu_conv - 0.013 < v))
print(v[C1])
print(cut120[C1])

x = v[C1]
y = cut120[C1]

xconv = np.linspace(-100, 100, 20000)
yconv = convolving_term.pdf(xconv)

convolved_pdf = signal.convolve(y, yconv, mode='same')/sum(y)

plt.plot(x, convolved_pdf, label='convolved')
# plt.ylim(0, 1.2*max(convolved_pdf))
plt.legend()
plt.show()

# %%
# 装置関数を実装
# randomなデータを用意


def f(x):
    if np.absolute(x) > 0.5:
        y = 0
    else:
        y = 1
    return y


X = np.linspace(-2.0, 2.0, num=100)

F = [f(x) for x in X]

G = [f(x) for x in X]

C = np.convolve(F, G)

# plt.plot(X,F)
plt.plot(C)

plt.title("1D convolution")
plt.show()

# %%
# sav file openプログラム
data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
sav_fname = 'ORB0006_1.sav'
sav_data = readsav(sav_fname)
print(sav_data.keys())
wvl = sav_data['wvl']

CO2 = np.where((wvl > 1.8) & (wvl < 2.2))
print(CO2)

"""
# %%
"""
Hitrandata = np.loadtxt('4545-5556_hitrandata.txt')
vij = Hitrandata[:, 0]
Sij = Hitrandata[:, 1]
gammaair = Hitrandata[:, 3]
gammaself = Hitrandata[:, 2]
E = Hitrandata[:, 4]
nair = Hitrandata[:, 5]
deltaair = Hitrandata[:, 6]

# CGS単位系
# file open＆定数定義
k = 1.380649E-16  # erg K-1
c = 2.9979E+10  # cm s-1
h = 6.62607015E-27  # erg s
c2 = (h*c)/k  # cm K (hc / k)
Na = 6.02214129E+23
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
QT = QT.to_numpy()
Qref = 286.09

# Number Density
mixCO2 = 0.9532
Pref = 1013250  # hPa(1atm) --> Ba
R = 8.31E+7  # (g*cm^2*s^2)/(mol*K)
Pself = mixCO2*P
# nd=((Pself*Na)/(R*T))
nd = P / (k * T)
# %%
# 波数幅：計算する波数を決定　変更するパラメータ
# 1.8 cm-1から2.2m-1までは4545cm-1から5556cm-1, 1011000(0.001Step)
# 最後まで使う

dv = 0.01
v = np.arange(4445, 5656, dv)


"""
"""
# 旧波数, 1011000(0.001Step)
v_all = np.zeros(1011000)
for i in range(1011000):
    v_all[i] = 4545.000 + (0.001*i)
    # v[i] = 4545.0000+(1.00*i)
# print('波数', v)
"""

"""
# (1)ドップラー幅νD(T)
# Voigt functionの計算まで使用


def Doppler(vijk):
    vD = ((vijk/c)*((2*k*T*Na)/M)**(1/2))  # cm-1
    # print('ドップラー幅', vD)

    return vD

# %%
# (2)ローレンツ幅νL(p,T)


def Lorenz(nairk, gammaairk, gammaselfk):
    Pself = mixCO2*P  # 分圧
    vL = (((Tref/T)**nairk)*(gammaairk*(P-Pself) /
          Pref+gammaselfk*Pself/Pref))  # cm-1
    # print('ローレンツ幅', vL)

    return vL
# nyo=np.stack([νD,νL],1)
# np.savetxt('doppler_lorentzian.dat',nyo)


# (3)x
def Voigt_x(vijk, deltaairk, vDk):
    x = np.zeros(((len(T), len(v))))
    vijs = np.zeros((len(T)))
    vijs = vijk + ((deltaairk*P/Pref))
    v_new = np.repeat(v[None, :], 31, axis=0)
    x = (v_new.T-vijs)/vDk
    x = x.T

    return x


# (4)y
@jit(nopython=True, fastmath=True)
def Voigt_y(vLk, vDk):
    y = vLk/vDk

    # print('y', y)
    return y


# Region separate
@jit(nopython=True, fastmath=True)
def K1_calc(xk, yk):
    # Region1 　|x|+y >15 の領域
    a1 = 0.2820948*yk + 0.5641896*yk**3
    b1 = 0.5641896*yk
    a2 = 0.5 + yk**2 + yk**4
    b2 = -1 + 2*yk**2

    # return (a1 + b1*xki**2)/(a2+b2*xki**2+xki**4)
    return ((a1 + b1*xk.T**2)/(a2+b2*xk.T**2+xk.T**4)).T


@jit(nopython=True, fastmath=True)
def K2_calc(xk, yk):
    a3 = 1.05786*yk + 4.65456*yk**3 + 3.10304*yk**5 + 0.56419*yk**7
    b3 = 2.962*yk + 0.56419*yk**3 + 1.69257*yk**5
    c3 = 1.69257*yk**3 - 2.53885*yk
    d3 = 0.56419*yk
    a4 = 0.5625 + 4.5*yk**2 + 10.5*yk**4 + 6*yk**6 + yk**8
    b4 = -4.5 + 9*yk**2 + 6*yk**4 + 4*yk**6
    c4 = 10.5 - 6*yk**2 + 6*yk**4
    d4 = -6 + 4*yk**2

    # return (a3 + b3*xki**2 + c3*xki**4 + d3*xki**6) / (a4 + b4*xki**2 + c4*xki**4 + d4*xki**6 + xki**8)
    return ((a3 + b3*xk.T**2 + c3*xk.T**4 + d3*xk.T**6) / (a4 + b4*xk.T**2 + c4*xk.T**4 + d4*xk.T**6 + xk.T**8)).T


@jit(nopython=True, fastmath=True)
def K3_calc(xk, yk):
    a5 = 272.102 + 973.778*yk + 1629.76*yk**2 + 1678.33*yk**3 + 1174.8*yk**4 + \
        581.746*yk**5 + 204.501*yk**6 + 49.5213*yk**7 + 7.55895*yk**8 + 0.564224*yk**9
    b5 = -60.5644 - 2.34403*yk + 220.843*yk**2 + 336.364*yk**3 + \
        247.198*yk**4 + 100.705*yk**5 + 22.6778*yk**6 + 2.25689*yk**7
    c5 = 4.58029 + 18.546*yk + 42.5683*yk**2 + \
        52.8454*yk**3 + 22.6798*yk**4 + 3.38534*yk**5
    d5 = -0.128922 + 1.66203*yk + 7.56186*yk**2 + 2.25689*yk**3
    e5 = 0.000971457 + 0.564224*yk
    a6 = 272.102 + 1280.83*yk + 2802.87*yk**2 + 3764.97*yk**3 + 3447.63*yk**4 + \
        2256.98*yk**5 + 1074.41*yk**6 + 369.199*yk**7 + \
        88.2674*yk**8 + 13.3988*yk**9 + yk**10
    b6 = 211.678 + 902.306*yk + 1758.34*yk**2 + 2037.31*yk**3 + 1549.68 * \
        yk**4 + 793.427*yk**5 + 266.299*yk**6 + 53.5952*yk**7 + 5*yk**8
    c6 = 78.866 + 308.186*yk + 497.302*yk**2 + 479.258 * \
        yk**3 + 269.292*yk**4 + 80.3928*yk**5 + 10*yk**6
    d6 = 22.0353 + 55.0293*yk + 92.7568*yk**2 + 53.5952*yk**3 + 10*yk**4
    e6 = 1.49645 + 13.3988*yk + 5*yk**2

    # return (a5 + b5*xki**2 + c5*xki**4 + d5*xki**6 + e5*xki**8)/(a6+b6*xki**2+c6*xki**4 + d6*xki**6 + e6*xki**8 + xki**10)
    return ((a5 + b5*xk.T**2 + c5*xk.T**4 + d5*xk.T**6 + e5*xk.T**8)/(a6+b6*xk.T**2+c6*xk.T**4 + d6*xk.T**6 + e6*xk.T**8 + xk.T**10)).T


@jit(nopython=True, fastmath=True)
def K4_calc(xk, yk):
    a7 = 1.16028e9*yk - 9.86604e8*yk**3 + 4.56662e8*yk**5 - 1.53575e8*yk**7 + 4.08168e7*yk**9 - 9.69463e6*yk**11 + 1.6841e6 * \
        yk**13 - 320772*yk**15 + 40649.2*yk**17 - 5860.68*yk**19 + 571.687 * \
        yk**21 - 72.9359*yk**23 + 2.35944*yk**25 - 0.56419*yk**27
    b7 = -5.60505e8*yk - 9.85386e8*yk**3 + 8.06985e8*yk**5 - 2.91876e8*yk**7 + 8.64829e7*yk**9 - 7.72359e6*yk**11 + \
        3.59915e6*yk**13 - 234417*yk**15 + 45251.3*yk**17 - 2269.19 * \
        yk**19 - 234.143*yk**21 + 23.0312*yk**23 - 7.33447*yk**25
    c7 = -6.51523e8*yk + 2.47157e8*yk**3 + 2.94262e8*yk**5 - 2.04467e8*yk**7 + 2.29302e7*yk**9 - 2.3818e7 * \
        yk**11 + 576054*yk**13 + 98079.1*yk**15 - 25338.3*yk**17 + \
        1097.77*yk**19 + 97.6203*yk**21 - 44.0068*yk**23
    d7 = -2.63894e8*yk + 2.70167e8*yk**3 - 9.96224e7*yk**5 - 4.15013e7*yk**7 + 3.83112e7*yk**9 + \
        2.2404e6*yk**11 - 303569*yk**13 - 66431.2*yk**15 + \
        8381.97*yk**17 + 228.563*yk**19 - 161.358*yk**21
    e7 = -6.31771e7*yk + 1.40677e8*yk**3 + 5.56965e6*yk**5 + 2.46201e7*yk**7 + 468142 * \
        yk**9 - 1.003e6*yk**11 - 66212.1*yk**13 + \
        23507.6*yk**15 + 296.38*yk**17 - 403.396*yk**19
    f7 = -1.69846e7*yk + 4.07382e6*yk**3 - 3.32896e7*yk**5 - 1.93114e6*yk**7 - \
        934717*yk**9 + 8820.94*yk**11 + 37544.8 * \
        yk**13 + 125.591*yk**15 - 726.113*yk**17
    g7 = -1.23165e6*yk + 7.52883e6*yk**3 - 900010*yk**5 - 186682*yk**7 + \
        79902.5*yk**9 + 37371.9*yk**11 - 260.198*yk**13 - 968.15*yk**15
    h7 = -610622*yk + 86407.6*yk**3 + 153468*yk**5 + 72520.9 * \
        yk**7 + 23137.1*yk**9 - 571.645*yk**11 - 968.15*yk**13
    o7 = -23586.5*yk + 49883.8*yk**3 + 26538.5*yk**5 + \
        8073.15*yk**7 - 575.164*yk**9 - 726.113*yk**11
    p7 = -8009.1*yk + 2198.86*yk**3 + 953.655*yk**5 - 352.467*yk**7 - 403.396*yk**9
    q7 = -622.056*yk - 271.202*yk**3 - 134.792*yk**5 - 161.358*yk**7
    r7 = -77.0535*yk - 29.7896*yk**3 - 44.0068*yk**5
    s7 = -2.92264*yk - 7.33447*yk**3
    t7 = -0.56419*yk
    a8 = 1.02827e9 - 1.5599e9*yk**2 + 1.17022e9*yk**4 - 5.79099e8*yk**6 + 2.11107e8*yk**8 - 6.11148e7*yk**10 + 1.44647e7*yk**12 - \
        2.85721e6*yk**14 + 483737*yk**16 - 70946.1*yk**18 + 9504.65 * \
        yk**20 - 955.194*yk**22 + 126.532*yk**24 - 3.68288*yk**26 + yk**28
    b8 = 1.5599e9 - 2.28855e9*yk**2 + 1.66421e9*yk**4 - 7.53828e8*yk**6 + 2.89676e8*yk**8 - 7.01358e7*yk**10 + 1.39465e7 * \
        yk**12 - 2.84954e6*yk**14 + 498334*yk**16 - 55600*yk**18 + \
        3058.26*yk**20 + 533.254*yk**22 - 40.5117*yk**24 + 14*yk**26
    c8 = 1.17022e9 - 1.66421e9*yk**2 + 1.06002e9*yk**4 - 6.60078e8*yk**6 + 6.33496e7*yk**8 - 4.60396e7*yk**10 + \
        1.4841e7*yk**12 - 1.06352e6*yk**14 - 217801*yk**16 + 48153.3 * \
        yk**18 - 1500.17*yk**20 - 198.876*yk**22 + 91*yk**24
    d8 = 5.79099e8 - 7.53828e8*yk**2 + 6.60078e8*yk**4 + 5.40367e7*yk**6 + 1.99846e8*yk**8 - 6.87656e6 * \
        yk**10 - 6.89002e6*yk**12 + 280428*yk**14 + 161461 * \
        yk**16 - 16493.7*yk**18 - 567.164*yk**20 + 364*yk**22
    e8 = 2.11107e8 - 2.89676e8*yk**2 + 6.33496e7*yk**4 - 1.99846e8*yk**6 - 5.01017e7*yk**8 - \
        5.25722e6*yk**10 + 1.9547e6*yk**12 + 240373*yk**14 - \
        55582*yk**16 - 1012.79*yk**18 + 1001*yk**20
    f8 = 6.11148e7 - 7.01358e7*yk**2 + 4.60396e7*yk**4 - 6.87656e6*yk**6 + 5.25722e6 * \
        yk**8 + 3.04316e6*yk**10 + 123052*yk**12 - \
        106663*yk**14 - 1093.82*yk**16 + 2002*yk**18
    g8 = 1.44647e7 - 1.39465e7*yk**2 + 1.4841e7*yk**4 + 6.89002e6*yk**6 + \
        1.9547e6*yk**8 - 123052*yk**10 - 131337*yk**12 - 486.14*yk**14 + 3003*yk**16
    h8 = 2.85721e6 - 2.84954e6*yk**2 + 1.06352e6*yk**4 + 280428 * \
        yk**6 - 240373*yk**8 - 106663*yk**10 + 486.14*yk**12 + 3432*yk**14
    o8 = 483737 - 498334*yk**2 - 217801*yk**4 - 161461 * \
        yk**6 - 55582*yk**8 + 1093.82*yk**10 + 3003*yk**12
    p8 = 70946.1 - 55600*yk**2 - 48153.3*yk**4 - \
        16493.7*yk**6 + 1012.79*yk**8 + 2002*yk**10
    q8 = 9504.65 - 3058.26*yk**2 - 1500.17*yk**4 + 567.164*yk**6 + 1001*yk**8
    r8 = 955.194 + 533.254*yk**2 + 198.876*yk**4 + 364*yk**6
    s8 = 126.532 + 40.5117*yk**2 + 91*yk**4
    t8 = 3.68288 + 14*yk**2

    return ((np.exp(yk**2) / np.exp(xk.T**2)) * np.cos(2*xk.T*yk) - ((a7 + b7*xk.T**2 + c7*xk.T**4 + d7*xk.T**6 + e7*xk.T**8 + f7*xk.T**10 + g7*xk.T**12 + h7*xk.T**14 + o7*xk.T**16 + p7*xk.T**18 + q7*xk.T**20 + r7*xk.T**22 + s7*xk.T**24 + t7*xk.T**26)/(a8+b8*xk.T**2+c8*(xk.T)**4 + d8*xk.T**6 + e8*xk.T**8 + f8*xk.T**10 + g8*xk.T**12 + h8*xk.T**14 + o8*xk.T**16 + p8*xk.T**18 + q8*xk.T**20 + r8*xk.T**22 + s8*xk.T**24 + t8*xk.T**26 + xk.T**28))).T


# %%
# (5)Voigt function f(ν,p,T)
# 近似式導入 [M.Kuntz 1997 etal.]
@jit(nopython=True, fastmath=True)
def where_func(xk, yk):
    # |x|+yの計算   # K1, K2, ... の計算部分と同じループに入れる
    xy = np.zeros((len(T), len(v)))
    xy = np.abs(xk.T)+yk
    xy = xy.T

    # -y + 0.195|x|-0.176の計算 # K1, K2, ... の計算部分と同じループに入れる
    x_y = np.zeros((len(T), len(v)))
    x_y = -yk + 0.195*np.abs(xk.T) - 0.176
    x_y = x_y.T
    # print('X_y', x_y)

    # Region1〜Region4を満たさない要素番号場所を検索
    C1 = np.where(xy > 15)
    C2 = np.where((xy < 15) & (xy > 5.5))
    C3 = np.where((xy < 5.5) & (0 > x_y))
    C4 = np.where((xy < 5.5) & (0 < x_y))

    return C1, C2, C3, C4


def Voigt(xk, yk, vDk):
    # functionを呼び出して、tupleで拾う
    C1, C2, C3, C4 = where_func(xk, yk)

    K1 = np.zeros((len(T), len(v)))
    K2 = np.zeros((len(T), len(v)))
    K3 = np.zeros((len(T), len(v)))
    K4 = np.zeros((len(T), len(v)))

    # ここがネック[edited by shin]
    K1[C1] = K1_calc(xk, yk)[C1]
    K2[C2] = K2_calc(xk, yk)[C2]
    K3[C3] = K3_calc(xk, yk)[C3]
    K4[C4] = K4_calc(xk, yk)[C4]    # ここがネック[edited by shin]

    # 多項式近似ができたVoigt functionの式 K
    K = K1 + K2 + K3 + K4
    # print("K", K)

    # KK1~KK4,K1~K4を削除
    del K1, K2, K3, K4

    K = (1/vDk)*(K.T)/np.sqrt(np.pi)

    K = (K.T)

    return K


# (6)吸収線強度S(T)
@jit
def Sintensity(Ek, Sijk, vijk):
    S = (Sijk*(Qref/QT)*((np.exp((-c2*Ek)/T))/(np.exp((-c2*Ek)/Tref))) *
         ((1.0-np.exp((-c2*vijk)/T))/(1.0-np.exp((-c2*vijk)/Tref))))  # cm-1/(molecule-1 cm2)
    # print('S', S)

    return S


# (7)吸収係数σ(z, ν)
@jit
def crosssection(Sk, Kk):
    sigma = np.zeros((len(T), len(v)))
    sigma = Sk * Kk.T  # molecule-1 cm2 Voigt functionを使用して計算
    # sigma = sigma.T
    # print('吸収係数', sigma)

    return sigma.T


# (8-1)光学的厚みτ(ν)：シングルライン計算
@jit
def tau_absorption(sigmak):
    tau = np.zeros((len(v)))
    for j in range(len(T)):
        if j == 0:
            tau = 0.5 * sigmak[j, :]*nd[j]*mixCO2*(2e5)
        elif j == len(T)-1:
            tau += 0.5 * sigmak[j, :] * nd[j] * mixCO2 * (2e5)
        else:
            tau += 0.5 * sigmak[j, :]*nd[j] * mixCO2 * (4e5)

# 適当な温度と適当な温度を使って線の広がりを見る
# 光学的厚みの足し合わせは、台形近似をして積分を行っている
# (8-2)：マルチライン増やす(ラインバイライン計算)
    # sumtau = np.sum(tau, axis=0)
    # print('光学的厚み', tau)

    return tau


@ profile
def main():
    # 吸収線の要素番号(k)でループ
    tausum = np.zeros((len(v)))

    for k in range(len(vij)):
        start = time.time()
        vijk = vij[k]
        Sijk = Sij[k]
        gammaselfk = gammaself[k]
        gammaairk = gammaair[k]
        Ek = E[k]
        nairk = nair[k]
        deltaairk = deltaair[k]
        vDk = Doppler(vijk)
        vLk = Lorenz(nairk, gammaairk, gammaselfk)
        # loopstart = time.time()
        xk = Voigt_x(vijk, deltaairk, vDk)
        # if k % 5000 == 0:  # ループにかかる時間を出力(5000回に1度)
        #     print('loop time: ', time.time()-loopstart)
        yk = Voigt_y(vLk, vDk)
        Kk = Voigt(xk, yk, vDk)  # ここがネック[edited by shin]
        Sk = Sintensity(Ek, Sijk, vijk)
        sigmak = crosssection(Sk, Kk)
        tauk = tau_absorption(sigmak)
        print(tauk)

        tausum += tauk
        print('1ループの所要時間: ', time.time()-start)
        print('今なんループ？', k)

    # グラフ作成
    # データ読み込み&定義

    x1 = v
    y1 = tausum

    fig = plt.figure()
    ax = fig.add_subplot(111, title='CO2')
    ax.grid(c='lightgray', zorder=1)
    ax.plot(x1, y1, color='b')
    # ax.set_xlim(4973, 4975)
    ax.set_yscale('log')
    ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # データセーブ
    tau_v = np.stack([v, tausum], 1)
    np.savetxt('4445-5656_0.01step.txt', tau_v, fmt='%.10e')

    # 凡例
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc='lower right', fontsize=14)
    plt.show()

# pressure plot
# %%
if __name__ == '__main__':
    main()

"""

"""
# SZA＝0、EA=0, PA=0, TA=135, TB=80, SurfaceA=0.05, waterI=0, Dust=0
# pressure index 1 :: 50 Pa
path = '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP'+str(i)+'_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat'

ARS_1 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP1_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x1 = ARS_1[:, 0]
ARS_x1 = (1/ARS_x1)*10000
ARS_x1 = ARS_x1[::-1]
ARS_y1 = ARS_1[:, 1]
ARS_y1 = ARS_y1[::-1]

# pressure index 2 :: 150 Pa
ARS_2 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP2_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x2 = ARS_2[:, 0]
ARS_x2 = (1/ARS_x2)*10000
ARS_x2 = ARS_x2[::-1]
ARS_y2 = ARS_2[:, 1]
ARS_y2 = ARS_y2[::-1]

# pressure index 3 :: 180 Pa
ARS_3 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP3_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x3 = ARS_3[:, 0]
ARS_x3 = (1/ARS_x3)*10000
ARS_x3 = ARS_x3[::-1]
ARS_y3 = ARS_3[:, 1]
ARS_y3 = ARS_y3[::-1]

# pressure index 4 :: 215 Pa
ARS_4 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP4_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x4 = ARS_4[:, 0]
ARS_x4 = (1/ARS_x4)*10000
ARS_x4 = ARS_x4[::-1]
ARS_y4 = ARS_4[:, 1]
ARS_y4 = ARS_y4[::-1]

# pressure index 5 :: 257 Pa
ARS_5 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP5_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x5 = ARS_5[:, 0]
ARS_x5 = (1/ARS_x5)*10000
ARS_x5 = ARS_x5[::-1]
ARS_y5 = ARS_5[:, 1]
ARS_y5 = ARS_y5[::-1]

# pressure index 56:: 308 Pa
ARS_6 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP6_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x6 = ARS_6[:, 0]
ARS_x6 = (1/ARS_x6)*10000
ARS_x6 = ARS_x6[::-1]
ARS_y6 = ARS_6[:, 1]
ARS_y6 = ARS_y6[::-1]

# pressure index 7 :: 369 Pa
ARS_7 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP7_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x7 = ARS_7[:, 0]
ARS_x7 = (1/ARS_x7)*10000
ARS_x7 = ARS_x7[::-1]
ARS_y7 = ARS_7[:, 1]
ARS_y7 = ARS_y7[::-1]

# pressure index 8 :: 442 Pa
ARS_8 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP8_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x8 = ARS_8[:, 0]
ARS_x8 = (1/ARS_x8)*10000
ARS_x8 = ARS_x8[::-1]
ARS_y8 = ARS_8[:, 1]
ARS_y8 = ARS_y8[::-1]

# pressure index 9 :: 529 Pa
ARS_9 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP9_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x9 = ARS_9[:, 0]
ARS_x9 = (1/ARS_x9)*10000
ARS_x9 = ARS_x9[::-1]
ARS_y9 = ARS_9[:, 1]
ARS_y9 = ARS_y9[::-1]

# pressure index 10 :: 633 Pa
ARS_10 = np.loadtxt(
    '/Users/nyonn/Desktop/pythoncode/ARS_calc/SP10_TA1_TB1_SZA1_EA1_PA1_Dust1_WaterI1_SurfaceA1_rad.dat')
ARS_x10 = ARS_10[:, 0]
ARS_x10 = (1/ARS_x10)*10000
ARS_x10 = ARS_x10[::-1]
ARS_y10 = ARS_10[:, 1]
ARS_y10 = ARS_y10[::-1]

# pressure index 11 :: 758 Pa
# pressure index 12 :: 907 Pa
# pressure index 13 :: 1096 Pa
# pressure index 14 :: 1300 Pa
# pressure index 15 :: 1500 Pa

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, title='CO2 absorption')
ax.grid(c='lightgray', zorder=1)

ax.plot(ARS_x1, ARS_y1, color='blue', label="Pa = 50", zorder=1)
ax.plot(ARS_x2, ARS_y2, color='red', label="Pa = 150", zorder=2)
ax.plot(ARS_x3, ARS_y3, color='green', label="Pa = 180", zorder=3)
ax.plot(ARS_x4, ARS_y4, color='orange', label="Pa = 215", zorder=4)
ax.plot(ARS_x5, ARS_y5, color='pink', label="Pa = 257", zorder=5)
ax.plot(ARS_x6, ARS_y6, color='purple', label="Pa = 308", zorder=6)
ax.plot(ARS_x7, ARS_y7, color='skyblue', label="Pa = 369", zorder=7)
ax.plot(ARS_x8, ARS_y8, color='yellow', label="Pa = 442", zorder=8)
ax.plot(ARS_x9, ARS_y9, color='grey', label="Pa = 529", zorder=8)
ax.plot(ARS_x10, ARS_y10, color='black', label="Pa = 529", zorder=8)

h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=8)
plt.show()
"""
