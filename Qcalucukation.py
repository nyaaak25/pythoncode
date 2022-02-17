import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

# データの読み込みをする。T_nyonは温度データ。Q_CO2はHITRANからのQ(Tデータ

T_nyon = pd.read_fwf(
    'Temp_pres_Kazama.dat', header=None)
Q_CO2 = pd.read_csv(
    'QCO2.csv', header=None, delimiter=',')

T_nyon.set_axis(['height', 'Tempreture', 'Pressure',
                '200', '6'], axis=1, inplace=True)
Q_CO2.set_axis(['Tempreture', 'Q'], axis=1, inplace=True)

# 温度データを整数に
T_nyon = T_nyon.astype({'Tempreture': 'int64'})

# データをがっちゃんこ
QplusT = pd.merge(T_nyon, Q_CO2, on='Tempreture', how='inner')
print(QplusT)
# データの指定
QplusT.to_csv('CO2_Q.csv', columns=['Tempreture', 'Q'], index=False)
