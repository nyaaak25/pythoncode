# %%
# -*- coding: utf-8 -*-
"""
HITRAN dataを作成
created on Mon Dec 6 10:44:00 2021
@author : A. Kazama
"""
#hapi.pyファイルを読み込んで、HITRAN line-by-lineからパラメータを持ってくる作業
from hapi import *
import numpy as np

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
vij,Sij,E,gammaair,gammaself,nair,deltaair = getColumns(name, ['nu', 'sw','elower','gamma_air','gamma_self','n_air','delta_air']) 
print('nu=',vij,'Sij=',Sij,'γself=',gammaself,'γair=',gammaair,'E"=',E,'nair=',nair,'δair=',deltaair)

#HITRANから引っ張ってきた吸収線データをテキストファイルに落とし込む
savearray = np.array([vij,Sij,gammaself,gammaair,E,nair,deltaair])
np.savetxt('4545-5556_hitrandata.txt',savearray.T)

#Hitrandata = np.loadtxt('4971-4976_hitrandata.txt')
#vij = Hitrandata[:,0]
#Sij = Hitrandata[:,1]
#gammaair = Hitrandata[:,3]
#gammaself = Hitrandata[:,2]
#E = Hitrandata[:,4]
#nair = Hitrandata[:,5]
#δair = Hitrandata[:,6]

# %%
