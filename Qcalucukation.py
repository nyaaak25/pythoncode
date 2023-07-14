# %%
import pandas as pd
import glob
import os

# データの読み込みをする。T_nyonは温度データ。Q_CO2はHITRANからのQ(Tデータ


def filesearch(dir):
    # 指定されたディレクトリ内の全てのファイルを取得
    # path_list = glob.glob("LookUpTable_HTP/*.txt")
    path_list = glob.glob("ret_dust_2.txt")
    name_list = []                          # ファイル名の空リストを定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in path_list:
        file = os.path.basename(i)          # 拡張子ありファイル名を取得
        name, ext = os.path.splitext(file)  # 拡張子なしファイル名と拡張子を取得
        name_list.append(name)              # 拡張子なしファイル名をリスト化
    return path_list, name_list


# ファイル情報を取得する関数を実行
path_list, name_list = filesearch('dir')


for i in range(len(name_list)):
    T_nyon = pd.read_fwf(
        str(path_list[i]), header=None)
    Q_CO2 = pd.read_csv(
        'QCO2.csv', header=None, delimiter=',')

    T_nyon.set_axis(['height', 'Pressure', 'Tempreture'], axis=1, inplace=True)
    Q_CO2.set_axis(['Tempreture', 'Q'], axis=1, inplace=True)

    # 温度データを整数に
    T_nyon = T_nyon.astype({'Tempreture': 'int64'})

    # データをがっちゃんこ
    QplusT = pd.merge(T_nyon, Q_CO2, on='Tempreture', how='inner')
    # print(QplusT)
    # %%
    # データの指定
    # QplusT.to_csv('LookUpTable_Q/'+str(name_list[i])+'.csv',
    #              columns=['Tempreture', 'Q'], index=False)
    QplusT.to_csv(str(name_list[i])+'.csv',
                  columns=['Tempreture', 'Q'], index=False)


# %%
