@author AKira Kazama
contact: kazama@pparc.gp.tohoku.ac.jp

・　Mars_tau_kazama_cutoff.py
これは吸収の光学的厚みを計算するプログラム。
現在(2022-02-17)は、look-up-table作成途中である1つの温度・圧力プロファイルのみで計算をさせている。

・　Intensity.py
これは導出された光学的厚みからOMEGAの波長分解能にまで落とすまでを行うプログラムになる予定。
現在(2022/02/17)は、装置関数を掛け合わされる前の放射強度導出段階である。

・　tau_plot.py
これは色々なplotをするファイル。絵がかける。

・　Hitrandata.py
Input用のHitran吸収線データセットを作成するプログラム。これを動かすために「hapi.py」が必要

・　nyooon.py
思考整理に使うファイルなのであまり気にしないでください。。