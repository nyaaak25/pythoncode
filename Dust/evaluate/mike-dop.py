
# ------------------------------------------
# Mike Wolff+2009のDOPを読み込むプログラム
# fits fileを読み込んで、そのヘッダーを確認
# そして、そのヘッダーからDust properties情報を取得

# 2024/10/17 Thu 11:27 by Akira Kazama
# ------------------------------------------
# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# %%
# fits fileのヘッダーを確認する
# FITSファイルを開く
hdul = fits.open('/Users/nyonn/Desktop/pythoncode/Dust/evaluate/mike/mars045i_all_area_s0780.fits')

# ファイル情報を表示する
hdul.info()

# データにアクセスする (例: 1番目のHDUのデータ)
data = hdul[0].data
header = hdul[0].header

# データの表示
print(data)

# ファイルを閉じる
hdul.close()

# %%
# 必要なデータを取得する
# mike wolffのヘッダー情報は以下の通り

"""
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU      14   (1,)   uint8   
  1  FORW          1 ImageHDU        16   (3, 228, 25)   float32   
  2  PMOM          1 ImageHDU        15   (228, 25, 160)   float32   
  3  PHSFN         1 ImageHDU        15   (228, 25, 498)   float32   
  4  EXPANSION     1 ImageHDU        15   (228, 25, 498)   float32   
  5  PARTICLE_SIZES    1 ImageHDU        11   (25,)   float32   
  6  WAVELENGTHS    1 ImageHDU        11   (228,)   float32   
  7  SCATTERING_ANGLE    1 ImageHDU        11   (498,)   float32   
[32]
"""

# この中から必要なデータを取得する
fits_file = '/Users/nyonn/Desktop/pythoncode/Dust/evaluate/mike/mars045i_all_area_s0780.fits'

forw = fits.getdata(fits_file, ext=1)

ssa = fits.getdata(fits_file, ext=7)
phase_function = fits.getdata(fits_file, ext=3)
particle_size = fits.getdata(fits_file, ext=5)
wav = fits.getdata(fits_file, ext=6)


# %%
