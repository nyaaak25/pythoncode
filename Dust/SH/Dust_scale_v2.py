
# %%
# --------------------------------------------------
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama

# 2024.10.15 updated by AKira Kazama
# DOP_conv_v1.pyで作成されたreferenceの数密度データを使って、各層において数密度を増加させる
# outputとして出てくるファイルは、各層において増加された数密度のデータが保存されている
# このデータは、/Dust/SH/data-altitude/input/input-hd/に保存される

# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ------- set a parameter -------
filename = "277_1d"  # 波長+何倍かを示している。1なら等倍、2なら10倍, 3なら100倍

# ------- calculate  -------
HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/input-hc/" +filename+ ".hc")
altitude = HD_base[:,0]
density = HD_base[:,1]
new_density = np.zeros(len(density))

inc_density = density[0] * 0.1

# for test
#inc_density = 0.1

fig = plt.figure()
plt.plot(density, altitude, label="Original", color="black", linestyle="solid")
plt.ylabel("Altitude [km]")
plt.xlabel("Density [km-1]")
plt.title("Density profile of dust")

for i in range(len(density)):
    new_density = density
    new_density[i] = density[i] + inc_density

    # for test
    #new_density[i] = density[i] + (density[i] * inc_density)

    plt.plot(new_density, altitude, label="Scaled", linestyle="dashed")
    new_HD = np.column_stack((altitude, new_density))
    np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/input-hd/" +filename+ "/" +str(i)+ "d.hc", new_HD)

    # for test
    #np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/data-altitude/input/input-hd/" +filename+ "_test/" +str(i)+ "d.hc", new_HD)
# %%
