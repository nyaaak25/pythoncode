
# %%
# This script is used to scale the dust density to the desired value
# 2024.04.09 created by AKira Kazama

import numpy as np
import matplotlib.pyplot as plt

HD_base = np.loadtxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/1dd.hc")
altitude = HD_base[:,0]
density = HD_base[:,1]
new_density = np.zeros(len(density))

inc_density = 1

fig = plt.figure()
plt.plot(density, altitude, label="Original", color="black", linestyle="solid")
plt.ylabel("Altitude [km]")
plt.xlabel("Density [km-1]")
plt.title("Density profile of dust")

for i in range(len(density)):
    new_density = density
    new_density[i] = density[i] + inc_density
    plt.plot(new_density, altitude, label="Scaled", linestyle="dashed")
    new_HD = np.column_stack((altitude, new_density))
    np.savetxt("/Users/nyonn/Desktop/pythoncode/Dust/SH/HDD/" +str(i)+ "d.hc", new_HD)
# %%
