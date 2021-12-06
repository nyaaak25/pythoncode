import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Tau_txt = np.loadtxt('τ_4973-4975.dat')
Tau=Tau_txt[:,1]
v=Tau_txt[:,0]
theta = 88.9

I0 = np.exp(-Tau/np.cos(theta))
Iobs = I0 * np.exp(-Tau)

x1 = v
y1 = Iobs

fig = plt.figure()
ax = fig.add_subplot(111, title='CO2')
ax.grid(c='lightgray', zorder=1)
ax.plot(x1, y1, color='b')
ax.set_xlim(4973,4975)
#ax.set_yscale('log') 
ax.set_xlabel('Wavenumber [$cm^{-1}$]', fontsize=14)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

#凡例
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, loc='lower right', fontsize=14)
plt.show()