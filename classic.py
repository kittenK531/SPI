import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.array([2.25, 2.67, 3.04, 3.50, 3.87, 4.29, 4.68])
y = np.array([0.034, 1.622, 0.039, 2.030, 0.106, 1.733, 0.180])

err = np.array([0.001, 0.002, 0.001, 0.003, 0.002, 0.001, 0.002])

f = interp1d(x, y, kind='quadratic') 

xn = np.linspace(2.25, 4.68, num=101, endpoint=True)

plt.figure(figsize=(12,5))

plt.errorbar(x, y, yerr=err, fmt=".", label = "fluctuations")
plt.plot(xn, f(xn))
plt.title('Double slits interference profile obtained from diode-laser illumination')
plt.xlabel(r'Detector slit position ($mm$)')
plt.ylabel(r'Signal from photodiode I-to-V converter ($V$)')

plt.legend(loc='best')
plt.show(block=False)
plt.savefig('plots/pdf/classical.pdf')