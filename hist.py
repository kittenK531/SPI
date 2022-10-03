import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

print([item.get_text() for item in ax.get_xticklabels()])

fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = -1
labels[4] = 0
labels[5] = +1


zeroth = 3.454 * np.ones(78)
posone = 4.1 * np.ones(13)
negone = 2.81 * np.ones(14)

cent = np.append(zeroth, posone)
cent = np.append(cent, negone)

near = np.append( 2.853 * np.ones(22),  3.4648* np.ones(25))
near = np.append(near, 4.1435*np.ones(26))

far_ = np.append( 3.371* np.ones(20), 2.766 * np.ones(21))
far_ = np.append(far_,  4.056* np.ones(18))

ax.grid(axis='y')

ax.errorbar(np.array([2.786, 2.83, 2.873, 3.391, 3.434, 3.4748, 4.036, 4.08, 4.1235]), np.array([21, 14, 22, 20, 78, 25, 18, 13, 26]), yerr=np.array([1.38, 1.135, 1.28, 1.37, 2.67, 1.51, 1.06, 1.12, 1.55]), fmt='none', label="errors")

ax.hist(cent, bins=30, label = "Central Maximum", color="#17becf")
ax.hist(near, bins=30, label = "Near slit diffraction", color = "#ff7f0e")
ax.hist(far_, bins=30, label = "Far Slit diffraction", color="#2ca02c")
ax.set_xticklabels(['',r'$-1$','','','0','','',r'$+1$',''])
ax.legend(loc='best')
ax.set_axisbelow(True)
plt.title("Photo count rate as a function of detection position")
plt.xlabel(r"Double-slit position in terms of order of period ($T/2$)")
plt.ylabel(r"Photon count rate from PCIT ($count/s$)")
plt.show(block=False)
plt.savefig("plots/pdf/histogram/quantum_paradox.pdf")

##ERROR bar
