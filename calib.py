import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Discrimiator = np.array([700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900])

both    = np.array([[1, 1, 0, 2, 1, 2, 1, 0, 2, 1],
                    [2, 3, 1, 1, 3, 6, 2, 1, 1, 3],
                    [8, 4, 3, 6, 4, 4, 7, 6, 3, 5],
                    [8, 6, 5, 12, 14, 11, 7, 8, 10, 14],
                    [20, 24, 15, 15, 14, 18, 22, 11, 16, 12],
                    [15, 22, 20, 25, 19, 20, 26, 17, 28, 36],
                    [33, 35, 30, 49, 37, 25, 34, 41, 38, 31],
                    [52, 45, 45, 47, 46, 49, 58, 46, 36, 48],
                    [82, 62, 64, 83, 66, 65, 60, 56, 55, 76],
                    [74, 75, 79, 71, 85, 72, 61, 75, 79, 61],
                    [82, 83, 61, 94, 88, 63, 88, 86, 91, 90],
                    [87, 109, 90, 81, 104, 86, 112, 106, 93, 119],
                    [112, 108, 107, 103, 82, 94, 92, 95, 110, 110],
                    [100, 115, 114, 115, 119, 115, 125, 131, 116, 113],
                    [129, 127, 123, 102, 122, 114, 136, 121, 96, 128],
                    [131, 124, 133, 130, 118, 144, 127, 145, 150, 118],
                    [137, 128, 131, 146, 164, 142, 137, 158, 170, 151],
                    [189, 153, 160, 149, 146, 155, 157, 183, 171, 162],
                    [177, 166, 140, 179, 157, 142, 171, 175, 163, 171],
                    [161, 199, 159, 145, 175, 166, 150, 185, 174, 185],
                    [184, 187, 167, 164, 188, 191, 194, 186, 177, 176]])

dark    = np.array([[1, 0, 1, 2, 1, 0, 1, 1, 1, 3],
                    [4, 1, 2, 1, 3, 3, 4, 1, 0, 2],
                    [2, 4, 6, 1, 4, 3, 1, 1, 1, 3],
                    [1, 2, 3, 3, 1, 2, 3, 2, 4, 2],
                    [1, 3, 4, 3, 3, 2, 3, 4, 3, 1],
                    [6, 6, 3, 4, 2, 3, 6, 4, 7, 6],
                    [7, 7, 4, 7, 2, 3, 4, 7, 3, 7],
                    [10, 7, 4, 7, 4, 13, 6, 4, 12, 9],
                    [13, 10, 9, 11, 10, 6, 8, 9, 11, 8],
                    [10, 12, 4, 12, 7, 12, 7, 20, 19, 13],
                    [16, 19, 15, 15, 11, 16, 12, 24, 11, 11],
                    [30, 14, 22, 19, 13, 13, 20, 15, 18, 21],
                    [21, 20, 28, 17, 22, 25, 30, 18, 23, 28],
                    [25, 31, 30, 30, 27, 25, 27, 22, 25, 28],
                    [27, 28, 25, 46, 40, 22, 30, 40, 37, 38],
                    [34, 42, 43, 55, 45, 41, 39, 44, 35, 31],
                    [50, 51, 41, 50, 53, 46, 30, 56, 28, 49],
                    [43, 61, 53, 49, 51, 48, 46, 59, 56, 50],
                    [64, 47, 53, 50, 65, 51, 69, 55, 63, 66],
                    [61, 54, 64, 76, 51, 51, 64, 60, 56, 55],
                    [77, 69, 73, 69, 80, 65, 72, 68, 82, 67]])


clse   = np.array([[1, 1, 1, 0, 0, 0, 0, 3, 1, 1],
                    [0, 0, 1, 0, 1, 2, 0, 0, 1, 0],
                    [1, 0, 3, 1, 1, 2, 1, 0, 1, 1],
                    [3, 1, 1, 1, 1, 2, 1, 1, 2, 4],
                    [5, 2, 2, 4, 5, 2, 2, 2, 2, 1],
                    [5, 1, 4, 8, 3, 5, 6, 2, 5, 3],
                    [5, 6, 4, 5, 6, 2, 2, 5, 4, 7],
                    [7, 3, 3, 9, 8, 3, 8, 7, 7, 7],
                    [7, 4, 8, 11, 9, 7, 10, 6, 4, 10],
                    [15, 10, 10, 6, 6, 6, 10, 17, 12, 16],
                    [22, 17, 16, 10, 12, 14, 22, 15, 25, 16],
                    [27, 14, 13, 24, 13, 17, 10, 16, 20, 19],
                    [24, 23, 21, 21, 26, 20, 22, 28, 13, 38],
                    [27, 26, 22, 23, 24, 25, 28, 26, 36, 29],
                    [33, 27, 25, 24, 43, 40, 34, 23, 39, 19],
                    [40, 30, 41, 25, 34, 45, 49, 38, 37, 28],
                    [38, 34, 43, 63, 38, 37, 42, 44, 42, 48],
                    [48, 49, 58, 54, 49, 45, 37, 49, 52, 47],
                    [61, 61, 50, 57, 58, 59, 66, 48, 53, 66],
                    [69, 65, 76, 53, 71, 60, 40, 39, 64, 69],
                    [70, 61, 56, 51, 66, 70, 75, 69, 66, 57]])

both_avg = np.array([both[x].sum()/len(both[x]) for x in range(len(both))])
dark_avg = np.array([dark[x].sum()/len(dark[x]) for x in range(len(dark))])
clse_avg = np.array([clse[x].sum()/len(clse[x]) for x in range(len(clse))])

both_ebr = np.array([np.sqrt(both[x].sum())/len(both[x]) for x in range(len(both))])
dark_ebr = np.array([np.sqrt(dark[x].sum())/len(dark[x]) for x in range(len(dark))])
clse_ebr = np.array([np.sqrt(clse[x].sum())/len(clse[x]) for x in range(len(clse))])

f = interp1d(Discrimiator, both_avg, kind='cubic')
f2 = interp1d(Discrimiator, dark_avg, kind='cubic')
f3 = interp1d(Discrimiator, clse_avg, kind='cubic')

plt.figure(figsize = (12, 10))

both_xnew = np.linspace(700, 900, num=301, endpoint=True)


# plt.plot(both_xnew, f3(both_xnew), color ="#2ca02c")
# plt.plot(both_xnew, f2(both_xnew), color ="#ff7f0e")

bg1__err = np.sqrt(((both_ebr**2 + clse_ebr**2)/(both_avg - clse_avg))**2 + (clse_ebr/clse_avg)**2) * (both_avg - clse_avg)/clse_avg
bg2__err = np.sqrt(((both_ebr**2 + dark_ebr**2)/(both_avg - dark_avg +0.0001))**2 + (dark_ebr/dark_avg)**2) * (both_avg - dark_avg)/dark_avg

"""plt.plot(both_xnew, (f(both_xnew) - f3(both_xnew))/f3(both_xnew), color = "#1f77b4")
plt.errorbar(Discrimiator, (both_avg - clse_avg)/clse_avg, yerr=bg1__err, fmt='.', label = "SNR with bg as PMT shuttered")"""
plt.plot(both_xnew, (f(both_xnew) - f2(both_xnew))/f2(both_xnew), color = "#ff7f0e")
plt.errorbar(Discrimiator, (both_avg - dark_avg)/dark_avg, yerr=bg2__err, fmt='.', label = "SNR with bg as slits blocked")
# plt.errorbar(Discrimiator, dark_avg, yerr=dark_ebr, fmt='.', label = "Slits Blocked")
# plt.errorbar(Discrimiator, clse_avg, yerr=clse_ebr, fmt='.', label = "PMT shuttered")

plt.title("SNR as a function of PMT bias voltage with slits blocked signal as background")
plt.ylabel(r"Signal to Noise ratio [SNR]")
plt.xlabel(r"Photomultiplier tube bias setting ($V$)")
plt.grid(axis='y')
# plt.yscale('log')
plt.legend(loc='best')
plt.show(block=False)
plt.savefig("plots/pdf/Blocked_SNR.pdf")