from sys import float_repr_style
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
"""
both_x  = np.array([2.25, 2.33, 2.41, 2.49, 2.57, 2.81, 3.05, 3.13, 3.21, 3.29, 3.37, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.93, 4.00, 4.10, 4.25, 4.33, 4.41])

both    = np.array([[32, 33, 38, 28, 30, 38, 33, 43, 43, 24, 40],
                    [19, 23, 27, 23, 21, 19, 23, 21, 27, 22, 19],
                    [22, 16, 21, 23, 19, 15, 25, 19, 19, 18, 21],
                    [17, 14, 20, 19, 23, 20, 21, 17, 18, 11, 21],
                    [33, 36, 38, 35, 30, 30, 33, 25, 28, 25, 32],
                    [75, 42, 57, 51, 58, 67, 67, 66, 62, 54, 66],
                    [28, 27, 15, 29, 20, 26, 21, 33, 21, 30, 24],
                    [12, 11, 16, 15, 18, 19, 14, 5,  20, 14, 12],
                    [22, 17, 19, 29, 22, 29, 22, 18, 20, 22, 25],
                    [42, 39, 47, 45, 64, 57, 36, 44, 35, 38, 45],
                    [67, 50, 47, 55, 49, 51, 52, 61, 64, 59, 74],
                    [63, 92, 83, 65, 61, 71, 90, 89, 91, 80, 78],
                    [64, 64, 69, 75, 73, 55, 73, 83, 56, 64, 68],
                    [45, 61, 44, 54, 52, 44, 52, 48, 58, 51, 47],
                    [27, 33, 38, 25, 24, 29, 34, 31, 35, 30, 19],
                    [13, 12, 14, 13, 12, 12, 12, 15, 17, 17, 14],
                    [18, 8,  20, 14, 12, 14, 13, 14, 11, 16, 16],
                    [29, 33, 32, 28, 20, 21, 23, 25, 28, 23, 20],
                    [39, 45, 31, 38, 50, 49, 37, 48, 40, 39, 48],
                    [53, 55, 57, 70, 58, 55, 62, 75, 65, 61, 63],
                    [55, 56, 47, 58, 48, 47, 53, 45, 50, 61, 46],
                    [33, 27, 37, 28, 36, 32, 22, 32, 28, 41, 39],
                    [18, 19, 15, 13, 15, 28, 20, 16, 20, 11, 26]])"""


both_x  = np.array([2.49, 2.57, 2.81, 3.05, 3.13, 3.21, 3.29, 3.37, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.93, 4.00, 4.10, 4.25, 4.33, 4.41])

both    = np.array([[17, 14, 20, 19, 23, 20, 21, 17, 18, 11, 21],
                    [33, 36, 38, 35, 30, 30, 33, 25, 28, 25, 32],
                    [75, 42, 57, 51, 58, 67, 67, 66, 62, 54, 66],
                    [28, 27, 15, 29, 20, 26, 21, 33, 21, 30, 24],
                    [12, 11, 16, 15, 18, 19, 14, 5,  20, 14, 12],
                    [22, 17, 19, 29, 22, 29, 22, 18, 20, 22, 25],
                    [42, 39, 47, 45, 64, 57, 36, 44, 35, 38, 45],
                    [67, 50, 47, 55, 49, 51, 52, 61, 64, 59, 74],
                    [63, 92, 83, 65, 61, 71, 90, 89, 91, 80, 78],
                    [64, 64, 69, 75, 73, 55, 73, 83, 56, 64, 68],
                    [45, 61, 44, 54, 52, 44, 52, 48, 58, 51, 47],
                    [27, 33, 38, 25, 24, 29, 34, 31, 35, 30, 19],
                    [13, 12, 14, 13, 12, 12, 12, 15, 17, 17, 14],
                    [18, 8,  20, 14, 12, 14, 13, 14, 11, 16, 16],
                    [29, 33, 32, 28, 20, 21, 23, 25, 28, 23, 20],
                    [39, 45, 31, 38, 50, 49, 37, 48, 40, 39, 48],
                    [53, 55, 57, 70, 58, 55, 62, 75, 65, 61, 63],
                    [55, 56, 47, 58, 48, 47, 53, 45, 50, 61, 46],
                    [33, 27, 37, 28, 36, 32, 22, 32, 28, 41, 39],
                    [18, 19, 15, 13, 15, 28, 20, 16, 20, 11, 26]])


blkd_x  = np.array([0, 1, 2.3, 3.26, 3.5, 4.5, 7, 8])

blkd    = np.array([[6, 14, 5, 5, 5, 17, 11, 5, 11, 4, 10],
                    [10, 10, 13, 9, 7, 10, 4, 9, 16, 10, 6],
                    [7, 11, 5, 10, 8, 13, 7, 4, 4, 8, 7],
                    [7, 11, 8, 7, 14, 10, 7, 7, 13, 8, 6],
                    [8, 7, 7, 12, 14, 10, 14, 8, 8, 11, 8],
                    [7, 10, 12, 9, 13, 14, 10, 16, 10, 7, 9],
                    [12, 3, 8, 9, 7, 8, 11, 8, 9, 9, 3],
                    [10, 12, 13, 10, 12, 13, 10, 12, 11, 23, 9]])


near_x  = np.array([1, 3, 3.5, 4.5, 7, 8])

near    = np.array([[6.8, 9, 5, 3, 11, 10, 8, 10, 9, 10, 10],
                    [19, 18, 23, 27, 14, 20, 25, 23, 23, 16, 21],
                    [15, 30, 21, 25, 27, 27, 29, 24, 30, 30, 19],
                    [22, 30, 23, 35, 17, 30, 24, 30, 29, 21, 28],
                    [10, 12, 10, 15, 15, 9, 16, 9, 16, 8, 11],
                    [15, 13, 15, 19, 17, 12, 6, 15, 21, 18, 16]])


far__x  = np.array([0, 1.5, 3, 3.5, 4, 6, 7])                    

far_    = np.array([[5, 11, 7, 6, 12, 19, 11, 8, 3, 14, 8],
                    [13, 11, 12, 18, 19, 18, 18, 20, 18, 21, 15],
                    [20, 17, 35, 21, 16, 16, 18, 22, 26, 19, 21],
                    [21, 19, 24, 19, 19, 22, 21, 19, 23, 17, 22],
                    [27, 13, 15, 19, 16, 15, 22, 15, 25, 20, 13],
                    [13, 16, 7, 9, 12, 17, 11, 15, 14, 9, 13],
                    [9, 9, 9, 12, 7, 7, 11, 9, 10, 7, 4]])

both_avg = np.array([np.sum(both[x])/len(both[x]) for x in range(len(both))])
blkd_avg = np.array([np.sum(blkd[x])/len(blkd[x]) for x in range(len(blkd))])
far__avg = np.array([np.sum(far_[x])/len(far_[x]) for x in range(len(far_))])
near_avg = np.array([np.sum(near[x])/len(near[x]) for x in range(len(near))])

dark_avg_scalar = np.sum(blkd_avg)/len(blkd_avg)
dark_avg = dark_avg_scalar * np.ones(len(blkd_avg))

"""Error for averaging dark counts"""
total_dark = blkd.sum()
num_of_interval = np.array(np.shape(blkd)).sum()
err_avg_count_rate_scalar = total_dark ** (-1/2) / num_of_interval * dark_avg_scalar
err_avg_count_rate = total_dark ** (-1/2) / num_of_interval * dark_avg

print(err_avg_count_rate)


"""Interpolation for better curve"""
f = interp1d(both_x, both_avg, kind='cubic') 
f1 = interp1d(near_x, near_avg, kind='cubic')
f2 = interp1d(far__x, far__avg, kind='cubic')
f3 = interp1d(blkd_x, blkd_avg, kind='cubic')

both_xnew = np.linspace(2.25, 4.405, num=101, endpoint=True)
both_xnew = np.linspace(2.49, 4.405, num=101, endpoint=True)
far__xnew = np.linspace(0, 7, num=101, endpoint=True)
near_xnew = np.linspace(1, 8, num=101, endpoint=True)
blkd_xnew = np.linspace(0, 8, num=101, endpoint=True)

both_errb = np.array([np.sqrt(np.sum(both[x]))/len(both[x])for x in range(len(both))])
far__errb = np.array([np.sqrt(np.sum(far_[x]))/len(far_[x])for x in range(len(far_))])
near_errb = np.array([np.sqrt(np.sum(near[x]))/len(near[x])for x in range(len(near))])
blkd_errb = np.array([np.sqrt(np.sum(blkd[x]))/len(blkd[x])for x in range(len(blkd))])

plt.figure(figsize=(12, 5))


"""Plotting rqaw data error bars (order matters)"""
"""
plt.errorbar(both_x, both_avg, yerr=both_errb,  fmt=".", label = "interference model")
plt.errorbar(near_x, near_avg, yerr=near_errb,  fmt=".", label = "diffraction from near slit")
plt.errorbar(far__x, far__avg, yerr=far__errb,  fmt=".", label = "diffraction from far slit")
# plt.errorbar(blkd_x, blkd_avg, yerr=blkd_errb,  fmt=".", label = "PMT shuttered")
# plt.errorbar(blkd_x, dark_avg, yerr=err_avg_count_rate,  fmt=".", label = "PMT shuttered (averaged)")

both_ynew, far__ynew, near_ynew, blkd_ynew = f(both_xnew), f2(far__xnew), f1(near_xnew), f3(blkd_xnew)

plt.plot(both_xnew, both_ynew, color = "#1f77b4")
plt.plot(far__xnew, far__ynew, color = "#2ca02c")
plt.plot(near_xnew, near_ynew, color = "#ff7f0e")
# plt.plot(blkd_xnew, blkd_ynew, color = "#d62728")
plt.plot(blkd_x, dark_avg, color = "black",linestyle='dashed')

plt.fill_between(blkd_x, dark_avg - err_avg_count_rate, dark_avg + err_avg_count_rate, color='gray', alpha=0.9, label = "PMT shuttered (averaged)")
"""

"""Corrected signal"""
both_errb_corr = np.sqrt(both_errb**2 + err_avg_count_rate_scalar**2)
far__errb_corr = np.sqrt(far__errb**2 + err_avg_count_rate_scalar**2)
near_errb_corr = np.sqrt(near_errb**2 + err_avg_count_rate_scalar**2)

plt.errorbar(both_x, both_avg - dark_avg_scalar, yerr=both_errb_corr,  fmt=".", label = "interference model")
plt.errorbar(near_x, near_avg - dark_avg_scalar, yerr=near_errb_corr,  fmt=".", label = "diffraction from far slit")
plt.errorbar(far__x, far__avg - dark_avg_scalar, yerr=far__errb_corr,  fmt=".", label = "diffraction from near slit")

both_ynew, far__ynew, near_ynew, blkd_ynew = f(both_xnew), f2(far__xnew), f1(near_xnew), f3(blkd_xnew)

plt.plot(both_xnew, both_ynew - dark_avg_scalar, color = "#1f77b4")
plt.plot(far__xnew, far__ynew - dark_avg_scalar, color = "#2ca02c")
plt.plot(near_xnew, near_ynew - dark_avg_scalar, color = "#ff7f0e")
# plt.plot(blkd_xnew, blkd_ynew, color = "#d62728", label = "dark")
# plt.plot(blkd_x, dark_avg, color = "black",linestyle='dashed', label = "averaged dark counts")

plt.xlim(1, 6)
plt.ylim(0,75)

plt.title("Two slit interference signal obtained with bulb illumination (corrected)")
plt.xlabel(r"Double-slit position ($mm$)")
plt.ylabel(r"Photon count rate from PCIT ($count/s$)")
plt.legend(loc="best")

plt.show(block = False)
plt.savefig("plots/pdf/interference_pattern/final_inter.pdf")


