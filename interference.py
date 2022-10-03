import numpy as np
import matplotlib.pyplot as plt

d = 500 #0.406
a = 100# 2.0
hbar = 1

x = np.linspace(-np.pi/4,np.pi/4, num=100)

def G(theta):

    phi =  np.pi * d * np.sin(theta) 
    alpha = np.pi * a * np.sin(theta) 

    return np.cos(phi)**2 * np.sinc(alpha)**2 

def P(theta):

    return  np.cos(theta/2)**2 * np.sinc(theta)**2 

d = 0.406

y = np.sinc(np.pi * d * x /(1000 * 625e-6)) **2 

def z(n):

    return np.cos(np.pi * n * d * x / 625e-6) **2

plt.plot(x, y)
for i in [7]:# 3.1 3 peaks
    plt.plot(x, z(i)*y, label = f"{i}")

plt.legend()
plt.show(block = False)
plt.savefig("tist.pdf")