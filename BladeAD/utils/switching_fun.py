import numpy as np
import csdl_alpha as csdl
import matplotlib.pyplot as plt


def sigmoid(x, critical_angle):
    
    s = 1 / (1 + csdl.exp(3 * (x-critical_angle)))

    return s

if __name__ == "__main__":
    x = np.linspace(0, 17, 100)
    F = 0.8 
    s = sigmoid(x, critical_angle=12)
    print(1 + (F-1) *sigmoid(8, critical_angle=12))
    print(1 + (F-1) *sigmoid(20, critical_angle=12))
    plt.plot(x, s)
    plt.grid()
    plt.show()
