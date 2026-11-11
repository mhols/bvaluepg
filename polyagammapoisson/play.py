__doc__="""module to play and try things of interest for the project"""

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def phi(t):
    return np.log(1+np.exp(t))


def plot1():
    t = np.linspace(-10,20, 10000)
    plt.figure()

    for n in range(1, 10):
        ph = phi(t)**n * sigmoid(-t)
        ph /= np.sum(ph)
        plt.plot( t, ph, label=f'{n}')
    plt.legend()


if __name__=='__main__':
    plot1()



    plt.show()

     