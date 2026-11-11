#%%
'''
author: toni
created on: 2025-05-22
last modified on: 2025-05-22

Plots verschiedener Sigmoid-Funktionen
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid-Funktion

#%%
# Eigene Sigmoidfunktion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
# Sigmoid-Funktion
sigmoid_z = sigmoid(z)
# Sigmoid-Funktion von scipy
sigmoid_z_scipy = expit(z)

#%%
#plots
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid_z, label='Sigmoid (eigene Funktion)', color='blue')
plt.plot(z, sigmoid_z_scipy, label='Sigmoid (scipy)', linestyle='--', color='orange')
plt.title('Sigmoid-Funktion')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
#%%
beta_0 = 0
beta_1 = 1
x = np.linspace(-10, 10, 100)
# Sigmoid-Funktion
sigmoid_x = sigmoid(beta_0 + beta_1 * x)
# Sigmoid-Funktion von scipy
sigmoid_x_scipy = expit(beta_0 + beta_1 * x)
#%%
#plots
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_x, label='Sigmoid (eigene Funktion)', color='blue')
plt.plot(x, sigmoid_x_scipy, label='Sigmoid (scipy)', linestyle='--', color='orange')
plt.title('Sigmoid-Funktion mit beta_0=0 und beta_1=1')
plt.xlabel('x')
plt.ylabel('sigmoid(beta_0 + beta_1 * x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
#%%
for beta_0 in [-1, -0.5, 0, 0.5, 1]:
    for beta_1 in [-1, 0, 1]:
        x = np.linspace(-10, 10, 100)
        # Sigmoid-Funktion
        sigmoid_x = sigmoid(beta_0 + beta_1 * x)
        # Sigmoid-Funktion von scipy
        sigmoid_x_scipy = expit(beta_0 + beta_1 * x)
        #plots
        # plt.figure(figsize=(10, 6))
        plt.plot(x, sigmoid_x, label='Sigmoid (eigene Funktion)', color='blue')
        plt.plot(x, sigmoid_x_scipy, label='Sigmoid (scipy)', linestyle='--', color='orange')
        plt.title(f'Sigmoid-Funktion mit beta_0={beta_0} und beta_1={beta_1}')
        plt.xlabel('x')
        plt.ylabel('sigmoid(beta_0 + beta_1 * x)')
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.axvline(0, color='black', lw=0.5, ls='--')
        plt.grid()
        plt.legend()
        plt.show()


#%%


colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # beliebig erweiterbar

plt.figure(figsize=(10, 6))
for idx, beta_1 in enumerate([-1, 0, 0.01, 0.1, 0.5, 1, 10]):
    beta_0 = 0
    x = np.linspace(-10, 10, 100)
    # Sigmoid-Funktion
    sigmoid_x = sigmoid(beta_0 + beta_1 * x)
    #plots
    color=colors[idx % len(colors)]  # Farbwechsel
    # plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid_x, label=f'beta1 = {beta_1}', color=color)
    plt.title(f'Sigmoid-Funktion mit beta_0={beta_0} und beta_1={beta_1}')
    plt.xlabel('x')
    plt.ylabel('sigmoid(beta_0 + beta_1 * x)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
plt.show()

#%%
plt.figure(figsize=(10, 6))
for idx, beta_0 in enumerate([-5, -1, 0, 0.01, 0.1, 0.5, 1, 10]):
    beta_1 = 1
    x = np.linspace(-10, 10, 100)
    # Sigmoid-Funktion
    sigmoid_x = sigmoid(beta_0 + beta_1 * x)
    #plots
    color=colors[idx % len(colors)]  # Farbwechsel
    # plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid_x, label=f'beta0 = {beta_0}', color=color)
    plt.title(f'Sigmoid-Funktion mit beta_0={beta_0} und beta_1={beta_1}')
    plt.xlabel('x')
    plt.ylabel('sigmoid(beta_0 + beta_1 * x)')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
plt.show()
# %%
