import numpy as np

import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
T = 1.0  # Total time
N = 100  # Number of grid points
alpha = 0.01  # Thermal diffusivity
theta = 0.5  # Theta value for the method

# Discretization
dx = L / (N - 1)
dt = (dx ** 2) * theta / alpha
nt = int(T / dt)

# Initial condition
x = np.linspace(0, L, N)
u0 = np.sin(np.pi * x)

# Solution array
u = np.zeros((nt, N))
u[0] = u0

# Time-stepping scheme
for n in range(1, nt):
    u[n, 1:-1] = u[n-1, 1:-1] + theta * alpha * dt / (dx ** 2) * (u[n-1, 2:] - 2 * u[n-1, 1:-1] + u[n-1, :-2])

# Plotting the solution
plt.figure()
for n in range(0, nt, int(nt/10)):
    plt.plot(x, u[n], label=f"t = {n*dt:.2f}")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()