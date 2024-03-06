import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the rod
T = 1.0  # Total simulation time
Nx = 100  # Number of spatial grid points
Nt = 1000  # Number of time steps
alpha = 0.01  # Thermal diffusivity
dx = L / Nx  # Spatial step size
dt = T / Nt  # Time step size

# Initialize temperature field
u = np.zeros(Nx + 1)

# Set initial condition (u(x, 0) = sin(pi*x))
x = np.linspace(0, L, Nx + 1)
u[:] = np.sin(np.pi * x)

# Neumann boundary conditions (du/dx = 0 at both ends)
u[0] = u[1]
u[-1] = u[-2]

# Finite difference scheme
for n in range(Nt):
    u_new = np.copy(u)
    for i in range(1, Nx):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new

# Plot the final temperature distribution
plt.figure(figsize=(8, 4))
plt.plot(x, u, label="Numerical solution")
plt.xlabel("Position (x)")
plt.ylabel("Temperature (u)")
plt.title("1D Heat Equation with Neumann Boundary Conditions")
plt.grid(True)
plt.legend()
plt.show()
