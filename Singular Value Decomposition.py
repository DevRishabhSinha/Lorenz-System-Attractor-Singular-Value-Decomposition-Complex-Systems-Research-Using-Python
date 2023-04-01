import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
sigma = 10
rho = 28
beta = 8/3

# Lorenz system
def lorenz(t, xyz):
    x, y, z = xyz
    dxdt = sigma*(y - x)
    dydt = rho*x - y - x*z
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

# Initial conditions
x0 = [-8, 8, 27]

# Time interval and step
t_span = [0, 50]
t_eval = np.arange(t_span[0], t_span[1], 0.01)

# Solve the differential equations
sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval)

# Trajectory matrix
X = sol.y

# Singular value decomposition
U, S, Vt = np.linalg.svd(X)

# Convert Vt to V
V = Vt.T

# Trajectory matrix plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X[0], X[1], X[2], c=t_eval, cmap='viridis')
fig.colorbar(sc)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Trajectory Matrix of Lorenz Model')
plt.show()

# SVD plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=t_eval, cmap='viridis')
fig.colorbar(sc)
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_zlabel('v3')
ax.set_title('Trajectory Matrix of SVD')
plt.show()
