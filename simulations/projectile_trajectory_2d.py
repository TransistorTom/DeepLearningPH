import numpy as np
import torch

def projectile_trajectory_2d(m, dt, theta, v0, g = 9.81, x0 = 0, y0 = 0, N=1, dim=2):
    """
    Calculates a 2D cartesian trajectory of a projectile of mass m launched at angle theta with initial velocity v0.  
    
        Parameters:
    theta (int or float): angle between projectile trajectory and x-axis at t=0s in degrees \\
    v0 (int or float): the intial velocity in m/s \\
    m (int or float): mass of the projectile in kg \\
    dt (int or float): time step for simulation in s \\
    g (int or float): gravity constant (standard is 9.81) in m/s^2 \\
    x0, y0 (int or float): initial position of ball in m \\

        Returns: 
    The output is a dictionary containing the {time, positions, velocities, masses} coordinates of the projectile
    """

    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Initial velocity components
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    # Time array
    t_max = (vy0 + np.sqrt(vy0**2+2*g*y0)) / g
    t = np.arange(0, t_max, dt)
    T = len(t)

    # Compute trajectory
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2

    # Remove negative y values and make np array
    valid = y >= 0
    x, y, t = x[valid], y[valid], t[valid]
    trajectory = np.column_stack((x,y))

    vx = np.full_like(x, vx0)
    vy = vy0 - g * t
    trajectory_velocities = np.column_stack((vx,vy))
    
    trajectory_data = {
        "time": torch.arange(T, dtype=torch.float32),
        "positions": torch.zeros((T, N, dim), dtype=torch.float32),
        "velocities": torch.zeros((T, N, dim), dtype=torch.float32),
        "masses": torch.arange(N, dtype=torch.float32)
    }
    
    trajectory_data["time"] = torch.tensor(t)
    trajectory_data["positions"] = torch.tensor(trajectory[:, :].reshape(T, N, dim))
    trajectory_data["velocities"] = torch.tensor(trajectory_velocities[:, :].reshape(T, N, dim))
    trajectory_data["masses"] = torch.tensor(m)

    return trajectory_data
