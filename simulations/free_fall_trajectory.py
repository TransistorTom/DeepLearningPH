import numpy as np
import torch

def free_fall_trajectory(m, dt, y0, g = 9.81, N = 1, dim = 1):
    """
    Calculates a 1D cartesian trajectory for the free fall of an object of mass m in gravitational field g.  
    
        Parameters:
    m (int or float): mass of the object in kg \\
    dt (int or float): time step for simulation in s \\
    g (int or float): gravity constant (standard is 9.81) in m/s^2 \\
    y0 (int or float): initial position of object in m \\

        Returns: 
    The output is a numpy array containing the (y,t) coordinates of the falling object
    """
    t_max = np.sqrt(2 * y0 / g)
    t = np.arange(0, t_max, dt)
    T = len(t)
    # compute free fall trajectory
    y = y0 - 0.5 * g * t**2

    # Remove negative y values and make np array
    valid = y >= 0
    y, t = y[valid], t[valid]
    trajectory = np.column_stack((y,t))

    trajectory_data = {
        "time": torch.arange(T, dtype=torch.float32),
        "positions": torch.zeros((T, N, dim), dtype=torch.float32),
        "masses": None
    }
    trajectory_data["time"] = torch.tensor(t)
    trajectory_data["positions"] = torch.tensor(trajectory[:, :-1].reashape(T, N, dim))
    trajectory_data["masses"] = torch.tensor(m)

    return trajectory_data

