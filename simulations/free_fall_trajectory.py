import numpy as np
import torch

def free_fall_trajectory(m, dt, y0, g = 9.81, N = 1, dim = 1):
    """
    Calculates the 1D Cartesian trajectory for the free fall of an object of mass m in a gravitational field g.
    
    This function computes the time evolution of an object's position under constant gravitational acceleration in 1D. 
    The trajectory is calculated based on the object's initial position and the gravitational constant.

        Parameters:
        m (int or float): mass of the object in kg (currently not used in the calculations).
        dt (int or float): time step for simulation in seconds.
        g (int or float): gravitational acceleration in m/s^2 (default is 9.81 m/s^2).
        y0 (int or float): initial position of the object in meters (height from which the object falls).
        N (int, optional): number of objects (default is 1).
        dim (int, optional): number of spatial dimensions (default is 1 for a 1D trajectory).

        Returns: 
        dict: A dictionary containing the following:
            - "time": A tensor of time values corresponding to the trajectory.
            - "positions": A tensor of positions, with shape (T, N, dim), where T is the number of time steps.
            - "masses": A tensor representing the mass of the object (currently a scalar, repeated for N objects).
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
    trajectory_data["positions"] = torch.tensor(trajectory[:, :-1].reshape(T, N, dim))
    trajectory_data["masses"] = torch.tensor(m)

    return trajectory_data

