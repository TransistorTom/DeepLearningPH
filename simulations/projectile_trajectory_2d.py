import numpy as np
import torch

def projectile_trajectory_2d(m, dt, theta, v0, g = 9.81, x0 = 0, y0 = 0, N=1, dim=2):
    """
    Calculates the 2D Cartesian trajectory of a projectile of mass m launched at an angle theta with initial velocity v0.
    
    This function computes the trajectory of a projectile under the influence of gravity, launched from an initial position 
    (x0, y0) with an initial velocity v0 at an angle theta to the horizontal. The object is assumed to be in a vacuum (no air resistance).

        Parameters:
        m (int or float): mass of the projectile in kg (currently not used in the calculations).
        dt (int or float): time step for simulation in seconds.
        theta (int or float): launch angle of the projectile relative to the x-axis in degrees.
        v0 (int or float): initial velocity of the projectile in m/s.
        g (int or float): gravitational acceleration in m/s^2 (default is 9.81 m/s^2).
        x0 (int or float, optional): initial x-coordinate of the projectile in meters (default is 0).
        y0 (int or float, optional): initial y-coordinate of the projectile in meters (default is 0).
        N (int, optional): number of projectiles (default is 1).
        dim (int, optional): number of spatial dimensions (default is 2 for 2D trajectory).

        Returns: 
        dict: A dictionary containing the following:
            - "time": A tensor of time values corresponding to the trajectory.
            - "positions": A tensor of projectile positions, with shape (T, N, dim), where T is the number of time steps.
            - "masses": A tensor representing the mass of the projectile (currently a scalar, repeated for N projectiles).
    """
    
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Initial velocity components
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    # Time array
    t_max = (vy0 + np.sqrt(vy0**2 + 2 * g * y0)) / g  # Maximum time until projectile hits the ground
    t = np.arange(0, t_max, dt)  # Generate time array from 0 to t_max
    T = len(t)  # Total number of time steps

    # Compute projectile trajectory
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2

    # Remove negative y values (when the projectile hits the ground)
    valid = y >= 0
    x, y, t = x[valid], y[valid], t[valid]
    
    # Stack the x, y, and t values into a trajectory array
    trajectory = np.column_stack((x, y, t))

    # Create trajectory data dictionary
    trajectory_data = {
        "time": torch.arange(T, dtype=torch.float32),
        "positions": torch.zeros((T, N, dim), dtype=torch.float32),
        "masses": torch.arange(N, dtype=torch.float32)
    }
    
    # Convert time and positions into tensors
    trajectory_data["time"] = torch.tensor(t)
    trajectory_data["positions"] = torch.tensor(trajectory[:, :-1].reshape(T, N, dim))
    trajectory_data["masses"] = torch.tensor(m)

    return trajectory_data
