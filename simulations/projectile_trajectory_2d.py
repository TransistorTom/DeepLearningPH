import numpy as np

def projectile_trajectory_2d(theta, v0, m, dt, g = 9.81, x0 = 0, y0 = 0):
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
    The output is a numpy array containing the (x,y,t) coordinates of the projectile
    """
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Initial velocity components
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    # Time array
    t_max = (vy0 + np.sqrt(vy0**2+2*g*y0)) / g
    t = np.arange(0, t_max, dt)

    # Compute trajectory
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2

    # Remove negative y values and make np array
    valid = y >= 0
    x, y, t = x[valid], y[valid], t[valid]
    trajectory = np.array([[[x[i], y[i]], t[i]] for i in range(len(t))], dtype=object)

    return trajectory
