import numpy as np

def free_fall_trajectory(m, dt, g = 9.81, y0 = 0):
    """
    Calculates a 1D cartesian trajectory for the free fall of an object of mass m in gravitational field g.  
    
        Parameters:
    m (int or float): mass of the object in kg \\
    dt (int or float): time step for simulation in s \\
    g (int or float): gravity constant (standard is 9.81) in m/s^2 \\
    y0 (int or float): initial position (standard is 0) of object in m \\

        Returns: 
    The output is a numpy array containing the (y,t) coordinates of the falling object
    """
    t_max = np.sqrt(2 * y0 / g)
    t=np.arange(0, t_max, dt)

    # compute free fall trajectory
    y = y0 - 0.5 * g * t**2

    # Remove negative y values and make np array
    valid = y >= 0
    x, y, t = x[valid], y[valid], t[valid]
    trajectory = np.column_stack((x,y,t))

    return trajectory
