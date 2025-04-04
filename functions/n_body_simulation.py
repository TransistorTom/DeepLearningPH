import torch
import random
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def generate_random_positions(N, dim, min_dist, box_size):
    positions = []
    while len(positions) < N:
        pos = torch.rand(dim) * box_size
        if all(torch.norm(pos - p) >= min_dist for p in positions):
            positions.append(pos)
    return torch.stack(positions)

def generate_random_velocities(N, dim, velocity_scale):
    return (torch.rand((N, dim)) - 0.5) * 2 * velocity_scale

def generate_unique_masses(N, mass_range, resolution=25):

    # Create a fine grid in the range
    mass_grid = torch.linspace(mass_range[0], mass_range[1], resolution).tolist()
    # Randomly choose N unique masses
    unique_masses = random.sample(mass_grid, N)

    return torch.tensor(unique_masses, dtype=torch.float32)

def compute_gravitational_forces(positions, masses, G=1.0, eps=5e-3):
    N, dim = positions.shape
    forces = torch.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            dist = torch.norm(r_vec)
            force_mag = G * masses[i] * masses[j] / (dist**2 + eps)
            force_dir = r_vec / dist
            force = force_mag * force_dir
            forces[i] += force
            forces[j] -= force
    return forces

def n_body_simulation(N=5, T=100, dt=0.01, dim=2,
                      mass_range=(1.0, 7.0), min_dist=0.5,
                      box_size=10.0, velocity_scale=1.0, G=1.0):
    # Initialize
    masses = generate_unique_masses(N, mass_range)
    positions = generate_random_positions(N, dim, min_dist, box_size)
    velocities = generate_random_velocities(N, dim, velocity_scale)

    # Store results
    trajectory = torch.zeros((T, N, dim), dtype=torch.float32)
    trajectory_velocities = torch.zeros((T, N, dim), dtype=torch.float32)
    t_array  = torch.arange(0, T * dt, dt, dtype=torch.float32)

    for t in range(T):
        trajectory[t] = positions
        trajectory_velocities[t] = velocities

        # Compute forces and update positions & velocities (Euler method)
        forces = compute_gravitational_forces(positions, masses, G=1.0)
        accelerations = forces / masses[:, None]
        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt

    trajectory_data = {
        "time": t_array,
        "positions": trajectory,
        "velocities": trajectory_velocities,
        "masses": masses
    }

    return trajectory_data