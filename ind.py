import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
dt = 0.1
num_particles = 1000
num_steps = 100
n_balls = 2  # Number of indistinguishable balls

# Trajectory simulator
def true_trajectory(x0, y0, v0, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    x, y = [x0], [y0]
    for _ in range(num_steps):
        vy -= g * dt
        new_x = x[-1] + vx * dt
        new_y = y[-1] + vy * dt
        if new_y < 0:
            break
        x.append(new_x)
        y.append(new_y)
    return np.array(x), np.array(y)

# Add observation noise
def noisy_observations(x_true, y_true, noise_std=2.0):
    x_obs = x_true + np.random.normal(0, noise_std, size=x_true.shape)
    y_obs = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    return x_obs, y_obs

# Initialize particles
def init_particles(n, x0_range, y0_range, vx_range, vy_range):
    particles = np.empty((n, 4))
    particles[:, 0] = np.random.uniform(*x0_range, size=n)
    particles[:, 1] = np.random.uniform(*y0_range, size=n)
    particles[:, 2] = np.random.uniform(*vx_range, size=n)
    particles[:, 3] = np.random.uniform(*vy_range, size=n)
    return particles

# Prediction
def predict(particles):
    particles[:, 0] += particles[:, 2] * dt
    particles[:, 1] += particles[:, 3] * dt
    particles[:, 3] -= g * dt
    return particles

# Update weights
def update(particles, weights, obs):
    distance = np.linalg.norm(particles[:, 0:2] - obs, axis=1)
    weights[:] = np.exp(-distance**2 / 4.0)
    weights += 1e-300
    weights /= np.sum(weights)
    return weights

# Resampling
def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    particles[:] = particles[indices]
    weights.fill(1.0 / len(weights))
    return particles, weights

# -----------------------------
# Main Simulation (Indistinguishable Balls)
# -----------------------------

true_paths = []
observations_all = []
x_obs_all, y_obs_all = [], []

# Generate trajectories and observations
for _ in range(n_balls):
    x0 = np.random.uniform(5, 20)
    y0 = np.random.uniform(0, 5)
    v0 = np.random.uniform(20, 40)
    angle = np.random.uniform(30, 60)

    x_true, y_true = true_trajectory(x0, y0, v0, angle)
    x_obs, y_obs = noisy_observations(x_true, y_true)

    true_paths.append((x_true, y_true))
    x_obs_all.append(x_obs)
    y_obs_all.append(y_obs)

# Compute max steps safely (min of all lengths)
max_steps = min(len(x) for x in x_obs_all)

# Initialize particle filters
filters = []
estimates_all = []

for b in range(n_balls):
    x0 = x_obs_all[b][0]
    y0 = y_obs_all[b][0]
    vx0 = (x_obs_all[b][1] - x_obs_all[b][0]) / dt
    vy0 = (y_obs_all[b][1] - y_obs_all[b][0]) / dt

    particles = init_particles(
        num_particles,
        x0_range=(x0 - 5, x0 + 5),
        y0_range=(y0 - 5, y0 + 5),
        vx_range=(vx0 - 5, vx0 + 5),
        vy_range=(vy0 - 5, vy0 + 5)
    )
    weights = np.ones(num_particles) / num_particles
    filters.append((particles, weights))
    estimates_all.append([])

# Run filter with greedy data association
for t in range(max_steps):
    obs_list = list(zip([x[t] for x in x_obs_all], [y[t] for y in y_obs_all]))
    used = [False] * n_balls

    for b in range(n_balls):
        particles, weights = filters[b]
        particles = predict(particles)

        # Current estimated position
        est_pos = np.average(particles[:, 0:2], weights=weights, axis=0)

        # Find closest unused observation
        min_dist = float('inf')
        chosen_obs = None
        chosen_idx = -1
        for i, obs in enumerate(obs_list):
            if used[i]:
                continue
            dist = np.linalg.norm(est_pos - obs)
            if dist < min_dist:
                min_dist = dist
                chosen_obs = obs
                chosen_idx = i

        used[chosen_idx] = True
        weights = update(particles, weights, np.array(chosen_obs))

        estimate = np.average(particles[:, 0:2], weights=weights, axis=0)
        estimates_all[b].append(estimate)
        particles, weights = resample(particles, weights)
        filters[b] = (particles, weights)

# ----------------------------------
# Plotting
# ----------------------------------

colors = ['blue', 'orange', 'green', 'red', 'purple']

plt.figure(figsize=(12, 6))
for b in range(n_balls):
    x_true, y_true = true_paths[b]
    estimates = np.array(estimates_all[b])
    color = colors[b % len(colors)]

    plt.plot(x_true, y_true, label=f"True Path {b+1}", color=color)
    plt.plot(estimates[:, 0], estimates[:, 1], linestyle='--', label=f"Estimate {b+1}", color=color)
    plt.scatter(x_obs_all[b], y_obs_all[b], s=8, color=color, alpha=0.3)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Tracking Indistinguishable Balls with Particle Filters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
