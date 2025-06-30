import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Constants
g = 9.81
dt = 0.1
num_particles = 1000
num_steps = 100
n_balls = 2  # Number of balls

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

# Add noise to trajectory
def noisy_observations(x_true, y_true, noise_std=2.0):
    x_obs = x_true + np.random.normal(0, noise_std, size=x_true.shape)
    y_obs = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    return x_obs, y_obs

# Initialize particles
def init_particles(n, x0_range, y0_range, vx_range, vy_range):
    particles = np.empty((n, 4))
    particles[:, 0] = np.random.uniform(*x0_range, size=n)  # x
    particles[:, 1] = np.random.uniform(*y0_range, size=n)  # y
    particles[:, 2] = np.random.uniform(*vx_range, size=n)  # vx
    particles[:, 3] = np.random.uniform(*vy_range, size=n)  # vy
    return particles

# Predict step with noise
def predict(particles, noise_std=1.0):
    particles[:, 0] += particles[:, 2] * dt + np.random.normal(0, noise_std, size=len(particles))
    particles[:, 1] += particles[:, 3] * dt + np.random.normal(0, noise_std, size=len(particles))
    particles[:, 2] += np.random.normal(0, noise_std * 0.1, size=len(particles))
    particles[:, 3] -= g * dt + np.random.normal(0, noise_std * 0.1, size=len(particles))
    return particles

# Update weights based on Gaussian likelihood
def update(particles, weights, obs, sigma=3.0):
    distance = np.linalg.norm(particles[:, 0:2] - obs, axis=1)
    weights[:] = np.exp(-0.5 * (distance / sigma) ** 2)
    weights += 1e-300
    weights /= np.sum(weights)
    return weights

# Resample particles
def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    particles[:] = particles[indices]
    weights.fill(1.0 / len(weights))
    return particles, weights

# =========================
# Simulate true trajectories
# =========================
true_paths = []
observations = []

for _ in range(n_balls):
    x0 = np.random.uniform(5, 20)
    y0 = np.random.uniform(0, 5)
    v0 = np.random.uniform(15, 40)
    angle = np.random.uniform(30, 60)
    x_true, y_true = true_trajectory(x0, y0, v0, angle)
    x_obs, y_obs = noisy_observations(x_true, y_true)
    true_paths.append((x_true, y_true))
    observations.append((x_obs, y_obs))

# =========================
# Initialize particle filters
# =========================
particles_all = []
weights_all = []

for b in range(n_balls):
    x_true, y_true = true_paths[b]
    vx0 = (x_true[1] - x_true[0]) / dt
    vy0 = (y_true[1] - y_true[0]) / dt + g * dt
    x0_range = (x_true[0] - 5, x_true[0] + 5)
    y0_range = (y_true[0] - 5, y_true[0] + 5)
    vx_range = (vx0 - 5, vx0 + 5)
    vy_range = (vy0 - 5, vy0 + 5)
    particles = init_particles(num_particles, x0_range, y0_range, vx_range, vy_range)
    weights = np.ones(num_particles) / num_particles
    particles_all.append(particles)
    weights_all.append(weights)

# =========================
# Particle Filter with Assignment
# =========================
estimates_all = [[] for _ in range(n_balls)]
max_timesteps = max(len(x) for x, _ in observations)

for t in range(max_timesteps):
    obs_t = np.array([[x_obs[t], y_obs[t]] for x_obs, y_obs in observations if t < len(x_obs)])
    if len(obs_t) < n_balls:
        continue

    # Cluster observations
    kmeans = KMeans(n_clusters=n_balls, n_init=10).fit(obs_t)
    cluster_centers = kmeans.cluster_centers_

    # Predict all particles
    for b in range(n_balls):
        particles_all[b] = predict(particles_all[b])

    # Get current estimates
    estimated_positions = [np.average(p[:, 0:2], weights=w, axis=0)
                           for p, w in zip(particles_all, weights_all)]

    # Optimal assignment (Hungarian algorithm)
    cost_matrix = np.linalg.norm(
        np.array(estimated_positions)[:, None, :] - cluster_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Update and resample
    for i, j in zip(row_ind, col_ind):
        particles = particles_all[i]
        weights = weights_all[i]
        weights = update(particles, weights, cluster_centers[j])
        estimate = np.average(particles[:, 0:2], weights=weights, axis=0)
        estimates_all[i].append(estimate)
        particles, weights = resample(particles, weights)
        particles_all[i] = particles
        weights_all[i] = weights

# =========================
# Plot Results
# =========================
colors = ['blue', 'orange', 'green', 'purple']
plt.figure(figsize=(12, 6))

for b in range(n_balls):
    x_true, y_true = true_paths[b]
    x_obs, y_obs = observations[b]
    estimates = np.array(estimates_all[b])
    color = colors[b % len(colors)]
    plt.plot(x_true, y_true, label=f"True Path {b+1}", color=color)
    plt.scatter(x_obs, y_obs, s=10, alpha=0.4, color=color, label=f"Observations {b+1}")
    if len(estimates) > 0:
        plt.plot(estimates[:, 0], estimates[:, 1], linestyle='--', color=color, label=f"Estimate {b+1}")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Ball Tracking with Improved Particle Filter")
plt.legend()
plt.tight_layout()
plt.show()
