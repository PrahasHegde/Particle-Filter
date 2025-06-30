import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
dt = 0.1
num_particles = 1000
num_steps = 100
n_balls = 4
dropout_rate = 0.3  # Dropout probability

# Dropout range (in time steps)
dropout_start = 20
dropout_end = 30

# True trajectory function
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

# Noisy observations
def noisy_observations(x_true, y_true, noise_std=2.0):
    x_obs = x_true + np.random.normal(0, noise_std, size=x_true.shape)
    y_obs = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    return x_obs, y_obs

# Particle initialization
def init_particles(n, x0_range, y0_range, vx_range, vy_range):
    particles = np.empty((n, 4))
    particles[:, 0] = np.random.uniform(*x0_range, size=n)
    particles[:, 1] = np.random.uniform(*y0_range, size=n)
    particles[:, 2] = np.random.uniform(*vx_range, size=n)
    particles[:, 3] = np.random.uniform(*vy_range, size=n)
    return particles

# Prediction step
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

# Resample particles
def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    particles[:] = particles[indices]
    weights.fill(1.0 / len(weights))
    return particles, weights

# Main simulation
true_paths = []
observations = []
estimates_all = []
dropout_log_all = []

colors = ['blue', 'orange', 'green', 'purple', 'magenta', 'brown']

for b in range(n_balls):
    
    x0 = np.random.uniform(0, 20)
    y0 = np.random.uniform(0, 5)
    v0 = np.random.uniform(20, 40)
    angle = np.random.uniform(30, 60)

    x_true, y_true = true_trajectory(x0, y0, v0, angle)
    x_obs, y_obs = noisy_observations(x_true, y_true)
    true_paths.append((x_true, y_true))
    observations.append((x_obs, y_obs))

    vx0 = v0 * np.cos(np.deg2rad(angle))
    vy0 = v0 * np.sin(np.deg2rad(angle))
    particles = init_particles(
        num_particles,
        x0_range=(x0 - 5, x0 + 5),
        y0_range=(y0 - 5, y0 + 5),
        vx_range=(vx0 - 5, vx0 + 5),
        vy_range=(vy0 - 5, vy0 + 5)
    )
    weights = np.ones(num_particles) / num_particles
    estimates = []
    dropout_log = []

    for t in range(len(x_obs)):
        particles = predict(particles)

        # Dropout condition
        if dropout_start <= t <= dropout_end and np.random.rand() < dropout_rate:
            estimate = np.average(particles[:, 0:2], weights=weights, axis=0)
            dropout_log.append(t)
        else:
            obs = np.array([x_obs[t], y_obs[t]])
            weights = update(particles, weights, obs)
            estimate = np.average(particles[:, 0:2], weights=weights, axis=0)
            particles, weights = resample(particles, weights)

        estimates.append(estimate)

    estimates_all.append(np.array(estimates))
    dropout_log_all.append(dropout_log)

# Plot results
plt.figure(figsize=(12, 6))

for b in range(n_balls):
    x_true, y_true = true_paths[b]
    x_obs, y_obs = observations[b]
    estimates = estimates_all[b]
    dropout_log = dropout_log_all[b]
    color = colors[b % len(colors)]

    plt.plot(x_true, y_true, label=f"True Path {b+1}", color=color)
    plt.scatter(x_obs, y_obs, s=10, alpha=0.4, color=color, label=f"Observations {b+1}")
    plt.plot(estimates[:, 0], estimates[:, 1], linestyle='--', color=color, label=f"Estimate {b+1}")

    # Vertical lines at dropout estimates
    for i, t in enumerate(dropout_log):
        x_pos = estimates[t][0]
        plt.axvline(x=x_pos, color='black', linestyle='-', alpha=0.6,
                    label=f"Dropout {b+1}" if i == 0 else "")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Particle Filter with Dropout")
plt.legend()
plt.tight_layout()
plt.show()
