import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
dt = 0.1
num_particles = 1000
num_steps = 100
n_balls = 4  # Number of balls

# Trajectory simulator (with initial position)
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


# Initialize particles for a specific region
def init_particles(n, x0_range, y0_range, vx_range, vy_range):
    particles = np.empty((n, 4))
    particles[:, 0] = np.random.uniform(*x0_range, size=n)  # x
    particles[:, 1] = np.random.uniform(*y0_range, size=n)  # y
    particles[:, 2] = np.random.uniform(*vx_range, size=n)  # vx
    particles[:, 3] = np.random.uniform(*vy_range, size=n)  # vy
    return particles

# Prediction step
def predict(particles):
    particles[:, 0] += particles[:, 2] * dt
    particles[:, 1] += particles[:, 3] * dt
    particles[:, 3] -= g * dt
    return particles

# Update weights based on observations
def update(particles, weights, obs):
    distance = np.linalg.norm(particles[:, 0:2] - obs, axis=1)
    weights[:] = np.exp(-distance**2 / 4.0)
    weights += 1.e-300
    weights /= np.sum(weights)
    return weights

# Resample particles
def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    particles[:] = particles[indices]
    weights.fill(1.0 / len(weights))
    return particles, weights

# ==============================
# Multi-ball simulation
# ==============================

# Store all results
true_paths = []
observations = []
estimates_all = []

for b in range(n_balls):
    # Generate random initial conditions
    x0 = np.random.uniform(5, 20)
    y0 = np.random.uniform(0, 5)
    v0 = np.random.uniform(20, 40)
    angle = np.random.uniform(30, 60)

    #Simulate true path and noisy observations
    x_true, y_true = true_trajectory(x0, y0, v0, angle)
    x_obs, y_obs = noisy_observations(x_true, y_true)
    true_paths.append((x_true, y_true))
    observations.append((x_obs, y_obs))

    #Initialize particles around a rough range
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

    #Run particle filter
    for t in range(len(x_obs)):
        particles = predict(particles)
        weights = update(particles, weights, np.array([x_obs[t], y_obs[t]]))
        estimate = np.average(particles[:, 0:2], weights=weights, axis=0)
        estimates.append(estimate)
        particles, weights = resample(particles, weights)

    estimates_all.append(np.array(estimates))

# ==============================
# Plotting
# ==============================

colors = ['blue', 'orange', 'green', 'purple', 'magenta', 'brown']
# colors = ['blue','blue','blue','blue','blue','blue','blue']


plt.figure(figsize=(12, 6))
for b in range(n_balls):
    x_true, y_true = true_paths[b]
    x_obs, y_obs = observations[b]
    estimates = estimates_all[b]
    color = colors[b % len(colors)]

    plt.plot(x_true, y_true, label=f"True Path {b+1}", color=color)
    plt.scatter(x_obs, y_obs, s=10, alpha=0.4, color=color, label=f"Observations {b+1}")
    plt.plot(estimates[:, 0], estimates[:, 1], linestyle='--', color=color, label=f"Estimate {b+1}")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Multi-Ball Tracking with Particle Filters")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

