import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
dt = 0.1
num_steps = 100
n_balls = 4  # Number of balls
noise_std = 2.0  # Standard deviation of observation noise

# Function to simulate trajectory
def simulate_trajectory(x0, y0, v0, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    x, y = [x0], [y0]
    for _ in range(num_steps):
        vy -= g * dt
        new_x = x[-1] + vx * dt
        new_y = y[-1] + vy * dt
        if new_y < 0:  # Stop if ball hits the ground
            break
        x.append(new_x)
        y.append(new_y)
    return np.array(x), np.array(y)

# Add noise to observations
def add_noise(x, y, noise_std):
    x_noisy = x + np.random.normal(0, noise_std, size=x.shape)
    y_noisy = y + np.random.normal(0, noise_std, size=y.shape)
    return x_noisy, y_noisy

# Simulate and store all trajectories
trajectories = []
noisy_observations = []
for b in range(n_balls):
    x0 = np.random.uniform(0, 5)
    y0 = np.random.uniform(0, 2)
    v0 = np.random.uniform(20, 40)
    angle = np.random.uniform(30, 60)

    x, y = simulate_trajectory(x0, y0, v0, angle)
    x_noisy, y_noisy = add_noise(x, y, noise_std)

    trajectories.append((x, y))
    noisy_observations.append((x_noisy, y_noisy))

# Plot all trajectories with noisy observations
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
plt.figure(figsize=(10, 6))
for i, ((x_true, y_true), (x_noisy, y_noisy)) in enumerate(zip(trajectories, noisy_observations)):
    color = colors[i % len(colors)]
    plt.plot(x_true, y_true, label=f"True Path {i+1}", color=color)
    plt.scatter(x_noisy, y_noisy, s=10, alpha=0.4, color=color, label=f"Noisy Obs {i+1}")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Projectile Motion with Noisy Observations")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
