import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
dt = 0.1
num_steps = 100
n_balls = 4  # Number of balls

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

# Simulate and store all trajectories
trajectories = []
for b in range(n_balls):
    x0 = np.random.uniform(0, 10)
    y0 = np.random.uniform(0, 5)
    v0 = np.random.uniform(20, 40)
    angle = np.random.uniform(30, 60)

    x, y = simulate_trajectory(x0, y0, v0, angle)
    trajectories.append((x, y))

# Plot all trajectories
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(trajectories):
    plt.plot(x, y, label=f"Ball {i+1}", color=colors[i % len(colors)])

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Projectile Motion of Multiple Balls")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
