import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
NUM_PARTICLES = 1000
NUM_STEPS = 50
DT = 0.1  # time step (seconds)
GRAVITY = 9.8  # m/s^2
PROCESS_NOISE_STD = 0.3
MEASUREMENT_NOISE_STD = 1.0

# Initial true state
true_x, true_y = 0.0, 0.0
true_vx, true_vy = 5.0, 10.0  # initial velocity

# Initialize particles: [x, y, vx, vy]
particles = np.zeros((NUM_PARTICLES, 4))
particles[:, 0] = np.random.normal(true_x, 0.5, NUM_PARTICLES)
particles[:, 1] = np.random.normal(true_y, 0.5, NUM_PARTICLES)
particles[:, 2] = np.random.normal(true_vx, 0.5, NUM_PARTICLES)
particles[:, 3] = np.random.normal(true_vy, 0.5, NUM_PARTICLES)

# Tracking history
true_positions = []
estimated_positions = []

for t in range(NUM_STEPS):
    # True motion update
    true_x += true_vx * DT
    true_y += true_vy * DT - 0.5 * GRAVITY * DT**2
    true_vy -= GRAVITY * DT  # gravity effect
    true_positions.append((true_x, true_y))

    if true_y < 0:
        break  # stop if ball hits ground

    # Noisy measurement
    measured_x = true_x + np.random.normal(0, MEASUREMENT_NOISE_STD)
    measured_y = true_y + np.random.normal(0, MEASUREMENT_NOISE_STD)

    # Predict step
    particles[:, 0] += particles[:, 2] * DT
    particles[:, 1] += particles[:, 3] * DT - 0.5 * GRAVITY * DT**2
    particles[:, 3] -= GRAVITY * DT  # update vertical velocity
    particles[:, 0] += np.random.normal(0, PROCESS_NOISE_STD, NUM_PARTICLES)
    particles[:, 1] += np.random.normal(0, PROCESS_NOISE_STD, NUM_PARTICLES)

    # Measurement update
    dists = np.linalg.norm(particles[:, :2] - np.array([measured_x, measured_y]), axis=1)
    weights = np.exp(- (dists**2) / (2 * MEASUREMENT_NOISE_STD**2))
    weights += 1e-300
    weights /= np.sum(weights)

    # Estimate state
    est_x = np.sum(particles[:, 0] * weights)
    est_y = np.sum(particles[:, 1] * weights)
    estimated_positions.append((est_x, est_y))

    # Resample
    indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
    particles = particles[indices]

# Convert for plotting
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Path')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b--', label='Estimated Path (PF)')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='green', s=10)
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='blue', s=10)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("2D Particle Filter with Projectile Motion")
plt.grid(True)
plt.legend()
plt.ylim(bottom=0)
plt.show()
