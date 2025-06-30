import numpy as np
import matplotlib.pyplot as plt

# Config
NUM_BALLS = 5
NUM_PARTICLES = 1000
NUM_STEPS = 50
DT = 0.1
GRAVITY = 9.8
PROCESS_NOISE_STD = 0.3
MEASUREMENT_NOISE_STD = 1.0

# Initial true states for each ball: [x, y, vx, vy]
initial_states = [
    [1.0, 0.0, 5.0, 10.0],
    [3.0, 2.0, 4.0, 12.0],
    [7.0, 5.0, 6.0, 9.0],
    [10.0, 1.0, 3.0, 8.0],
    [12.0, 3.0, 2.0, 7.0]
]

# Create particle filters for each ball
particles = [np.zeros((NUM_PARTICLES, 4)) for _ in range(NUM_BALLS)]
for i, (x, y, vx, vy) in enumerate(initial_states):
    particles[i][:, 0] = np.random.normal(x, 0.5, NUM_PARTICLES)
    particles[i][:, 1] = np.random.normal(y, 0.5, NUM_PARTICLES)
    particles[i][:, 2] = np.random.normal(vx, 0.5, NUM_PARTICLES)
    particles[i][:, 3] = np.random.normal(vy, 0.5, NUM_PARTICLES)

# Initialize true states
true_states = [state.copy() for state in initial_states]
true_paths = [[] for _ in range(NUM_BALLS)]
estimated_paths = [[] for _ in range(NUM_BALLS)]

# Simulation loop
for step in range(NUM_STEPS):
    for i in range(NUM_BALLS):
        x, y, vx, vy = true_states[i]

        # True motion
        x += vx * DT
        y += vy * DT - 0.5 * GRAVITY * DT**2
        vy -= GRAVITY * DT
        true_states[i] = [x, y, vx, vy]

        if y < 0:
            continue  # ball has hit the ground

        true_paths[i].append((x, y))

        # Observation
        measured_x = x + np.random.normal(0, MEASUREMENT_NOISE_STD)
        measured_y = y + np.random.normal(0, MEASUREMENT_NOISE_STD)

        # Predict
        p = particles[i]
        p[:, 0] += p[:, 2] * DT
        p[:, 1] += p[:, 3] * DT - 0.5 * GRAVITY * DT**2
        p[:, 3] -= GRAVITY * DT
        p[:, 0] += np.random.normal(0, PROCESS_NOISE_STD, NUM_PARTICLES)
        p[:, 1] += np.random.normal(0, PROCESS_NOISE_STD, NUM_PARTICLES)

        # Update
        dists = np.linalg.norm(p[:, :2] - np.array([measured_x, measured_y]), axis=1)
        weights = np.exp(- (dists**2) / (2 * MEASUREMENT_NOISE_STD**2))
        weights += 1e-300
        weights /= np.sum(weights)

        # Estimate
        est_x = np.sum(p[:, 0] * weights)
        est_y = np.sum(p[:, 1] * weights)
        estimated_paths[i].append((est_x, est_y))

        # Resample
        indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
        particles[i] = p[indices]

# Convert and plot
colors = ['red', 'green', 'blue', 'orange', 'purple']
plt.figure(figsize=(10, 6))

for i in range(NUM_BALLS):
    true_arr = np.array(true_paths[i])
    est_arr = np.array(estimated_paths[i])
    if len(true_arr) == 0 or len(est_arr) == 0:
        continue
    plt.plot(true_arr[:, 0], true_arr[:, 1], color=colors[i], label=f'Ball {i+1} True')
    plt.plot(est_arr[:, 0], est_arr[:, 1], linestyle='--', color=colors[i], label=f'Ball {i+1} Estimated')
    plt.scatter(true_arr[:, 0], true_arr[:, 1], c=colors[i], s=10)
    plt.scatter(est_arr[:, 0], est_arr[:, 1], c=colors[i], s=10, marker='x')

plt.title("Particle Filter for Multiple Balls in Projectile Motion")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.show()
