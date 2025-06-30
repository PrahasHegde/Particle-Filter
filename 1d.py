import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_PARTICLES = 1000
NUM_STEPS = 20
INITIAL_POSITION = 0
INITIAL_VELOCITY = 10
PROCESS_NOISE_STD = 1.0
MEASUREMENT_NOISE_STD = 2.0

# Initialize particles: [position, velocity]
particles = np.zeros((NUM_PARTICLES, 2))
particles[:, 0] = np.random.normal(INITIAL_POSITION, 1, NUM_PARTICLES)  # position
particles[:, 1] = np.random.normal(INITIAL_VELOCITY, 1, NUM_PARTICLES)  # velocity

# True state
true_pos = INITIAL_POSITION
true_vel = INITIAL_VELOCITY

estimated_positions = []
true_positions = []

for t in range(NUM_STEPS):
    # Simulate true motion
    true_pos += true_vel + np.random.normal(0, PROCESS_NOISE_STD)
    true_positions.append(true_pos)

    # Simulate noisy measurement of position
    measured_pos = true_pos + np.random.normal(0, MEASUREMENT_NOISE_STD)

    # Prediction Step (motion model)
    particles[:, 0] += particles[:, 1] + np.random.normal(0, PROCESS_NOISE_STD, NUM_PARTICLES)

    # Measurement Update (weighting based on observation)
    distances = np.abs(particles[:, 0] - measured_pos)
    weights = np.exp(- (distances**2) / (2 * MEASUREMENT_NOISE_STD**2))
    weights += 1e-300  # avoid divide by zero
    weights /= np.sum(weights)

    # Estimate current position
    estimated_pos = np.sum(particles[:, 0] * weights)
    estimated_positions.append(estimated_pos)

    # Resampling Step
    indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    particles = particles[indices]

# Plotting
plt.plot(true_positions, label="True Position")
plt.plot(estimated_positions, label="Estimated Position (Particle Filter)")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Particle Filter: Estimating 1D Ball Position")
plt.legend()
plt.grid(True)
plt.show()
