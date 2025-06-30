import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = None
        self.weights = np.ones(num_particles) / num_particles
        self.g = 9.81  # Gravity (m/s²)
        
    def initialize(self, x_range=(0, 10), y_range=(0, 10), v_range=(5, 20), theta_range=(0, np.pi/2)):
        # Random initial positions
        x0 = np.random.uniform(*x_range, self.num_particles)
        y0 = np.random.uniform(*y_range, self.num_particles)
        
        # Random initial velocities and angles
        v0 = np.random.uniform(*v_range, self.num_particles)
        theta = np.random.uniform(*theta_range, self.num_particles)
        
        # Convert to velocity components
        vx0 = v0 * np.cos(theta)
        vy0 = v0 * np.sin(theta)
        
        self.particles = np.column_stack((x0, y0, vx0, vy0))
    
    def predict(self, dt=0.1):
        # Projectile motion equations
        self.particles[:, 0] += self.particles[:, 2] * dt  # x = x + vx*dt
        self.particles[:, 1] += self.particles[:, 3] * dt - 0.5*self.g*dt**2  # y = y + vy*dt - 0.5*g*dt²
        self.particles[:, 3] -= self.g * dt  # vy = vy - g*dt
    
    def update(self, measurement, measurement_std=0.5):
        # Measurement likelihood (Gaussian)
        dx = self.particles[:, 0] - measurement[0]
        dy = self.particles[:, 1] - measurement[1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Update weights based on measurement likelihood
        self.weights = np.exp(-0.5 * (distances / measurement_std)**2)
        self.weights /= np.sum(self.weights)  # Normalize
    
    def resample(self):
        # Systematic resampling
        indices = np.arange(self.num_particles)
        cumulative_weights = np.cumsum(self.weights)
        uniform_samples = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        new_indices = np.searchsorted(cumulative_weights, uniform_samples)
        self.particles = self.particles[new_indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate(self):
        # Weighted mean of particles
        return np.average(self.particles, weights=self.weights, axis=0)

# Simulation parameters
def simulate_trajectory(x0=0, y0=0, v0=15, theta=np.pi/4, dt=0.1, steps=50):
    """Simulate true projectile motion"""
    g = 9.81
    t = np.arange(0, steps*dt, dt)
    x = x0 + v0*np.cos(theta)*t
    y = y0 + v0*np.sin(theta)*t - 0.5*g*t**2
    vx = v0*np.cos(theta)*np.ones_like(t)
    vy = v0*np.sin(theta) - g*t
    return np.column_stack((x, y, vx, vy))

# Generate noisy measurements
def add_noise(true_positions, std=0.5):
    return true_positions[:, :2] + np.random.normal(0, std, true_positions[:, :2].shape)

# Run simulation
if __name__ == "__main__":
    # Create filter and initialize with random values
    pf = ParticleFilter(num_particles=1000)
    pf.initialize(
        x_range=(0, 5), 
        y_range=(0, 5), 
        v_range=(10, 20), 
        theta_range=(0.2, 1.0)
    )
    
    # Simulate true trajectory
    true_states = simulate_trajectory()
    measurements = add_noise(true_states)
    
    # Tracking results storage
    estimates = []
    
    # Run particle filter
    for z in measurements:
        pf.predict()
        pf.update(z)
        estimates.append(pf.estimate())
        pf.resample()
    
    # Convert to arrays
    estimates = np.array(estimates)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(true_states[:, 0], true_states[:, 1], 'r-', label='True Trajectory')
    plt.plot(measurements[:, 0], measurements[:, 1], 'b.', label='Measurements', alpha=0.3)
    plt.plot(estimates[:, 0], estimates[:, 1], 'g--', label='Particle Filter Estimate')
    plt.title('Projectile Motion Tracking')
    plt.xlabel('Horizontal Position (m)')
    plt.ylabel('Vertical Position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
