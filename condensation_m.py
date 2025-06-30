#MAIN CODE FOR PORTFOLIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========================
# Particle Filter Ball Class
# ========================
class ParticleFilterBall:
    def __init__(self, init_pos, init_vel, num_particles=500, dt=0.1,
                 gravity=9.81, process_noise_std=0.5, obs_noise_std=2.0):
        self.num_particles = num_particles
        self.dt = dt
        self.gravity = gravity
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std

        # True state (position and velocity)
        self.true_pos = np.array(init_pos, dtype=np.float64)
        self.true_vel = np.array(init_vel, dtype=np.float64)

        # Store trajectories for plotting
        self.true_path = []
        self.obs_path = []
        self.est_path = []

        # Initialize particles: state = [x, y, vx, vy]
        self.particles = np.empty((self.num_particles, 4))
        self.particles[:, 0] = np.random.normal(init_pos[0], 1.0, self.num_particles)
        self.particles[:, 1] = np.random.normal(init_pos[1], 1.0, self.num_particles)
        self.particles[:, 2] = np.random.normal(init_vel[0], 1.0, self.num_particles)
        self.particles[:, 3] = np.random.normal(init_vel[1], 1.0, self.num_particles)

    def propagate_true_state(self):
        # Update true velocity and position under gravity
        self.true_vel[1] -= self.gravity * self.dt
        self.true_pos += self.true_vel * self.dt
        return self.true_pos.copy()

    def observe(self):
        # Simulate noisy observation of true position
        return self.true_pos + np.random.randn(2) * self.obs_noise_std

    def transition(self, particles):
        # Propagate particles according to motion model + noise (vectorized)
        particles[:, 0] += particles[:, 2] * self.dt
        particles[:, 1] += particles[:, 3] * self.dt - 0.5 * self.gravity * self.dt**2
        particles[:, 3] -= self.gravity * self.dt
        particles[:, :2] += np.random.randn(self.num_particles, 2) * self.process_noise_std
        return particles

    def compute_weights(self, particles, observation):
        # Vectorized weight calculation using Gaussian likelihood
        diffs = particles[:, :2] - observation
        dists_squared = np.sum(diffs**2, axis=1)
        weights = np.exp(-0.5 * dists_squared / self.obs_noise_std**2)
        weights += 1e-300  # avoid zero weights
        weights /= np.sum(weights)
        return weights

    def resample_particles(self, weights):
        # Systematic resampling for efficiency and diversity
        positions = (np.arange(self.num_particles) + np.random.rand()) / self.num_particles
        indexes = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]

    def update(self):
        # Propagate true state
        self.propagate_true_state()

        # Generate observation
        obs = self.observe()

        # Particle filter update steps
        self.particles = self.transition(self.particles)
        weights = self.compute_weights(self.particles, obs)
        self.resample_particles(weights)

        # Estimate state as weighted mean
        estimate = np.average(self.particles[:, :2], weights=weights, axis=0)

        # Record paths
        self.true_path.append(self.true_pos.copy())
        self.obs_path.append(obs.copy())
        self.est_path.append(estimate.copy())

        return self.particles[:, :2], estimate


# ========================
# Multi-ball tracker manager class
# ========================
class MultiBallTracker:
    def __init__(self, num_balls=3, **pf_kwargs):
        self.balls = []
        self.num_balls = num_balls
        self.pf_kwargs = pf_kwargs
        self.colors = ['red', 'blue', 'green', 'purple', 'orange']

    def initialize_balls(self):
        self.balls.clear()
        for _ in range(self.num_balls):
            init_pos = [np.random.uniform(0, 30), np.random.uniform(15, 30)]
            init_vel = [np.random.uniform(10, 20), np.random.uniform(10, 20)]
            self.balls.append(ParticleFilterBall(init_pos, init_vel, **self.pf_kwargs))

    def update_all(self):
        results = []
        for ball in self.balls:
            if ball.true_pos[1] < 0:  # Skip if ball fell below ground
                results.append((np.empty((0, 2)), None))
                continue
            pts, est = ball.update()
            results.append((pts, est))
        return results


# ========================
# Visualization & Animation
# ========================
def run_animation(num_balls, max_steps):
    # Setup particle filter params
    pf_params = {
        'num_particles': 500,
        'dt': 0.1,
        'gravity': 9.81,
        'process_noise_std': 0.5,
        'obs_noise_std': 2.0
    }

    tracker = MultiBallTracker(num_balls=num_balls, **pf_params)
    tracker.initialize_balls()

    fig, ax = plt.subplots()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_title("Multi-Ball Tracking with Particle Filter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    sc_particles = [ax.scatter([], [], s=5, color=tracker.colors[i % len(tracker.colors)], alpha=0.3)
                    for i in range(num_balls)]
    ln_true = [ax.plot([], [], '-', color=tracker.colors[i % len(tracker.colors)],
                      label=f'Ball {i+1} True')[0] for i in range(num_balls)]
    ln_est = [ax.plot([], [], 'o--', color=tracker.colors[i % len(tracker.colors)],
                    label=f'Ball {i+1} Estimate')[0] for i in range(num_balls)]

    ax.legend()

    def init():
        # No clearing to keep graph persistent as per your request
        return sc_particles + ln_true + ln_est

    def update(frame):
    
        results = tracker.update_all()
    
        # Collect all positions from all balls for axis limit adjustments
        all_x = []
        all_y = []
    
        for i, (pts, est) in enumerate(results):
            if pts.size == 0:
                continue
            sc_particles[i].set_offsets(pts)
            ln_true[i].set_data(*zip(*tracker.balls[i].true_path))
            ln_est[i].set_data(*zip(*tracker.balls[i].est_path))
        
            # Accumulate all positions to compute plot limits
            true_x, true_y = zip(*tracker.balls[i].true_path)
            est_x, est_y = zip(*tracker.balls[i].est_path)
            all_x.extend(true_x)
            all_x.extend(est_x)
            all_y.extend(true_y)
            all_y.extend(est_y)
    
        if all_x and all_y:
            margin = 10  # Add some margin around points
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
        return sc_particles + ln_true + ln_est


    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=100)
    plt.show()


# ========================
# Run the animation
# ========================
if __name__ == "__main__":
    run_animation(num_balls=5, max_steps=100)
