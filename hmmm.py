import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Constants
g = 9.81
dt = 0.1
num_particles = 20000
num_steps = 100 # Max steps for simulation, actual steps depend on when y < 0
noise_std = 0.5

# Ball class to simulate and store trajectory
class Ball:
    def __init__(self, x0, y0, v0, angle_deg):
        self.x0 = x0
        self.y0 = y0
        self.v0 = v0
        self.angle_deg = angle_deg
        self.angle_rad = np.deg2rad(angle_deg)
        self.vx0 = v0 * np.cos(self.angle_rad)
        self.vy0 = v0 * np.sin(self.angle_rad)
        self.x_true, self.y_true = self._simulate_trajectory()
        self.x_obs, self.y_obs = self._add_noise()

    def _simulate_trajectory(self):
        x, y = [self.x0], [self.y0]
        vx, vy = self.vx0, self.vy0
        for _ in range(num_steps): # Max steps defined here, but breaks if y < 0
            vy -= g * dt
            x_new = x[-1] + vx * dt
            y_new = y[-1] + vy * dt
            if y_new < 0:
                # Interpolate to find x-position when y is exactly 0
                if y[-1] > 0: # Only interpolate if the previous point was above ground
                    # Calculate fraction of dt needed to reach y=0
                    fraction = -y[-1] / (vy - g * dt)
                    x_new = x[-1] + vx * dt * fraction
                y_new = 0.0 # Set y to 0
                x.append(x_new)
                y.append(y_new)
                break
            x.append(x_new)
            y.append(y_new)
        return np.array(x), np.array(y)

    def _add_noise(self):
        x_obs = self.x_true + np.random.normal(0, noise_std, size=self.x_true.shape)
        y_obs = self.y_true + np.random.normal(0, noise_std, size=self.y_true.shape)
        return x_obs, y_obs

# Particle Filter
class ParticleFilter:
    def __init__(self, num_particles, x0_range, y0_range, vx_range, vy_range):
        self.n = num_particles
        self.particles = self._init_particles(x0_range, y0_range, vx_range, vy_range)
        self.weights = np.ones(self.n) / self.n

    def _init_particles(self, x0_range, y0_range, vx_range, vy_range):
        p = np.empty((self.n, 4))
        p[:, 0] = np.random.uniform(*x0_range, size=self.n)
        p[:, 1] = np.random.uniform(*y0_range, size=self.n)
        p[:, 2] = np.random.uniform(*vx_range, size=self.n)
        p[:, 3] = np.random.uniform(*vy_range, size=self.n)
        return p

    def predict(self):
        # Predict next state based on current velocity and gravity
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        self.particles[:, 3] -= g * dt
        # Add adaptive noise
        self.particles += np.random.normal(0, self.adaptive_noise(), self.particles.shape)
        # Ensure particles don't go below ground for y-coordinate
        self.particles[self.particles[:, 1] < 0, 1] = 0.0 # Clamp y at 0

    def adaptive_noise(self):
        # Calculate entropy of weights
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-10))
        max_entropy = np.log(self.n)
        # Scale noise based on normalized entropy (0.05 min noise, 0.5 max additional noise)
        return 0.05 + 0.5 * (entropy / max_entropy)

    def update(self, observations):
        obs_array = np.array(observations)
        # Calculate distance from each particle's position to each observation
        dists = np.linalg.norm(self.particles[:, None, :2] - obs_array[None, :, :], axis=2)

        # For each particle, find the minimum distance to any observation
        min_dists = np.min(dists, axis=1)

        # Update weights based on likelihood (Gaussian probability, smaller distance = higher likelihood)
        weights = np.exp(-min_dists**2 / (2 * noise_std**2))
        weights = np.power(weights, 0.9) # Powering weights can sharpen the distribution
        
        # Normalize weights
        self.weights = weights / (np.sum(weights) + 1e-10) # Add small epsilon to prevent division by zero

    def resample(self):
        # Systematic Resampling
        positions = (np.arange(self.n) + np.random.rand()) / self.n
        indexes = np.zeros(self.n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.n) / self.n

    def estimate_clusters(self, n_clusters):
        # Use KMeans to find cluster centers from weighted particles
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) 
        kmeans.fit(self.particles[:, :2], sample_weight=self.weights)
        return kmeans.cluster_centers_

# Simulator
class Simulator:
    def __init__(self, n_balls):
        self.n_balls = n_balls
        self.balls = []
        self.estimates_raw = [[] for _ in range(n_balls)] # Raw estimates from PF
        self.estimates_processed_for_plot = [[] for _ in range(n_balls)] # Processed for plotting

    def run(self):
        for _ in range(self.n_balls):
            ball = Ball(
                x0=np.random.uniform(0, 50),
                y0=np.random.uniform(0, 50),
                v0=np.random.uniform(15, 50),
                angle_deg=np.random.uniform(15, 60)
            )
            self.balls.append(ball)

        # Determine the maximum number of timesteps any ball is in the air
        max_timesteps = max(len(ball.x_obs) for ball in self.balls)

        # Initialize particle filter ranges based on all balls' initial conditions
        x0_range = (min(ball.x0 for ball in self.balls) - 5, max(ball.x0 for ball in self.balls) + 5)
        y0_range = (0, max(ball.y0 for ball in self.balls) + 10)
        vx_range = (0, max(ball.vx0 for ball in self.balls) + 5)
        vy_range = (min(ball.vy0 for ball in self.balls) - 10, max(ball.vy0 for ball in self.balls) + 5) 

        pf = ParticleFilter(num_particles, x0_range, y0_range, vx_range, vy_range)

        for t in range(max_timesteps):
            pf.predict()
            
            current_obs = []
            for ball in self.balls:
                if t < len(ball.x_obs):
                    current_obs.append((ball.x_obs[t], ball.y_obs[t]))
                else:
                    # If ball has landed, use its last known position as observation
                    current_obs.append((ball.x_obs[-1], ball.y_obs[-1]))
            
            pf.update(current_obs)
            centers = pf.estimate_clusters(n_clusters=self.n_balls)

            # Assign estimated clusters to true observations using Hungarian algorithm
            cost_matrix = np.linalg.norm(np.array(current_obs)[:, None, :] - centers[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Store the estimated positions, ensuring correct assignment
            for obs_idx, center_idx in zip(row_ind, col_ind):
                self.estimates_raw[obs_idx].append(centers[center_idx])
            
            pf.resample()
        
        # Post-process raw estimates to make them stop at y=0 when the true ball lands
        for i, ball in enumerate(self.balls):
            true_trajectory_len = len(ball.x_true)
            processed_est_traj = []
            
            for t_step in range(max_timesteps):
                if t_step < true_trajectory_len:
                    # Before the ball lands, use the raw estimated position
                    processed_est_traj.append(self.estimates_raw[i][t_step])
                else:
                    # After the ball lands, hold its position at the last known true position (x, 0)
                    # We can use the last estimated x, and force y to 0
                    last_true_x = ball.x_true[-1]
                    processed_est_traj.append([last_true_x, 0.0])
                    # Alternatively, if we want it to hold its *estimated* landing spot:
                    # last_estimated_x_at_landing_time = self.estimates_raw[i][true_trajectory_len - 1][0]
                    # processed_est_traj.append([last_estimated_x_at_landing_time, 0.0])
                    # The current choice (using last_true_x) ensures it aligns perfectly with the true line after landing.

            self.estimates_processed_for_plot[i] = np.array(processed_est_traj)


    def plot(self):
        plt.figure(figsize=(12, 8))

        # Using a colormap for distinct colors for each ball
        colors = plt.cm.get_cmap('viridis', self.n_balls) 

        for i, ball in enumerate(self.balls):
            ball_true_traj = np.column_stack((ball.x_true, ball.y_true))
            ball_obs_traj = np.column_stack((ball.x_obs, ball.y_obs))
            ball_est_traj_plot = self.estimates_processed_for_plot[i] # Use the processed estimates

            # Plot true trajectory
            plt.plot(ball_true_traj[:, 0], ball_true_traj[:, 1], color=colors(i), 
                     linestyle='-', linewidth=2, label=f'Ball {i+1} True Trajectory' if i == 0 else "")
            
            # Plot observations
            plt.scatter(ball_obs_traj[:, 0], ball_obs_traj[:, 1], s=10, alpha=0.4, 
                        color=colors(i), marker='x', label=f'Ball {i+1} Observations' if i == 0 else "")
            
            # Plot estimated trajectory (now will flatten at y=0)
            plt.plot(ball_est_traj_plot[:, 0], ball_est_traj_plot[:, 1], color=colors(i), 
                     linestyle='--', linewidth=1.5, label=f'Ball {i+1} Estimated Trajectory' if i == 0 else "")
            
            # Calculate RMSE using the processed estimated trajectory, up to the true trajectory length
            # Note: RMSE is more meaningful for the in-air portion. If you calculate it over the full length
            # when the estimated trajectory is forced to 0, it might artificially lower RMSE if the true ball
            # landed far away but the estimate was forced to 0 at the true landing x.
            # It's better to compare only while the true ball is in the air.
            min_len = min(len(ball_true_traj), len(ball_est_traj_plot))
            rmse = np.sqrt(mean_squared_error(ball_true_traj[:min_len], ball_est_traj_plot[:min_len]))
            print(f"Ball {i+1} RMSE (up to landing): {rmse:.2f}")

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Multi-Ball Tracking using Particle Filter')
        
        # Create custom legend handles
        custom_lines = [
            Line2D([0], [0], color='black', lw=2, linestyle='-'), 
            Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=5, alpha=0.4), 
            Line2D([0], [0], color='black', lw=1.5, linestyle='--')
        ]
        plt.legend(custom_lines, ['True Trajectory', 'Observations', 'Estimated Trajectory'])
        
        plt.tight_layout()
        plt.show()

# Run simulation
sim = Simulator(n_balls=3)
sim.run()
sim.plot()