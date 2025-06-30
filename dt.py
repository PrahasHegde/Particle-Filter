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
noise_std = 0.5 # Standard deviation of observation noise

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
        for _ in range(num_steps):
            vy -= g * dt
            x_new = x[-1] + vx * dt
            y_new = y[-1] + vy * dt
            if y_new < 0:
                if y[-1] > 0:
                    # Interpolate to find exact ground hit
                    fraction = -y[-1] / (vy - g * dt)
                    x_new = x[-1] + vx * dt * fraction
                y_new = 0.0
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
    def __init__(self, num_particles, x0_range, y0_range, vx_range, vy_range, n_tracked_objects):
        self.n = num_particles
        self.particles = self._init_particles(x0_range, y0_range, vx_range, vy_range)
        self.weights = np.ones(self.n) / self.n
        self.n_tracked_objects = n_tracked_objects # Store the number of objects we expect to track
        self.n_eff_history = [] # To store effective number of particles

    def _init_particles(self, x0_range, y0_range, vx_range, vy_range):
        p = np.empty((self.n, 4))
        p[:, 0] = np.random.uniform(*x0_range, size=self.n)
        p[:, 1] = np.random.uniform(*y0_range, size=self.n)
        p[:, 2] = np.random.uniform(*vx_range, size=self.n)
        p[:, 3] = np.random.uniform(*vy_range, size=self.n)
        return p

    def predict(self):
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        self.particles[:, 3] -= g * dt
        
        # Add adaptive noise
        noise_levels = self.adaptive_noise()
        self.particles += np.random.normal(0, noise_levels, self.particles.shape)
        
        # Ensure particles don't go below ground
        self.particles[self.particles[:, 1] < 0, 1] = 0.0

    def adaptive_noise(self):
        # Avoid log(0) by adding a small epsilon
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-10))
        max_entropy = np.log(self.n)
        
        # Handle case where max_entropy might be very small or zero (e.g., n=1)
        normalized_entropy = entropy / (max_entropy + 1e-10) 

        # Base noise for position and velocity components
        base_pos_noise = 0.05 
        base_vel_noise = 0.02 

        # Additional noise based on uncertainty (entropy)
        additional_pos_noise = 0.5 * normalized_entropy 
        additional_vel_noise = 0.2 * normalized_entropy 

        # Return an array of shape (4,) for [x, y, vx, vy] noise
        return np.array([base_pos_noise + additional_pos_noise, 
                         base_pos_noise + additional_pos_noise, 
                         base_vel_noise + additional_vel_noise, 
                         base_vel_noise + additional_vel_noise])

    def update(self, observations):
        # Convert list of tuples to numpy array once for efficiency
        obs_array = np.array(observations)
        
        if obs_array.shape[0] > 0: # Check if there are any observations
            # Calculate distance from each particle to each observation
            # particles[:, None, :2] creates (n_particles, 1, 2)
            # obs_array[None, :, :] creates (1, n_observations, 2)
            # Resulting diff array is (n_particles, n_observations, 2)
            dists = np.linalg.norm(self.particles[:, None, :2] - obs_array[None, :, :], axis=2)
            
            # Find the minimum distance from each particle to any observation
            min_dists = np.min(dists, axis=1)
            
            # Calculate weights based on these minimum distances
            weights = np.exp(-min_dists**2 / (2 * noise_std**2))
            
            # Normalize weights
            sum_weights = np.sum(weights)
            if sum_weights > 1e-10: # Avoid division by zero
                self.weights = weights / sum_weights
            else:
                self.weights = np.ones(self.n) / self.n # Reset to uniform if all weights are tiny
        # If no observations, weights are NOT updated; they remain uniform from resampling or previous step.

        # Calculate and store N_eff
        sum_of_squared_weights = np.sum(np.square(self.weights))
        N_eff = 1.0 / sum_of_squared_weights if sum_of_squared_weights > 1e-10 else 0
        self.n_eff_history.append(N_eff)


    def resample(self):
        # Systematic Resampling
        # Handle case where weights might sum to zero (e.g., after a prolonged dropout with no effective particles)
        if np.sum(self.weights) < 1e-10:
            self.weights = np.ones(self.n) / self.n # Reset to uniform weights if they are all zero

        positions = (np.arange(self.n) + np.random.rand()) / self.n
        indexes = np.zeros(self.n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.n:
            # Ensure j does not go out of bounds for cumulative_sum
            # This check is more robust, but j should naturally stop at n-1 if positions[i] keeps increasing
            # and cumulative_sum reaches 1.0
            if j >= self.n: 
                j = self.n - 1 
            
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.n) / self.n


    def estimate_clusters(self, n_observations_present):
        # If no observations were made, we assume the targets are in their last estimated positions,
        # or we use the overall weighted mean of particles as a fallback.
        if n_observations_present == 0:
            if np.sum(self.weights) < 1e-10:
                # If weights are all zero, take a simple mean (or last known estimates from Simulator)
                weighted_mean_pos = np.mean(self.particles[:,:2], axis=0) 
            else:
                weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)

        # Determine how many clusters KMeans should try to find
        # Cap clusters by the number of active observations and unique particle positions.
        unique_particles_pos = np.unique(self.particles[:,:2], axis=0)
        num_clusters_for_kmeans = min(self.n_tracked_objects, n_observations_present, len(unique_particles_pos))
        
        # Ensure at least one cluster if there are particles
        if num_clusters_for_kmeans == 0 and len(unique_particles_pos) > 0:
            num_clusters_for_kmeans = 1
        elif num_clusters_for_kmeans == 0: # If no unique particles either, return replicated mean
            weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights) if np.sum(self.weights) > 1e-10 else np.mean(self.particles[:,:2], axis=0)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)


        try:
            # Use `max_iter` for robustness, n_init for better convergence
            kmeans = KMeans(n_clusters=num_clusters_for_kmeans, n_init=10, random_state=42, max_iter=300) 
            kmeans.fit(self.particles[:, :2], sample_weight=self.weights)
            centers = kmeans.cluster_centers_

            # If fewer clusters are found than tracked objects, pad with duplicates (or last known positions)
            while len(centers) < self.n_tracked_objects:
                centers = np.vstack([centers, centers[0]]) # Simple padding, can be improved with more sophisticated logic

            # If more clusters are found than tracked objects (shouldn't happen with min() above, but for safety)
            if len(centers) > self.n_tracked_objects:
                centers = centers[:self.n_tracked_objects]
            
            return centers

        except Exception as e:
            # Fallback if KMeans fails for any reason
            # print(f"KMeans clustering failed: {e}. Falling back to weighted mean for all objects.")
            if np.sum(self.weights) < 1e-10:
                weighted_mean_pos = np.mean(self.particles[:,:2], axis=0)
            else:
                weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)


# Simulator
class Simulator:
    def __init__(self, n_balls, dropout_probability=0.3, dropout_duration_steps_min=5, dropout_duration_steps_max=20):
        self.n_balls = n_balls
        self.balls = []
        self.estimates_raw = [[] for _ in range(n_balls)]
        self.estimates_processed_for_plot = [[] for _ in range(n_balls)]
        self.error_history = [[] for _ in range(n_balls)] # To store error over time
        self.dropout_probability = dropout_probability
        self.dropout_duration_steps_min = dropout_duration_steps_min
        self.dropout_duration_steps_max = dropout_duration_steps_max
        self.global_dropout_interval = None
        self.n_eff_history_pf = [] # To store N_eff from particle filter

    def _generate_global_dropout_interval(self, max_timesteps):
        if np.random.rand() < self.dropout_probability:
            min_start_t = int(max_timesteps * 0.1) 
            max_possible_start_t = max_timesteps - self.dropout_duration_steps_min - 1
            
            if max_possible_start_t <= min_start_t: 
                # Not enough room for a dropout
                return
            
            start_t = np.random.randint(min_start_t, max_possible_start_t + 1)
            duration = np.random.randint(self.dropout_duration_steps_min, self.dropout_duration_steps_max + 1)
            end_t = min(start_t + duration, max_timesteps - 1)

            self.global_dropout_interval = (start_t, end_t)
            print(f"Global sensor dropout: from step {start_t} to {end_t}")


    def run(self):
        # Initialize balls and ranges
        initial_x = []
        initial_y = []
        initial_vx = []
        initial_vy = []

        for _ in range(self.n_balls):
            ball = Ball(
                x0=np.random.uniform(0, 50),
                y0=np.random.uniform(0, 50),
                v0=np.random.uniform(15, 50),
                angle_deg=np.random.uniform(15, 60)
            )
            self.balls.append(ball)
            initial_x.append(ball.x0)
            initial_y.append(ball.y0)
            initial_vx.append(ball.vx0)
            initial_vy.append(ball.vy0)


        max_timesteps = max(len(ball.x_obs) for ball in self.balls)
        self._generate_global_dropout_interval(max_timesteps)

        # Define particle filter ranges based on initial ball properties
        x0_range = (min(initial_x) - 5, max(initial_x) + 5)
        y0_range = (0, max(initial_y) + 10)
        vx_range = (0, max(initial_vx) + 5)
        vy_range = (min(initial_vy) - 10, max(initial_vy) + 5) 

        pf = ParticleFilter(num_particles, x0_range, y0_range, vx_range, vy_range, self.n_balls)

        # Prepare for storing last known estimates for smooth tracking during dropouts
        last_known_estimates = [np.array([ball.x0, ball.y0]) for ball in self.balls]

        # Track whether a dropout is active at each timestep for plotting
        self.is_dropout_active_at_t = [False] * max_timesteps

        for t in range(max_timesteps):
            pf.predict()
            
            is_global_dropout_active = False
            if self.global_dropout_interval:
                start_t_dropout, end_t_dropout = self.global_dropout_interval
                if start_t_dropout <= t <= end_t_dropout:
                    is_global_dropout_active = True
            
            self.is_dropout_active_at_t[t] = is_global_dropout_active # Store for plotting

            current_obs = []
            active_ball_indices = []
            
            if not is_global_dropout_active:
                for i, ball in enumerate(self.balls):
                    if t < len(ball.x_obs):
                        current_obs.append((ball.x_obs[t], ball.y_obs[t]))
                        active_ball_indices.append(i)
                    # If ball has landed but not in dropout, we still "observe" it at its landing spot
                    elif t >= len(ball.x_obs):
                        current_obs.append((ball.x_true[-1], 0.0))
                        active_ball_indices.append(i)


            pf.update(current_obs)
            # Ensure n_eff_history_pf is populated, even if pf.n_eff_history might be empty initially
            if pf.n_eff_history:
                self.n_eff_history_pf.append(pf.n_eff_history[-1]) 
            else:
                self.n_eff_history_pf.append(0) # Or some default value if no N_eff calculated yet


            estimated_centers = pf.estimate_clusters(len(current_obs)) 

            # Map estimated centers to actual balls
            current_timestep_estimates = [None] * self.n_balls

            if len(current_obs) > 0 and len(estimated_centers) > 0:
                cost_matrix = np.linalg.norm(np.array(current_obs)[:, None, :2] - estimated_centers[None, :, :2], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for obs_idx, center_idx in zip(row_ind, col_ind):
                    original_ball_idx = active_ball_indices[obs_idx]
                    current_timestep_estimates[original_ball_idx] = estimated_centers[center_idx]
            
            # Fill in missing estimates (e.g., during dropout or if no observation for a ball)
            for k in range(self.n_balls):
                if current_timestep_estimates[k] is not None:
                    last_known_estimates[k] = current_timestep_estimates[k] # Update last known estimate
                else:
                    # If no new estimate, use the last known estimate
                    current_timestep_estimates[k] = last_known_estimates[k]

                self.estimates_raw[k].append(current_timestep_estimates[k])
                
                # Calculate error for this step
                true_pos = np.array([self.balls[k].x_true[min(t, len(self.balls[k].x_true) - 1)], 
                                     self.balls[k].y_true[min(t, len(self.balls[k].y_true) - 1)]])
                estimated_pos = np.array(current_timestep_estimates[k])
                self.error_history[k].append(np.linalg.norm(true_pos - estimated_pos))

            pf.resample()
        
        # Post-process raw estimates to make them stop at y=0 when the true ball lands
        for i, ball in enumerate(self.balls):
            true_trajectory_len = len(ball.x_true)
            processed_est_traj = []
            
            for t_step in range(max_timesteps):
                current_raw_estimate = self.estimates_raw[i][t_step]
                
                if t_step < true_trajectory_len:
                    processed_est_traj.append(current_raw_estimate)
                else:
                    # If true ball has landed, force estimated y to 0 and x to landed x
                    processed_est_traj.append([ball.x_true[-1], 0.0])

            self.estimates_processed_for_plot[i] = np.array(processed_est_traj)


    def plot(self):
        colors = plt.cm.get_cmap('viridis', self.n_balls) 
        max_timesteps = max(len(ball.x_true) for ball in self.balls)
        time_axis = np.arange(max_timesteps) * dt # Convert steps to time

        # --- Plot 1: X-Y Trajectories ---
        plt.figure(figsize=(10, 7))
        ax = plt.gca() # Get current axes for adding custom legend

        # Calculate the bounding box for the dropout period in X-Y plane
        if self.global_dropout_interval:
            start_t_dropout, end_t_dropout = self.global_dropout_interval
            
            all_x_in_dropout = []
            all_y_in_dropout = []

            for ball in self.balls:
                # Ensure the time steps are within the bounds of the ball's true trajectory
                safe_start_t = min(start_t_dropout, len(ball.x_true) - 1)
                safe_end_t = min(end_t_dropout, len(ball.x_true) - 1)
                
                if safe_start_t <= safe_end_t:
                    all_x_in_dropout.extend(ball.x_true[safe_start_t : safe_end_t + 1])
                    all_y_in_dropout.extend(ball.y_true[safe_start_t : safe_end_t + 1])
            
            if all_x_in_dropout and all_y_in_dropout:
                min_x_dropout = min(all_x_in_dropout)
                max_x_dropout = max(all_x_in_dropout)
                min_y_dropout = min(all_y_in_dropout)
                max_y_dropout = max(all_y_in_dropout)
                
                # Add the shaded area
                ax.axvspan(min_x_dropout, max_x_dropout, color='gray', alpha=0.2, lw=0, label='Global Sensor Dropout')


        has_labeled_dropout_segment = False # To add 'Estimated (Dropout)' to legend only once

        for i, ball in enumerate(self.balls):
            ball_true_traj = np.column_stack((ball.x_true, ball.y_true))
            ball_est_traj_plot = self.estimates_processed_for_plot[i]

            plt.plot(ball_true_traj[:, 0], ball_true_traj[:, 1], color=colors(i), 
                        linestyle='-', linewidth=2, label=f'Ball {i+1} True Trajectory')
            
            valid_obs_x = []
            valid_obs_y = []
            for t_step in range(len(ball.x_obs)):
                is_global_dropout_active_at_t = False
                if self.global_dropout_interval:
                    start_t_dropout, end_t_dropout = self.global_dropout_interval
                    if start_t_dropout <= t_step <= end_t_dropout:
                        is_global_dropout_active_at_t = True
                if not is_global_dropout_active_at_t:
                    valid_obs_x.append(ball.x_obs[t_step])
                    valid_obs_y.append(ball.y_obs[t_step])

            plt.scatter(valid_obs_x, valid_obs_y, s=10, alpha=0.4, 
                         color=colors(i), marker='x', label=f'Ball {i+1} Observations')
            
            # Plot estimated trajectory. Break into segments if dropout occurs.
            if self.global_dropout_interval:
                start_t_dropout, end_t_dropout = self.global_dropout_interval
                
                # Segment before dropout
                if start_t_dropout > 0:
                    plt.plot(ball_est_traj_plot[:start_t_dropout+1, 0], ball_est_traj_plot[:start_t_dropout+1, 1], 
                             color=colors(i), linestyle='--', linewidth=1.5)
                
                # Segment during dropout
                if start_t_dropout <= end_t_dropout and end_t_dropout < len(ball_est_traj_plot):
                    dropout_segment = ball_est_traj_plot[start_t_dropout:end_t_dropout+1]
                    # Only add label once for the legend
                    label_dropout_segment = 'Estimated Trajectory (Dropout)' if not has_labeled_dropout_segment else None
                    plt.plot(dropout_segment[:, 0], dropout_segment[:, 1], 
                             color=colors(i), linestyle=':', linewidth=1.5, label=label_dropout_segment) # Use dotted for dropout
                    has_labeled_dropout_segment = True
                
                # Segment after dropout
                if end_t_dropout + 1 < len(ball_est_traj_plot):
                    plt.plot(ball_est_traj_plot[end_t_dropout+1:, 0], ball_est_traj_plot[end_t_dropout+1:, 1], 
                             color=colors(i), linestyle='--', linewidth=1.5)
            else:
                # No dropout, plot entire estimated trajectory normally
                plt.plot(ball_est_traj_plot[:, 0], ball_est_traj_plot[:, 1], 
                         color=colors(i), linestyle='--', linewidth=1.5, label=f'Ball {i+1} Estimated Trajectory')
            
            min_len = min(len(ball_true_traj), len(ball_est_traj_plot))
            rmse = np.sqrt(mean_squared_error(ball_true_traj[:min_len], ball_est_traj_plot[:min_len]))
            print(f"Ball {i+1} RMSE (up to landing): {rmse:.2f}")

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Multi-Ball Tracking with Global Sensor Dropout (X-Y Plane)')
        
        # Create custom handles for the legend to avoid duplicate labels for each ball
        custom_lines_xy = [
            Line2D([0], [0], color='black', lw=2, linestyle='-'), 
            Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=5, alpha=0.4), 
            Line2D([0], [0], color='black', lw=1.5, linestyle='--'),
            Line2D([0], [0], color='black', lw=1.5, linestyle=':'), # New line for dropout segment
            Line2D([0], [0], color='gray', lw=8, alpha=0.2) # New line for the dropout region
        ]
        # Adjust legend labels
        plt.legend(custom_lines_xy, ['True Trajectory', 'Observations', 'Estimated Trajectory', 'Estimated (Dropout)', 'Global Sensor Dropout'], loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


#Run the simulation
sim = Simulator(n_balls=3, dropout_probability=1.0, dropout_duration_steps_min=5, dropout_duration_steps_max=20)
sim.run()
sim.plot()