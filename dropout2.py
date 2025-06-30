import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Constants
g = 9.81 # Acceleration due to gravity (m/s^2)
dt = 0.1 # Time step for simulation (seconds)
num_particles = 20000 # Number of particles in the particle filter
num_steps = 100 # Maximum simulation steps for a single ball's trajectory, actual steps might be fewer if it lands
noise_std = 0.5 # Standard deviation of observation noise (meters)

# Ball class to simulate and store trajectory
class Ball:
    def __init__(self, x0, y0, v0, angle_deg):
        self.x0 = x0 # Initial x position
        self.y0 = y0 # Initial y position
        self.v0 = v0 # Initial velocity magnitude
        self.angle_deg = angle_deg # Initial launch angle in degrees
        self.angle_rad = np.deg2rad(angle_deg) # Convert angle to radians
        self.vx0 = v0 * np.cos(self.angle_rad) # Initial x-component of velocity
        self.vy0 = v0 * np.sin(self.angle_rad) # Initial y-component of velocity
        self.x_true, self.y_true = self._simulate_trajectory() # Simulate the true trajectory
        self.x_obs, self.y_obs = self._add_noise() # Add noise to true trajectory to get observations

    def _simulate_trajectory(self):
        x, y = [self.x0], [self.y0] # Initialize trajectory lists with starting position
        vx, vy = self.vx0, self.vy0 # Initialize velocities
        for _ in range(num_steps): # Loop for maximum simulation steps
            vy -= g * dt # Update y-velocity due to gravity
            x_new = x[-1] + vx * dt # Calculate new x position
            y_new = y[-1] + vy * dt # Calculate new y position
            if y_new < 0: # Check if the ball has hit or gone below ground (y=0)
                if y[-1] > 0: # If the previous y was above ground, interpolate
                    # Calculate the fraction of dt at which it crosses y=0
                    fraction = -y[-1] / (vy - g * dt)
                    x_new = x[-1] + vx * dt * fraction # Interpolate x position
                y_new = 0.0 # Set y position to exactly 0 (ground level)
                x.append(x_new) # Add the interpolated landing x
                y.append(y_new) # Add 0 for y
                break # Stop simulation as the ball has landed
            x.append(x_new) # Add new x position to trajectory
            y.append(y_new) # Add new y position to trajectory
        return np.array(x), np.array(y) # Return true trajectory as numpy arrays

    def _add_noise(self):
        # Add Gaussian noise to true x and y positions
        x_obs = self.x_true + np.random.normal(0, noise_std, size=self.x_true.shape)
        y_obs = self.y_true + np.random.normal(0, noise_std, size=self.y_true.shape)
        return x_obs, y_obs # Return noisy observations

# Particle Filter
class ParticleFilter:
    def __init__(self, num_particles, x0_range, y0_range, vx_range, vy_range, n_tracked_objects):
        self.n = num_particles # Number of particles
        self.particles = self._init_particles(x0_range, y0_range, vx_range, vy_range) # Initialize particles
        self.weights = np.ones(self.n) / self.n # Initialize uniform weights for all particles
        self.n_tracked_objects = n_tracked_objects # Number of objects the filter is tracking
        self.n_eff_history = [] # List to store effective number of particles over time

    def _init_particles(self, x0_range, y0_range, vx_range, vy_range):
        p = np.empty((self.n, 4)) # Create an empty array for particles (x, y, vx, vy)
        # Randomly initialize particles within the given ranges
        p[:, 0] = np.random.uniform(*x0_range, size=self.n) # Initial x
        p[:, 1] = np.random.uniform(*y0_range, size=self.n) # Initial y
        p[:, 2] = np.random.uniform(*vx_range, size=self.n) # Initial vx
        p[:, 3] = np.random.uniform(*vy_range, size=self.n) # Initial vy
        return p

    def predict(self):
        # Propagate particles based on physics (constant velocity in x, gravity in y)
        self.particles[:, 0] += self.particles[:, 2] * dt # x_new = x_old + vx * dt
        self.particles[:, 1] += self.particles[:, 3] * dt # y_new = y_old + vy * dt
        self.particles[:, 3] -= g * dt # vy_new = vy_old - g * dt (x-velocity is constant)
        
        # Add adaptive noise to particles to introduce diversity
        noise_levels = self.adaptive_noise()
        self.particles += np.random.normal(0, noise_levels, self.particles.shape)
        
        # Ensure particles don't go below ground (y >= 0)
        self.particles[self.particles[:, 1] < 0, 1] = 0.0

    def adaptive_noise(self):
        # Calculate entropy of weights to determine uncertainty
        # Add a small epsilon to weights to avoid log(0)
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-10))
        max_entropy = np.log(self.n) # Maximum possible entropy (when weights are uniform)
        
        # Normalize entropy to a value between 0 and 1
        # Add epsilon to max_entropy to handle edge cases like n=1
        normalized_entropy = entropy / (max_entropy + 1e-10) 

        # Base noise levels for position and velocity components
        base_pos_noise = 0.05 
        base_vel_noise = 0.02 

        # Additional noise proportional to normalized entropy (higher uncertainty -> more noise)
        additional_pos_noise = 0.5 * normalized_entropy 
        additional_vel_noise = 0.2 * normalized_entropy 

        # Return an array of noise standard deviations for [x, y, vx, vy]
        return np.array([base_pos_noise + additional_pos_noise, 
                         base_pos_noise + additional_pos_noise, 
                         base_vel_noise + additional_vel_noise, 
                         base_vel_noise + additional_vel_noise])

    def update(self, observations):
        # Convert list of observations (tuples) to a numpy array for efficient computation
        obs_array = np.array(observations)
        
        if obs_array.shape[0] > 0: # Proceed only if there are observations
            # Calculate Euclidean distance from each particle's position to each observation
            # self.particles[:, None, :2] -> (n_particles, 1, 2)
            # obs_array[None, :, :] -> (1, n_observations, 2)
            # Broadcasting results in (n_particles, n_observations, 2) for difference
            dists = np.linalg.norm(self.particles[:, None, :2] - obs_array[None, :, :], axis=2)
            
            # For each particle, find the minimum distance to any observation
            min_dists = np.min(dists, axis=1)
            
            # Update weights: particles closer to observations get higher weights
            # Using a Gaussian likelihood model: exp(-distance^2 / (2 * noise_std^2))
            weights = np.exp(-min_dists**2 / (2 * noise_std**2))
            
            # Normalize weights to sum to 1
            sum_weights = np.sum(weights)
            if sum_weights > 1e-10: # Avoid division by zero if all weights are extremely small
                self.weights = weights / sum_weights
            else:
                self.weights = np.ones(self.n) / self.n # Reset to uniform if all weights are negligible

        # Calculate Effective Number of Particles (N_eff)
        # N_eff = 1 / sum(weights^2). Lower N_eff indicates weight degeneracy.
        sum_of_squared_weights = np.sum(np.square(self.weights))
        N_eff = 1.0 / sum_of_squared_weights if sum_of_squared_weights > 1e-10 else 0
        self.n_eff_history.append(N_eff) # Store N_eff

    def resample(self):
        # Handle case where weights might sum to zero (e.g., after prolonged dropout)
        if np.sum(self.weights) < 1e-10:
            self.weights = np.ones(self.n) / self.n # Reset to uniform weights

        # Systematic Resampling: A low-variance resampling method
        positions = (np.arange(self.n) + np.random.rand()) / self.n # Generate evenly spaced points with a random offset
        indexes = np.zeros(self.n, dtype=int) # Array to store indices of resampled particles
        cumulative_sum = np.cumsum(self.weights) # Cumulative sum of weights
        i, j = 0, 0
        while i < self.n: # Loop through the desired number of resampled particles
            if j >= self.n: # Safety check to prevent index out of bounds
                j = self.n - 1 
            
            if positions[i] < cumulative_sum[j]: # If current position is within current cumulative weight segment
                indexes[i] = j # Select the particle corresponding to this segment
                i += 1 # Move to the next desired resampled particle
            else:
                j += 1 # Move to the next particle in the original set
        self.particles = self.particles[indexes] # Create new particle set using selected indices
        self.weights = np.ones(self.n) / self.n # Reset weights to uniform after resampling

    def estimate_clusters(self, n_observations_present):
        # If no observations were made at this timestep
        if n_observations_present == 0:
            # If weights are practically zero (degeneracy), use simple mean of particle positions
            if np.sum(self.weights) < 1e-10:
                weighted_mean_pos = np.mean(self.particles[:,:2], axis=0) 
            else:
                # Otherwise, use the weighted mean of particle positions
                weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights)
            # Return this single mean replicated for all tracked objects (since no individual observation)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)

        # Determine the number of clusters for KMeans
        # It should not exceed the number of observed objects or unique particle positions
        unique_particles_pos = np.unique(self.particles[:,:2], axis=0)
        num_clusters_for_kmeans = min(self.n_tracked_objects, n_observations_present, len(unique_particles_pos))
        
        # Ensure at least one cluster if there are particles
        if num_clusters_for_kmeans == 0 and len(unique_particles_pos) > 0:
            num_clusters_for_kmeans = 1
        elif num_clusters_for_kmeans == 0: # If no unique particles, return replicated mean
            weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights) if np.sum(self.weights) > 1e-10 else np.mean(self.particles[:,:2], axis=0)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)

        try:
            # Apply K-Means clustering to particle positions using their weights
            # n_init=10 runs K-Means 10 times with different centroids and picks the best
            kmeans = KMeans(n_clusters=num_clusters_for_kmeans, n_init=10, random_state=42, max_iter=300) 
            kmeans.fit(self.particles[:, :2], sample_weight=self.weights)
            centers = kmeans.cluster_centers_ # Get the cluster centers (estimated object positions)

            # Pad with duplicates if fewer clusters are found than tracked objects (e.g., during partial dropout)
            while len(centers) < self.n_tracked_objects:
                centers = np.vstack([centers, centers[0]]) # Simple padding
            
            # Trim if more clusters are found (should not happen with min() logic, but for safety)
            if len(centers) > self.n_tracked_objects:
                centers = centers[:self.n_tracked_objects]
            
            return centers

        except Exception as e:
            # Fallback if KMeans fails (e.g., not enough unique samples for required clusters)
            # print(f"KMeans clustering failed: {e}. Falling back to weighted mean for all objects.")
            if np.sum(self.weights) < 1e-10:
                weighted_mean_pos = np.mean(self.particles[:,:2], axis=0)
            else:
                weighted_mean_pos = np.average(self.particles[:, :2], axis=0, weights=self.weights)
            return np.array([weighted_mean_pos] * self.n_tracked_objects)

# Simulator
class Simulator:
    def __init__(self, n_balls, dropout_probability=0.3, dropout_duration_steps_min=5, dropout_duration_steps_max=20):
        self.n_balls = n_balls # Number of balls to simulate
        self.balls = [] # List to store Ball objects
        self.estimates_raw = [[] for _ in range(n_balls)] # Raw estimated positions
        self.estimates_processed_for_plot = [[] for _ in range(n_balls)] # Estimates adjusted for plotting
        self.error_history = [[] for _ in range(n_balls)] # List to store tracking error over time
        self.dropout_probability = dropout_probability # Probability of a global sensor dropout occurring
        self.dropout_duration_steps_min = dropout_duration_steps_min # Minimum duration of dropout
        self.dropout_duration_steps_max = dropout_duration_steps_max # Maximum duration of dropout
        self.global_dropout_interval = None # Stores (start_step, end_step) of dropout if active
        self.n_eff_history_pf = [] # To store N_eff from particle filter (copied from PF)

    def _generate_global_dropout_interval(self, max_timesteps):
        # Decide if a global dropout will occur based on probability
        if np.random.rand() < self.dropout_probability:
            # Define safe range for dropout start to ensure it fits
            min_start_t = int(max_timesteps * 0.1) 
            max_possible_start_t = max_timesteps - self.dropout_duration_steps_min - 1
            
            if max_possible_start_t <= min_start_t: # Not enough room for a dropout
                return
            
            # Randomly select start time and duration for the dropout
            start_t = np.random.randint(min_start_t, max_possible_start_t + 1)
            duration = np.random.randint(self.dropout_duration_steps_min, self.dropout_duration_steps_max + 1)
            end_t = min(start_t + duration, max_timesteps - 1) # Ensure dropout ends within simulation limits

            self.global_dropout_interval = (start_t, end_t)
            print(f"Global sensor dropout: from step {start_t} to {end_t}")

    def run(self):
        # Initialize balls with random initial conditions
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

        # Determine the maximum simulation time based on the longest ball trajectory
        max_timesteps = max(len(ball.x_obs) for ball in self.balls)
        # Generate a global dropout interval if applicable
        self._generate_global_dropout_interval(max_timesteps)

        # Define particle filter ranges based on initial ball properties (with some buffer)
        x0_range = (min(initial_x) - 5, max(initial_x) + 5)
        y0_range = (0, max(initial_y) + 10) # Y starts from 0 (ground)
        vx_range = (0, max(initial_vx) + 5) 
        vy_range = (min(initial_vy) - 10, max(initial_vy) + 5) 

        # Initialize the Particle Filter
        pf = ParticleFilter(num_particles, x0_range, y0_range, vx_range, vy_range, self.n_balls)

        # Stores the last known estimated positions for each ball to use during dropouts
        last_known_estimates = [np.array([ball.x0, ball.y0]) for ball in self.balls]

        # Track dropout status at each timestep for plotting
        self.is_dropout_active_at_t = [False] * max_timesteps

        # Main simulation loop
        for t in range(max_timesteps):
            pf.predict() # Predict particle positions

            is_global_dropout_active = False
            if self.global_dropout_interval:
                start_t_dropout, end_t_dropout = self.global_dropout_interval
                if start_t_dropout <= t <= end_t_dropout:
                    is_global_dropout_active = True
            
            self.is_dropout_active_at_t[t] = is_global_dropout_active # Store dropout status

            current_obs = [] # Observations for the current timestep
            active_ball_indices = [] # Indices of balls from which observations are available
            
            if not is_global_dropout_active: # If sensor is active (no dropout)
                for i, ball in enumerate(self.balls):
                    if t < len(ball.x_obs): # If ball is still in air and producing observations
                        current_obs.append((ball.x_obs[t], ball.y_obs[t]))
                        active_ball_indices.append(i)
                    elif t >= len(ball.x_obs): # If ball has landed, "observe" it at its landing spot
                        current_obs.append((ball.x_true[-1], 0.0))
                        active_ball_indices.append(i)

            pf.update(current_obs) # Update particle weights based on observations
            
            # Store N_eff from particle filter
            if pf.n_eff_history:
                self.n_eff_history_pf.append(pf.n_eff_history[-1]) 
            else:
                self.n_eff_history_pf.append(0) 

            # Estimate object positions using K-Means on particles
            estimated_centers = pf.estimate_clusters(len(current_obs)) 

            current_timestep_estimates = [None] * self.n_balls # Initialize estimates for current step

            # If there are observations and estimated centers, perform data association
            if len(current_obs) > 0 and len(estimated_centers) > 0:
                # Create a cost matrix for matching observations to estimated centers
                # Uses linear_sum_assignment (Hungarian Algorithm) for optimal assignment
                cost_matrix = np.linalg.norm(np.array(current_obs)[:, None, :2] - estimated_centers[None, :, :2], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Assign estimated centers to their corresponding true ball indices
                for obs_idx, center_idx in zip(row_ind, col_ind):
                    original_ball_idx = active_ball_indices[obs_idx]
                    current_timestep_estimates[original_ball_idx] = estimated_centers[center_idx]
            
            # Fill in missing estimates (e.g., for balls that didn't have observations or during dropout)
            for k in range(self.n_balls):
                if current_timestep_estimates[k] is not None:
                    last_known_estimates[k] = current_timestep_estimates[k] # Update last known estimate
                else:
                    current_timestep_estimates[k] = last_known_estimates[k] # Use last known estimate

                self.estimates_raw[k].append(current_timestep_estimates[k])
                
                # Calculate and store the error between true and estimated positions
                true_pos = np.array([self.balls[k].x_true[min(t, len(self.balls[k].x_true) - 1)], 
                                     self.balls[k].x_true[min(t, len(self.balls[k].x_true) - 1)]])
                estimated_pos = np.array(current_timestep_estimates[k])
                self.error_history[k].append(np.linalg.norm(true_pos - estimated_pos))

            pf.resample() # Resample particles for the next iteration
        
        # Post-process raw estimates: Ensure estimated trajectory stops at y=0 when the true ball lands
        for i, ball in enumerate(self.balls):
            true_trajectory_len = len(ball.x_true)
            processed_est_traj = []
            
            for t_step in range(max_timesteps):
                current_raw_estimate = self.estimates_raw[i][t_step]
                
                if t_step < true_trajectory_len:
                    processed_est_traj.append(current_raw_estimate)
                else:
                    # If true ball has landed, force estimated y to 0 and x to true landed x
                    processed_est_traj.append([ball.x_true[-1], 0.0])

            self.estimates_processed_for_plot[i] = np.array(processed_est_traj)

    def plot(self):
        colors = plt.cm.get_cmap('viridis', self.n_balls) # Colormap for different balls
        max_timesteps = max(len(ball.x_true) for ball in self.balls)
        time_axis = np.arange(max_timesteps) * dt # Time axis for plots

        # --- Plot 1: X-Y Trajectories ---
        plt.figure(figsize=(10, 7))
        ax = plt.gca() # Get current axes for adding custom legend elements

        # Calculate bounding box for the dropout period in X-Y plane
        if self.global_dropout_interval:
            start_t_dropout, end_t_dropout = self.global_dropout_interval
            
            all_x_in_dropout = []
            all_y_in_dropout = []

            for ball in self.balls:
                # Ensure time steps are within bounds of the ball's true trajectory
                safe_start_t = min(start_t_dropout, len(ball.x_true) - 1)
                safe_end_t = min(end_t_dropout, len(ball.x_true) - 1)
                
                if safe_start_t <= safe_end_t:
                    # Collect true x and y values during the dropout interval
                    all_x_in_dropout.extend(ball.x_true[safe_start_t : safe_end_t + 1])
                    all_y_in_dropout.extend(ball.y_true[safe_start_t : safe_end_t + 1])
            
            if all_x_in_dropout and all_y_in_dropout:
                min_x_dropout = min(all_x_in_dropout)
                max_x_dropout = max(all_x_in_dropout)
                min_y_dropout = min(all_y_in_dropout)
                max_y_dropout = max(all_y_in_dropout)
                
                # Add a shaded rectangle to represent the global sensor dropout region
                ax.axvspan(min_x_dropout, max_x_dropout, color='gray', alpha=0.2, lw=0, label='Global Sensor Dropout')

        has_labeled_dropout_segment = False # Flag to add 'Estimated (Dropout)' label to legend only once

        for i, ball in enumerate(self.balls):
            ball_true_traj = np.column_stack((ball.x_true, ball.y_true))
            ball_est_traj_plot = self.estimates_processed_for_plot[i]

            # Plot true trajectory for each ball
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
                if not is_global_dropout_active_at_t: # Only plot observations if not in dropout
                    valid_obs_x.append(ball.x_obs[t_step])
                    valid_obs_y.append(ball.y_obs[t_step])

            # Plot noisy observations (only those not during dropout)
            plt.scatter(valid_obs_x, valid_obs_y, s=10, alpha=0.4, 
                        color=colors(i), marker='x', label=f'Ball {i+1} Observations')
            
            # Plot estimated trajectory, differentiating between normal and dropout periods
            if self.global_dropout_interval:
                start_t_dropout, end_t_dropout = self.global_dropout_interval
                
                # Segment before dropout
                if start_t_dropout > 0:
                    plt.plot(ball_est_traj_plot[:start_t_dropout+1, 0], ball_est_traj_plot[:start_t_dropout+1, 1], 
                             color=colors(i), linestyle='--', linewidth=1.5)
                
                # Segment during dropout (dotted line)
                if start_t_dropout <= end_t_dropout and end_t_dropout < len(ball_est_traj_plot):
                    dropout_segment = ball_est_traj_plot[start_t_dropout:end_t_dropout+1]
                    label_dropout_segment = 'Estimated Trajectory (Dropout)' if not has_labeled_dropout_segment else None
                    plt.plot(dropout_segment[:, 0], dropout_segment[:, 1], 
                             color=colors(i), linestyle=':', linewidth=1.5, label=label_dropout_segment) 
                    has_labeled_dropout_segment = True # Set flag to prevent duplicate label
                
                # Segment after dropout
                if end_t_dropout + 1 < len(ball_est_traj_plot):
                    plt.plot(ball_est_traj_plot[end_t_dropout+1:, 0], ball_est_traj_plot[end_t_dropout+1:, 1], 
                             color=colors(i), linestyle='--', linewidth=1.5)
            else:
                # No dropout, plot entire estimated trajectory normally
                plt.plot(ball_est_traj_plot[:, 0], ball_est_traj_plot[:, 1], 
                         color=colors(i), linestyle='--', linewidth=1.5, label=f'Ball {i+1} Estimated Trajectory')
            
            # Calculate and print RMSE for each ball up to its landing point
            min_len = min(len(ball_true_traj), len(ball_est_traj_plot))
            rmse = np.sqrt(mean_squared_error(ball_true_traj[:min_len], ball_est_traj_plot[:min_len]))
            print(f"Ball {i+1} RMSE (up to landing): {rmse:.2f}")

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Multi-Ball Tracking with Global Sensor Dropout (X-Y Plane)')
        
        # Create custom handles for the legend to avoid repetitive labels for each ball
        custom_lines_xy = [
            Line2D([0], [0], color='black', lw=2, linestyle='-'), # True trajectory
            Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=5, alpha=0.4), # Observations
            Line2D([0], [0], color='black', lw=1.5, linestyle=':'), # Estimated (dropout)
            Line2D([0], [0], color='gray', lw=8, alpha=0.2) # Dropout region
        ]
        plt.legend(custom_lines_xy, ['True Trajectory', 'Observations', 'Estimated Trajectory', 'Global Sensor Dropout'], loc='upper left')
        plt.tight_layout()
        plt.show()

#Run the simulation
# n_balls=3: Simulate 3 balls
# dropout_probability=1.0: Ensure a global sensor dropout ALWAYS occurs
# dropout_duration_steps_min=5, dropout_duration_steps_max=20: Define dropout length
sim = Simulator(n_balls=3, dropout_probability=0.5, dropout_duration_steps_min=5, dropout_duration_steps_max=20)
sim.run() # Run the simulation
sim.plot() # Generate the plots