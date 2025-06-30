import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error

# Constants
g = 9.81
dt = 0.1
num_particles = 10000
num_steps = 100
noise_std = 1.0

# Ball Class
class Ball:
    def __init__(self, x0, y0, v0, angle_deg):
        self.x0, self.y0, self.v0, self.angle_deg = x0, y0, v0, angle_deg
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
                break
            x.append(x_new)
            y.append(y_new)
        return np.array(x), np.array(y)

    def _add_noise(self):
        x_obs = self.x_true + np.random.normal(0, noise_std, size=self.x_true.shape)
        y_obs = self.y_true + np.random.normal(0, noise_std, size=self.y_true.shape)
        return x_obs, y_obs


# Particle Filter Class
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
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        self.particles[:, 3] -= g * dt
        # self.particles[:, :4] += np.random.normal(0, self.adaptive_noise(), self.particles[:, :4].shape)

    # def adaptive_noise(self):
    #     # Example: increase noise if particle weights are too uniform (indicating uncertainty)
    #     entropy = -np.sum(self.weights * np.log(self.weights + 1e-10))
    #     max_entropy = np.log(self.n)
    #     return 0.05 + 0.5 * (entropy / max_entropy)

    #problems 1
    def update(self, observations):
        obs_array = np.array(observations)
        dists = np.linalg.norm(self.particles[:, None, :2] - obs_array[None, :, :], axis=2)  # shape (N, B)
        weights = np.exp(-np.min(dists, axis=1)**2 / (2 * noise_std ** 2))
        self.weights = weights / np.sum(weights)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        positions = (np.arange(self.n) + np.random.uniform()) / self.n
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
        gmm = GaussianMixture(n_components=n_clusters, n_init=5, random_state=42)
        gmm.fit(self.particles[:, :4])  # Use [x, y, vx, vy]
        return gmm.means_[:, :2]  # Return only positions for plotting



# Simulator
class Simulator:
    def __init__(self, n_balls):
        self.n_balls = n_balls
        self.balls = []
        self.estimates_all = [[] for _ in range(n_balls)]

    def run(self):
        for _ in range(self.n_balls):
            ball = Ball(
                x0=np.random.uniform(5, 30),
                y0=np.random.uniform(0, 15),
                v0=np.random.uniform(20, 50),
                angle_deg=np.random.uniform(30, 60)
            )
            self.balls.append(ball)

        min_timesteps = min(len(ball.x_obs) for ball in self.balls)

        x0_range = (min(ball.x0 for ball in self.balls) - 5, max(ball.x0 for ball in self.balls) + 5)
        y0_range = (0, max(ball.y0 for ball in self.balls) + 10)
        vx_range = (0, max(ball.vx0 for ball in self.balls) + 5)
        vy_range = (0, max(ball.vy0 for ball in self.balls) + 5)

        pf = ParticleFilter(num_particles, x0_range, y0_range, vx_range, vy_range)

        for t in range(min_timesteps):
            pf.predict()

            obs = [(ball.x_obs[t], ball.y_obs[t]) for ball in self.balls]
            pf.update(obs)

            centers = pf.estimate_clusters(n_clusters=self.n_balls)

            cost_matrix = np.linalg.norm(
                np.array(obs)[:, None, :] - centers[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for obs_idx, center_idx in zip(row_ind, col_ind):
                self.estimates_all[obs_idx].append(centers[center_idx])

            pf.resample()

    def plot(self):
        plt.figure(figsize=(12, 8))

        for i, ball in enumerate(self.balls):
            plt.plot(ball.x_true, ball.y_true, color='blue', label='True' if i == 0 else "")
            plt.scatter(ball.x_obs, ball.y_obs, s=10, alpha=0.4, color='black')
            est = np.array(self.estimates_all[i])
            plt.plot(est[:, 0], est[:, 1], '--', color='red', label='Estimated' if i == 0 else "")

            min_len = min(len(est), len(ball.x_true))
            true = np.array([ball.x_true[:min_len], ball.y_true[:min_len]]).T
            est = est[:min_len]
            rmse = np.sqrt(mean_squared_error(true, est))
            print(f"Ball {i+1} RMSE: {rmse:.2f}")

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Multi-Ball Tracking with Particle Filter')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Run Simulation
if __name__ == "__main__":
    sim = Simulator(n_balls=2)
    sim.run()
    sim.plot()
