import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Constants
g = 9.81
dt = 0.1
num_particles = 1000
num_steps = 100
noise_std = 2.0

# -------------------------------
# Ball class
# -------------------------------
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
                break
            x.append(x_new)
            y.append(y_new)
        return np.array(x), np.array(y)

    def _add_noise(self):
        x_obs = self.x_true + np.random.normal(0, noise_std, size=self.x_true.shape)
        y_obs = self.y_true + np.random.normal(0, noise_std, size=self.y_true.shape)
        return x_obs, y_obs

# -------------------------------
# Particle Filter with Clustering
# -------------------------------
class ParticleFilter:
    def __init__(self, num_particles, x0_range, y0_range, vx_range, vy_range, n_clusters=2):
        self.n = num_particles
        self.n_clusters = n_clusters
        self.particles = self._init_particles(x0_range, y0_range, vx_range, vy_range)
        self.weights = np.ones(self.n) / self.n

    def _init_particles(self, x0_range, y0_range, vx_range, vy_range):
        p = np.empty((self.n, 4))
        p[:, 0] = np.random.uniform(*x0_range, size=self.n)  # x
        p[:, 1] = np.random.uniform(*y0_range, size=self.n)  # y
        p[:, 2] = np.random.uniform(*vx_range, size=self.n)  # vx
        p[:, 3] = np.random.uniform(*vy_range, size=self.n)  # vy
        return p

    def predict(self):
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        self.particles[:, 3] -= g * dt

    def update(self, obs):
        distances = np.linalg.norm(self.particles[:, :2] - obs, axis=1)
        self.weights = np.exp(-distances**2 / 4.0)
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.n, self.n, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.n)

    def estimate_clusters(self):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(self.particles[:, :2], sample_weight=self.weights)
        return kmeans.cluster_centers_

# -------------------------------
# Simulator with Clustered Estimation
# -------------------------------
class Simulator:
    def __init__(self, n_balls, n_clusters=2):
        self.n_balls = n_balls
        self.n_clusters = n_clusters
        self.balls = []
        self.estimates_all = []

    def run(self):
        for _ in range(self.n_balls):
            x0 = np.random.uniform(5, 20)
            y0 = np.random.uniform(0, 5)
            v0 = np.random.uniform(20, 40)
            angle = np.random.uniform(30, 60)

            ball = Ball(x0, y0, v0, angle)
            self.balls.append(ball)

            pf = ParticleFilter(
                num_particles,
                x0_range=(x0 - 5, x0 + 5),
                y0_range=(y0 - 5, y0 + 5),
                vx_range=(ball.vx0 - 5, ball.vx0 + 5),
                vy_range=(ball.vy0 - 5, ball.vy0 + 5),
                n_clusters=self.n_clusters
            )

            estimates = []
            for t in range(len(ball.x_obs)):
                pf.predict()
                pf.update(np.array([ball.x_obs[t], ball.y_obs[t]]))
                centers = pf.estimate_clusters()
                estimates.append(centers)
                pf.resample()
            self.estimates_all.append(estimates)

    def plot(self):
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'orange', 'green', 'purple', 'magenta', 'brown']
        for i, ball in enumerate(self.balls):
            color = colors[i % len(colors)]
            plt.plot(ball.x_true, ball.y_true, color=color, label=f"True Path {i+1}")
            plt.scatter(ball.x_obs, ball.y_obs, color=color, s=10, alpha=0.4, label=f"Obs {i+1}")

            # For each cluster, plot separate estimate trajectory
            estimates = np.array(self.estimates_all[i])
            for cluster_id in range(self.n_clusters):
                cluster_traj = estimates[:, cluster_id, :]
                plt.plot(cluster_traj[:, 0], cluster_traj[:, 1], '--', label=f"Est {i+1}-C{cluster_id+1}", alpha=0.7)

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Multi-Ball Tracking with Clustering Particle Filter")
        plt.legend()
        plt.tight_layout()
        plt.show()

# -------------------------------
# Run the Simulation
# -------------------------------
if __name__ == "__main__":
    sim = Simulator(n_balls=5, n_clusters=1) 
    sim.run()
    sim.plot()
