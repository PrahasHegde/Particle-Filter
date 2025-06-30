import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Constants
g = 9.81  # gravity (m/s^2)
dt = 0.1  # time step (s)
num_particles = 1000
num_steps = 100
noise_std = 2.0


#Ball Class
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
    
# Particle Filter Class
class ParticleFilter:
    def __init__(self, num_particles, x0_range, y0_range, vx_range, vy_range, n_clusters=1):
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

    def update(self,obs):
        distances = np.linalg.norm(self.particles[:, :2] - obs, axis=1)
        self.weights = np.exp(-distances**2 / (2 * noise_std**2))
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.n, self.n, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n) / self.n


    def estimate_clusters(self):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(self.particles[:, :2])
        return kmeans.cluster_centers_
    

#clustering simulator
class Simulator:
    def __init__(self, n_balls, n_clusters=1):
        self.n_balls = n_balls
        self.n_clusters = n_clusters
        self.balls = []
        self.estimate_all = []


    def run(self):
        for _ in range(self.n_balls):
            ball = Ball(x0=np.random.uniform(5, 30), y0=np.random.uniform(0, 15),
                        v0=np.random.uniform(20, 50), angle_deg=np.random.uniform(30, 60))
            self.balls.append(ball)



        min_timesteps = min(len(ball.x_obs) for ball in self.balls)

        x0_range = (min(ball.x0 for ball in self.balls) - 5, max(ball.x0 for ball in self.balls) + 5)
        y0_range = (0, max(ball.y0 for ball in self.balls) + 10)
        vx_range = (0, max(ball.vx0 for ball in self.balls) + 5)
        vy_range = (0, max(ball.vy0 for ball in self.balls) + 5)

        pf = ParticleFilter(
            num_particles,
            x0_range=x0_range,
            y0_range=y0_range,
            vx_range=vx_range,
            vy_range=vy_range,
            n_clusters=self.n_balls
        )

        self.estimates_all = [[] for _ in range(self.n_balls)]

        for t in range(min_timesteps):
            pf.predict()

            obs = [(ball.x_obs[t], ball.y_obs[t]) for ball in self.balls]
            avg_obs = np.mean(obs, axis=0)
            pf.update(avg_obs)

            centers = pf.estimate_clusters()

            for i, ball in enumerate(self.balls):
                dists = np.linalg.norm(centers - np.array(obs[i]), axis=1)
                best_cluster_idx = np.argmin(dists)
                self.estimates_all[i].append(centers[best_cluster_idx])
            pf.resample()

    def plot(self):
        plt.figure(figsize=(12, 8))
        for i, ball in enumerate(self.balls):
            plt.plot(ball.x_true, ball.y_true, label=f'Ball {i+1} True Trajectory', color='blue')
            plt.scatter(ball.x_obs, ball.y_obs, label=f'Ball {i+1} Observations', s=10, alpha=0.5)
            estimates = np.array(self.estimates_all[i])
            plt.plot(estimates[:, 0], estimates[:, 1], label=f'Ball {i+1} Estimated Trajectory', linestyle='--')

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Particle Filter Ball Trajectories')
        plt.legend()
        plt.grid()
        plt.show()



# Run the Simulation
if __name__ == "__main__":
    sim = Simulator(n_balls=2, n_clusters=1)
    sim.run()
    sim.plot()


