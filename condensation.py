import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== Particle Filter for a Single Ball ==========
class ParticleFilter:
    def __init__(self, pos, vel, num_particles=300, dt=0.1, gravity=9.81, process_noise=0.5, obs_noise=2.0):
        self.dt, self.g, self.pn, self.on = dt, gravity, process_noise, obs_noise
        self.particles = np.empty((num_particles, 4))
        self.particles[:, 0] = np.random.normal(pos[0], 1.0, num_particles)  # x
        self.particles[:, 1] = np.random.normal(pos[1], 1.0, num_particles)  # y
        self.particles[:, 2] = np.random.normal(vel[0], 1.0, num_particles)  # vx
        self.particles[:, 3] = np.random.normal(vel[1], 1.0, num_particles)  # vy
        self.true_pos = np.array(pos, dtype=float)
        self.true_vel = np.array(vel, dtype=float)
        self.true_path, self.est_path = [], []
        self.landed = False  # Flag to indicate if ball hit the ground

    def step(self):
        if not self.landed:
            self.true_vel[1] -= self.g * self.dt
            self.true_pos += self.true_vel * self.dt
            if self.true_pos[1] <= 0:
                self.true_pos[1] = 0
                self.true_vel[1] = 0
                self.landed = True
            self.true_path.append(self.true_pos.copy())

            obs = self.true_pos + np.random.randn(2) * self.on

            self.particles[:, 0] += self.particles[:, 2] * self.dt
            self.particles[:, 1] += self.particles[:, 3] * self.dt - 0.5 * self.g * self.dt**2
            self.particles[:, 3] -= self.g * self.dt
            self.particles[:, 1] = np.maximum(self.particles[:, 1], 0)
            self.particles[:, :2] += np.random.randn(*self.particles[:, :2].shape) * self.pn

            d = self.particles[:, :2] - obs
            weights = np.exp(-0.5 * np.sum(d**2, axis=1) / self.on**2)
            weights += 1e-300
            weights /= np.sum(weights)

            indices = np.searchsorted(np.cumsum(weights), np.random.rand(len(weights)))
            self.particles[:] = self.particles[indices]

            estimate = np.average(self.particles[:, :2], weights=weights, axis=0)
            self.est_path.append(estimate)
            return self.particles[:, :2], estimate
        else:
            self.true_path.append(self.true_pos.copy())
            last_est = self.est_path[-1] if self.est_path else self.true_pos.copy()
            self.est_path.append(last_est)
            return self.particles[:, :2], last_est


# ========== Multi-Ball Tracker ==========
class MultiBallTracker:
    def __init__(self, num_balls):
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.balls = [ParticleFilter(
            pos=[np.random.uniform(0, 30), np.random.uniform(10, 30)],
            vel=[np.random.uniform(5, 20), np.random.uniform(5, 20)]
        ) for _ in range(num_balls)]

    def step_all(self):
        return [ball.step() for ball in self.balls]

    def all_landed(self):
        return all(ball.landed for ball in self.balls)


# ========== Animation ==========
def animate_balls(num_balls=3, max_steps=300):
    tracker = MultiBallTracker(num_balls)
    fig, ax = plt.subplots()
    ax.set_title("Multi-Ball Particle Filter")

    # Fixed axis limits to show entire trajectory (rough estimate)
    ax.set_xlim(0, 100)  # fixed width to show all horizontal motion
    ax.set_ylim(0, 80)  # fixed height from ground to top

    particle_scatter = [ax.scatter([], [], s=5, color=tracker.colors[i], alpha=0.3) for i in range(num_balls)]
    true_lines = [ax.plot([], [], '-', color=tracker.colors[i])[0] for i in range(num_balls)]
    est_lines = [ax.plot([], [], 'o--', color=tracker.colors[i])[0] for i in range(num_balls)]

    def init():
        return particle_scatter + true_lines + est_lines

    def update(frame):
        if tracker.all_landed() or frame >= max_steps:
            anim.event_source.stop()

        results = tracker.step_all()
        for i, (particles, est) in enumerate(results):
            ball = tracker.balls[i]
            particle_scatter[i].set_offsets(particles)
            true_xy = np.array(ball.true_path)
            est_xy = np.array(ball.est_path)
            true_lines[i].set_data(true_xy[:, 0], true_xy[:, 1])
            est_lines[i].set_data(est_xy[:, 0], est_xy[:, 1])
        return particle_scatter + true_lines + est_lines

    anim = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=100)
    plt.show()


if __name__ == "__main__":
    animate_balls(num_balls=4, max_steps=300)
