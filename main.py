import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import cv2

# Kalman Filter class
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.estimated_position = (0, 0)

    def update(self, measurement):
        if measurement is not None:
            self.kf.predict()
            estimated = self.kf.correct(np.array(measurement, dtype=np.float32))
        else:
            estimated = self.kf.predict()
        self.estimated_position = (estimated[0, 0], estimated[1, 0])
        return self.estimated_position

# Ball with movement and Kalman filter
class Ball:
    def __init__(self, id):
        self.id = id
        self.x = random.uniform(0, 20)
        self.y = random.uniform(100, 150)
        self.vx = random.uniform(1.0, 3.0)
        self.vy = random.uniform(5.0, 10.0)  # Initial upward velocity
        self.g = -0.4  # Gravity
        self.true_path = []
        self.est_path = []

    def move(self):
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Apply gravity
        self.vy += self.g

        # Optional bounce on ground (you can remove this if you want it to fall and stop)
        if self.y <= 0:
            self.y = 0
            self.vy *= -0.7  # Dampen bounce
            if abs(self.vy) < 0.5:
                self.vy = 0

    def get_position(self):
        return self.x, self.y


# Tracker for multiple balls
class BallTracker:
    def __init__(self, num_balls):
        self.balls = [Ball(i) for i in range(num_balls)]
        self.kalman_filters = [KalmanFilter2D() for _ in range(num_balls)]

    def generate_particles(self, ball_pos, num_particles=25, spread=4):
        particles = []
        for _ in range(num_particles):
            x = random.gauss(ball_pos[0], spread)
            y = random.gauss(ball_pos[1], spread)
            particles.append((x, y))
        return particles

    def update_all(self):
        results = []
        for i, ball in enumerate(self.balls):
            ball.move()
            ball_pos = ball.get_position()
            ball.true_path.append(ball_pos)

            # Simulate missing measurements
            measurement = ball_pos if random.random() < 0.9 else None
            estimated = self.kalman_filters[i].update(measurement)
            ball.est_path.append(estimated)

            particles = self.generate_particles(ball_pos)
            results.append((particles, ball.true_path, ball.est_path))
        return results

# Main animation function
def run_animation(num_balls=2):
    tracker = BallTracker(num_balls)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title('True vs Estimated Ball Tracking with Particles')

    sc_particles = [ax.plot([], [], 'ro', markersize=3, alpha=0.4)[0] for _ in range(num_balls)]
    ln_true = [ax.plot([], [], 'b-', linewidth=2, label="True Path")[0] for _ in range(num_balls)]
    ln_est = [ax.plot([], [], 'g--', linewidth=2, label="Estimated Path")[0] for _ in range(num_balls)]

    def update(frame):
        results = tracker.update_all()

        all_x = []
        all_y = []

        for i, (particles, true_path, est_path) in enumerate(results):
            px, py = zip(*particles)
            sc_particles[i].set_data(px, py)

            if true_path:
                x_true, y_true = zip(*true_path)
                ln_true[i].set_data(x_true, y_true)
                all_x.extend(x_true)
                all_y.extend(y_true)

            if est_path:
                x_est, y_est = zip(*est_path)
                ln_est[i].set_data(x_est, y_est)
                all_x.extend(x_est)
                all_y.extend(y_est)

        # Auto-adjust axes based on all points
        margin = 20
        if all_x and all_y:
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        return sc_particles + ln_true + ln_est

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Run the animation
run_animation(num_balls=4)
