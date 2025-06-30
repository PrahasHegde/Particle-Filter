## 🧪 Particle Filter for Estimating Ball Trajectories

This project implements a Particle Filter-based simulation to estimate the positions and velocity vectors of multiple simultaneously flying balls using noisy, incomplete observations. The implementation follows the requirements set forth in the problem statement and includes two separate approaches.

* * *

## 📌 Problem Statement

Simulate the motion of n ≥ 1 balls thrown into the air from unknown launch positions and directions. The goal is to estimate their positions and velocities over time using only:
- Noisy 2D position observations (x, y)
- Incomplete observation sequences (i.e., sensor dropout)

## Core Requirements:
- Balls are *observationally indistinguishable*.
- No knowledge of initial position or velocity (only broad area assumptions).
- Simulated observations must include:
  - High Gaussian noise.
  - Variable time intervals.
  - Gaps due to sensor failure.
- Estimate ball positions *even during dropout*.
- Handle both:
  - Similar initial conditions.
  - Clearly different launch parameters.

* * *

##🧠 Approaches Implemented
1. Single Particle Filter for Multiple Balls
Each particle represents the joint state of all balls.
Captures interdependencies but is computationally heavier.
Suitable for tight coupling or indistinguishable observations.
2. Multiple Independent Particle Filters
One filter per ball.
Easier to implement and scale.
Works well when ball paths diverge.
Manual association of observations to filters is required.

* * *

## ⚙ Simulation Details

- *State Definition*: [x, y, vx, vy] per ball
- *Transition Model*: Projectile motion under gravity
- *Observation Model*: Noisy position measurements
- *Noise Model*: Additive Gaussian noise on observations
- *Dropout Handling*: Missing observations are simulated for several time steps

* * *

## Overview

- Approach 1: The code demonstrates a multi-target tracking system where a single particle filter collectively tracks multiple independent objects. It handles the challenge of data association by employing K-Means clustering to identify distinct groups of particles and the Hungarian algorithm to assign these clusters to specific targets. This approach is particularly useful in scenarios where individual object identities might be ambiguous from raw sensor data.
- Approach 2: The code simulates projectile motion for multiple balls, generating noisy observations for each. It then employs multiple Particle Filters, with one filter dedicated to tracking each ball, using a set of weighted particles to estimate its state. For each time step, each filter predicts particle movement, updates weights based on observations, and utilizes K-Means clustering to pinpoint the estimated ball position.

* * *

## 📈 Output

- Plots show true vs estimated positions for each ball.
- Error visualization over time.
- Supports both low-noise and high-noise scenarios.

* * *
