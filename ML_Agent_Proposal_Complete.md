ML Agent Proposal: 3D RollerBall Target Navigation & Obstacle Avoidance

Institution: Tshwane University of Technology (TUT)
Department: Department of Computer Systems Engineering
Programme / Course: Computer Systems
Module: ARI216D - Artificial Intelligence 226 (2026/2)
Assignment: ML Agent Proposal (Self-Made Task)
Due Date: 14 September 2026, 11:59 PM
Submission Format: PDF

Student Information:
Student Name: Mashamba MC
Student Number: 213682156
Partner Details: [Partner Name & Student Number, or Solo Submission]


1. Task Overview
Our machine learning agent will learn to roll a ball across a 3D platform, avoid falling off edges, navigate around obstacles, and intercept target objects. The agent will be built and trained within the Unity game engine using the Unity ML-Agents Toolkit, applying Reinforcement Learning (RL) so that the agent improves its behavior through trial and error rather than being explicitly programmed with fixed procedural rules.


2. Environment Specification
The simulation environment is an elevated 3D stage containing:
1. Platform (Floor): A flat, bounded horizontal surface (10 x 10 units) elevated above an open void with no perimeter walls.
2. The Agent (Sphere): A 3D sphere equipped with a Rigidbody and continuous physics forces to simulate realistic rolling, momentum, and friction.
3. The Target (Cube): A geometric object spawned at randomized (x, z) coordinates on the platform for each episode.
4. Obstacles (Barriers): Solid static and dynamic geometric blocks positioned between the agent and target to prevent straight-line navigation.
5. Boundaries / Fall Zone: If the agent rolls beyond the platform perimeter (y < 0), it enters the failure state.


3. Agent Observations / Sensors
At every decision step, the environment supplies the agent with a continuous numerical observation vector:
- Agent's Own Position (Vector3): Current (x, y, z) coordinates of the sphere on the platform.
- Agent's Velocity (Vector3): Current linear velocity (vx, vy, vz) so the agent perceives its momentum and inertia.
- Target Position (Vector3): The current (x, y, z) coordinates of the target cube.
- Relative Direction & Distance (Vector3): Computed as (target.position - agent.position) to provide immediate heading information.
- Obstacle Proximity: Relative vector distances or Raycast Perception inputs detecting nearby obstacle barriers and platform edges.


4. Agent Actions / Actuators
The agent interacts with the environment using continuous control outputs:
- Action 0 (X-axis Force): Continuous torque/force to roll the sphere Left / Right.
- Action 1 (Z-axis Force): Continuous torque/force to roll the sphere Forward / Backward.

Note: Movement is driven by Unity physics forces (Rigidbody.AddForce), requiring the agent to learn acceleration, deceleration, and steering dynamics.


5. Reward and Punishment Structure
Following the lecturer's specific guidance on reward shaping:

- Big Reward (+1.0 to +2.0): Intercepting / touching the target cube. Primary task success; completes episode and triggers reset.
- Small Reward (Progress) (+0.01 per step): Decreasing distance towards the target. Distance shaping to guide the agent toward the target.
- Level / Speed Bonus (+0.5 bonus): Intercepting a moving target (Levels 2 and 3). Rewards higher-difficulty dynamic tracking.
- Minor Penalty (-0.1): Colliding with an obstacle. Teaches obstacle avoidance without terminating the run.
- Step Penalty (-0.0005 per step): Each elapsed step without reaching the goal. Discourages idling; incentivizes the fastest, most efficient path.
- Big Punishment (Terminal) (-1.0): Falling off the platform edge (y < 0). Catastrophic failure penalty; immediately ends episode (EndEpisode).


6. Progressive Difficulty & Variable Targets
To ensure progressive curriculum learning and demonstrate higher agent capability:
- Level 1 - Static Target: Target is stationary at random coordinates; agent learns basic rolling mechanics and edge avoidance.
- Level 2 - Slow-Moving Target + Static Obstacles: Target moves across the platform at slow/moderate speed; obstacles block straight-line paths.
- Level 3 - Dynamic Target & Shifting Obstacles (Bonus): Target navigates non-linearly; obstacles shift, requiring real-time predictive navigation.


7. Tools and Technology
- Game Engine: Unity (3D project)
- ML Framework: Unity ML-Agents Toolkit
- Scripting Language: C# (implementing Agent, CollectObservations, OnActionReceived, and Heuristic)
- Training Backend: Python (via mlagents-learn using Proximal Policy Optimization - PPO)


8. Expected Outcome
By the end of training, the agent will:
- Reliably balance and navigate the platform with an edge-avoidance survival rate exceeding 95%.
- Steer around obstacles rather than colliding with them.
- Reach both static and moving targets from randomized starting positions, demonstrating a complete and robust application of reinforcement learning within Unity.
