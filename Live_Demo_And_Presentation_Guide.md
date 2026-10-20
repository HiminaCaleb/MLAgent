# Friday Presentation & Live In-Person Demo Guide
**Course**: ARI216D - Artificial Intelligence 226 (TUT)  
**Student**: Mashamba MC (213682156)  
**Topic**: 3D RollerBall Reinforcement Learning Agent (Unity ML-Agents + PPO)  

---

# Part 1: How to Do the In-Person Live Demo

During your in-person evaluation, you will want to demonstrate two things:
1. **Show the trained agent running (Inference)** — shows immediate results.
2. **Train a new agent from scratch live (Training run)** — proves you know how the pipeline works.

---

### Demo Flow A: Show the Trained Ball (Inference Demo — 30 Seconds)
1. Open Unity.
2. Open `Assets/Scenes/SampleScene.unity`.
3. In the Hierarchy, click `RollerAgent`.
4. In the Inspector, ensure:
   - **Behavior Type**: `Inference Only`
   - **Model**: `RollerBall.onnx`
5. Press **Play (▶)** in Unity.
6. **What to say to the lecturer**:
   > *"Here you can see the trained agent running inside Unity with zero external connection. The ball takes in its position, velocity, and the target cube's position, running it through the 128-neuron ONNX neural network to calculate exact lateral and longitudinal forces."*

---

### Demo Flow B: Train a Brand-New Agent Live (Live Training Demo — 2 Minutes)
To prove you can train from scratch in front of the evaluator:

1. In Unity, set `RollerAgent`'s **Behavior Type** back to **`Default`** (and clear the Model slot, or leave it as Default).
2. Open File Explorer to:
   ```text
   C:\Users\musac\OneDrive\Desktop\AIR216D\ML Agent\MLAgent\MLAgent
   ```
3. Open PowerShell or double-click `run_training.bat`, but give it a fresh run name (e.g. `Demo_Run`):
   ```powershell
   & "C:\Users\musac\miniconda3\envs\mlagents\Scripts\mlagents-learn.exe" rollerball_config.yaml --run-id=Demo_Run --force
   ```
4. As soon as the terminal says:
   ```text
   Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
   ```
5. Click **Play (▶)** in Unity.
6. Watch the terminal:
   - At step 5000: you'll see the ball wobbling and falling off (Mean Reward ~0.4).
   - At step 10000: you'll see the ball start steering towards the cube (Mean Reward ~1.4).
   - At step 15000: you'll see the ball hitting the cube consistently (Mean Reward ~2.0+).
7. Press **Stop** in Unity.
8. Show the newly created model exported in `results\Demo_Run\My Behavior.onnx`.

---

# Part 2: How to Master and Explain the Code (`RollerAgent.cs`)

Here is the exact code breakdown explained simply so you can answer any question comfortably:

```
                  +----------------------------------------------+
                  |               RollerAgent.cs                 |
                  +----------------------------------------------+
                                         |
     +-------------------+---------------+-------------------+--------------------+
     |                   |                                   |                    |
Initialize()     OnEpisodeBegin()                  CollectObservations()   OnActionReceived()
(Setup Rigidbody) (Reset Ball & Target position)   (Feed 9 float inputs)   (Apply forces + Rewards)
```

### 1. `Initialize()`
```csharp
public override void Initialize()
{
    rBody = GetComponent<Rigidbody>();
}
```
* **Explanation**: Called once when the game starts. It grabs the `Rigidbody` physics component so we can apply physical forces and read velocity.

---

### 2. `OnEpisodeBegin()`
```csharp
public override void OnEpisodeBegin()
{
    // 1. Reset ball if it fell off
    if (transform.localPosition.y < fallThreshold)
    {
        rBody.angularVelocity = Vector3.zero;
        rBody.linearVelocity = Vector3.zero;
        transform.localPosition = new Vector3(0f, 0.5f, 0f);
    }
    else
    {
        // Reset momentum on success
        rBody.angularVelocity = Vector3.zero;
        rBody.linearVelocity = Vector3.zero;
        transform.localPosition = new Vector3(0f, 0.5f, 0f);
    }

    // 2. Randomize target position
    target.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    previousDistanceToTarget = Vector3.Distance(transform.localPosition, target.localPosition);
}
```
* **Key Concept**: Called every time an episode restarts (either after reaching the target or falling off).
* **Why reset velocity to zero?** If we don't zero `linearVelocity` and `angularVelocity`, momentum from the previous run carries over and ruins the new episode.
* **Why randomize the target?** To prevent the agent from memorizing a single path. Randomization forces the neural network to generalize.

---

### 3. `CollectObservations(VectorSensor sensor)`
```csharp
public override void CollectObservations(VectorSensor sensor)
{
    sensor.AddObservation(target.localPosition);     // 3 floats (x, y, z)
    sensor.AddObservation(transform.localPosition);  // 3 floats (x, y, z)
    sensor.AddObservation(rBody.linearVelocity);     // 3 floats (vx, vy, vz)
}
```
* **Observation Space Size**: **`9`** (3 + 3 + 3).
* **Why send velocity?** The ball has inertia. Knowing velocity tells the agent if it is moving too fast towards the edge, allowing it to brake or counter-steer.

---

### 4. `OnActionReceived(ActionBuffers actionBuffers)`
```csharp
public override void OnActionReceived(ActionBuffers actionBuffers)
{
    // 1. Actions -> Forces
    Vector3 controlSignal = Vector3.zero;
    controlSignal.x = actionBuffers.ContinuousActions[0];
    controlSignal.z = actionBuffers.ContinuousActions[1];
    rBody.AddForce(controlSignal * forceMultiplier);

    // 2. Distance Shaping Incentive
    float currentDistance = Vector3.Distance(transform.localPosition, target.localPosition);
    if (currentDistance < previousDistanceToTarget)
    {
        AddReward(0.005f);
    }
    previousDistanceToTarget = currentDistance;

    // 3. Step Penalty
    AddReward(-0.0005f);

    // 4. Goal Success
    if (currentDistance < reachDistance)
    {
        SetReward(2.0f);
        EndEpisode();
    }

    // 5. Fall Off Penalty
    if (transform.localPosition.y < fallThreshold)
    {
        SetReward(-1.0f);
        EndEpisode();
    }
}
```
* **Continuous Actions (2)**: Continuous values between -1 and +1 for X and Z axis.
* **`AddForce`**: Simulates realistic continuous physics rather than teleporting.
* **Reward Shaping**:
  * **+2.0**: Big positive reward for touching the target.
  * **-1.0**: Big penalty for falling off the edge.
  * **-0.0005**: Step penalty to encourage reaching the goal quickly.
  * **+0.005**: Dense reward for closing the distance (prevents the agent from wandering aimlessly).

---

### 5. `Heuristic(in ActionBuffers actionsOut)`
```csharp
public override void Heuristic(in ActionBuffers actionsOut)
{
    var continuousActionsOut = actionsOut.ContinuousActions;
    float h = 0f; float v = 0f;
    var keyboard = Keyboard.current;
    if (keyboard != null)
    {
        if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed) h = -1f;
        if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed) h = 1f;
        if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed) v = 1f;
        if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed) v = -1f;
    }
    continuousActionsOut[0] = h;
    continuousActionsOut[1] = v;
}
```
* **Purpose**: Used for manual player control (testing the physics with WASD or arrow keys before training AI).

---

# Part 3: Top Lecturer Questions & Perfect Answers

#### Q1: "What reinforcement learning algorithm did you use, and why?"
* **Answer**: *"We used **PPO (Proximal Policy Optimization)**. PPO is an actor-critic algorithm that clips the objective function to avoid destructive policy updates. It is ideal for continuous control tasks like our 3D physics-based RollerBall where actions are continuous forces rather than discrete button presses."*

#### Q2: "What is the observation space size and what does it contain?"
* **Answer**: *"The observation space has a size of **9**. It contains 3 floats for target position, 3 floats for the agent's own position, and 3 floats for the agent's linear velocity. Velocity is critical because it gives the neural network awareness of inertia and momentum."*

#### Q3: "What prevents the agent from just spinning or standing still?"
* **Answer**: *"We implemented two mechanisms: a **step penalty (-0.0005 per step)** which makes lingering costly, and **distance shaping (+0.005)** which provides a small reward gradient whenever the agent gets closer to the target."*

#### Q4: "Why do you have 4 training areas instead of just 1?"
* **Answer**: *"Parallel training environments allow Unity to collect 4 times as many experience tuples per second without extra overhead. It decorrelates training data and significantly speeds up convergence."*

#### Q5: "How did you know the model actually learned?"
* **Answer**: *"We analyzed our training metrics across 21,000 steps. The **Mean Reward** increased from **0.42** to **2.24**, and the **Standard Deviation** dropped from **1.52** down to **0.55**, proving that the ball transitioned from erratic exploration to consistent, high-reward behavior."*

---

# Part 4: 2-Minute Presentation Pitch Script

> *"Good day, everyone. Today I am presenting our Reinforcement Learning 3D RollerBall Agent built using Unity ML-Agents and Proximal Policy Optimization (PPO).*
>
> *The task is continuous control: a sphere on an elevated 10x10 platform must locate a randomly spawning cube, roll toward it, and avoid falling off the edge.*
>
> *Our agent observes a 9-dimensional state space: the target's position, its own position, and its 3D linear velocity. It outputs two continuous actions representing physics forces applied along the X and Z axes.*
>
> *We trained the agent over 21,000 steps across 4 parallel environments. As shown in our logs, the mean cumulative reward rose from 0.42 to 2.24, with variance dropping by over 60%.*
>
> *Finally, the trained policy was exported into an ONNX neural network, which now runs inside Unity with zero external dependencies in real time. Let me demonstrate the agent in action."*
