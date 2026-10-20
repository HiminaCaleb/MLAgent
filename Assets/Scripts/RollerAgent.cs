using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class RollerAgent : Agent
{
    [Header("Environment References")]
    [Tooltip("Reference to the target GameObject")]
    public Transform target;

    [Header("Movement Settings")]
    [Tooltip("Movement force multiplier")]
    public float forceMultiplier = 10f;

    [Tooltip("Platform Y threshold below which the agent is considered to have fallen off")]
    public float fallThreshold = -0.5f;

    [Tooltip("Distance threshold to consider the target reached")]
    public float reachDistance = 1.42f;

    private Rigidbody rBody;
    private float previousDistanceToTarget;

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
    }

    /// <summary>
    /// Called at the start of each episode (reset positions and velocities).
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // If agent fell off, reset momentum and position
        if (transform.localPosition.y < fallThreshold)
        {
            rBody.angularVelocity = Vector3.zero;
#if UNITY_6000_0_OR_NEWER
            rBody.linearVelocity = Vector3.zero;
#else
            rBody.velocity = Vector3.zero;
#endif
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
        }
        else
        {
            // Reset velocity and position on successful catch as well
            rBody.angularVelocity = Vector3.zero;
#if UNITY_6000_0_OR_NEWER
            rBody.linearVelocity = Vector3.zero;
#else
            rBody.velocity = Vector3.zero;
#endif
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
        }

        // Spawn target at a random position on the platform (platform size ~ 10x10)
        target.localPosition = new Vector3(
            Random.Range(-4f, 4f),
            0.5f,
            Random.Range(-4f, 4f)
        );

        previousDistanceToTarget = Vector3.Distance(transform.localPosition, target.localPosition);
    }

    /// <summary>
    /// Collect numerical vector observations (Total Vector Observation Space Size: 9).
    /// 1. Target Position (x, y, z) -> 3
    /// 2. Agent Position (x, y, z)  -> 3
    /// 3. Agent Velocity (vx, vy, vz) -> 3
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Target position (3 floats)
        sensor.AddObservation(target.localPosition);

        // Agent's own position (3 floats)
        sensor.AddObservation(transform.localPosition);

        // Agent's velocity (3 floats)
#if UNITY_6000_0_OR_NEWER
        sensor.AddObservation(rBody.linearVelocity);
#else
        sensor.AddObservation(rBody.velocity);
#endif
    }

    /// <summary>
    /// Receive actions from Python PPO policy or Heuristic and apply physics forces.
    /// Continuous Actions:
    /// actions[0] = force along X axis (Left / Right)
    /// actions[1] = force along Z axis (Forward / Backward)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 1. Actions -> Forces
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        // 2. Rewards: Distance progression shaping (small incentive to move towards target)
        float currentDistance = Vector3.Distance(transform.localPosition, target.localPosition);
        if (currentDistance < previousDistanceToTarget)
        {
            AddReward(0.005f); // Small reward for moving closer
        }
        previousDistanceToTarget = currentDistance;

        // Small step penalty to encourage speed and efficiency
        AddReward(-0.0005f);

        // 3. Rewards: Big Reward for reaching the target
        if (currentDistance < reachDistance)
        {
            SetReward(2.0f);
            EndEpisode();
        }

        // 4. Punishment: Falling off the platform edge
        if (transform.localPosition.y < fallThreshold)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    /// <summary>
    /// Manual control with keyboard for testing before training (Heuristic Mode).
    /// Supports both Unity's New Input System (Unity 6 default) and Legacy Input.
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        float h = 0f;
        float v = 0f;

#if ENABLE_INPUT_SYSTEM
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed) h = -1f;
            if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed) h = 1f;
            if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed) v = 1f;
            if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed) v = -1f;
        }
#else
        h = Input.GetAxis("Horizontal");
        v = Input.GetAxis("Vertical");
#endif

        continuousActionsOut[0] = h;
        continuousActionsOut[1] = v;
    }
}
