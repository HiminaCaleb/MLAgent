Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

==========================================================
   AIR216D: Unity ML-Agents Training Runner (PPO)
==========================================================

[+] Starting ML-Agents training session: RollerBall_Run1
[+] Config: C:\Users\musac\OneDrive\Desktop\AIR216D\ML Agent\MLAgent\MLAgent\rollerball_config.yaml
[+] When prompted, press Play in the Unity Editor to begin training!


            ┐  ╖
        ╓╖╬│╡  ││╬╖╖
    ╓╖╬│││││┘  ╬│││││╬╖
 ╖╬│││││╬╜        ╙╬│││││╖╖                               ╗╗╗
 ╬╬╬╬╖││╦╖        ╖╬││╗╣╣╣╬      ╟╣╣╬    ╟╣╣╣             ╜╜╜  ╟╣╣
 ╬╬╬╬╬╬╬╬╖│╬╖╖╓╬╪│╓╣╣╣╣╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╒╣╣╖╗╣╣╣╗   ╣╣╣ ╣╣╣╣╣╣ ╟╣╣╖   ╣╣╣
 ╬╬╬╬┐  ╙╬╬╬╬│╓╣╣╣╝╜  ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╣╙ ╙╣╣╣  ╣╣╣ ╙╟╣╣╜╙  ╫╣╣  ╟╣╣
 ╬╬╬╬┐     ╙╬╬╣╣      ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣     ╣╣╣┌╣╣╜
 ╬╬╬╜       ╬╬╣╣      ╙╝╣╣╬      ╙╣╣╣╗╖╓╗╣╣╣╜ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣╦╓    ╣╣╣╣╣
 ╙   ╓╦╖    ╬╬╣╣   ╓╗╗╖            ╙╝╣╣╣╣╝╜   ╘╝╝╜   ╝╝╝  ╝╝╝   ╙╣╣╣    ╟╣╣╣
   ╩╬╬╬╬╬╬╦╦╬╬╣╣╗╣╣╣╣╣╣╣╝                                             ╫╣╣╣╣
      ╙╬╬╬╬╬╬╬╣╣╣╣╣╣╝╜
          ╙╬╬╬╣╣╣╜
             ╙

 Version information:
  ml-agents: 1.1.0,
  ml-agents-envs: 1.1.0,
  Communicator API: 1.5.0,
  PyTorch: 2.2.2+cpu
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
[INFO] Connected to Unity environment with package version 4.1.0 and communication version 1.5.0
[INFO] Connected new brain: My Behavior?team=0
[INFO] Hyperparameters for behavior name My Behavior:
        trainer_type:   ppo
        hyperparameters:
          batch_size:   64
          buffer_size:  1024
          learning_rate:        0.0003
          beta: 0.005
          epsilon:      0.2
          lambd:        0.95
          num_epoch:    3
          shared_critic:        False
          learning_rate_schedule:       linear
          beta_schedule:        linear
          epsilon_schedule:     linear
        checkpoint_interval:    500000
        network_settings:
          normalize:    True
          hidden_units: 128
          num_layers:   2
          vis_encode_type:      simple
          memory:       None
          goal_conditioning_type:       hyper
          deterministic:        False
        reward_signals:
          extrinsic:
            gamma:      0.99
            strength:   1.0
            network_settings:
              normalize:        False
              hidden_units:     128
              num_layers:       2
              vis_encode_type:  simple
              memory:   None
              goal_conditioning_type:   hyper
              deterministic:    False
        init_path:      None
        keep_checkpoints:       5
        even_checkpoints:       False
        max_steps:      500000
        time_horizon:   64
        summary_freq:   5000
        threaded:       False
        self_play:      None
        behavioral_cloning:     None
[INFO] My Behavior. Step: 5000. Time Elapsed: 90.140 s. Mean Reward: 0.423. Std of Reward: 1.524. Training.
[INFO] My Behavior. Step: 10000. Time Elapsed: 117.027 s. Mean Reward: 1.423. Std of Reward: 1.422. Training.
[INFO] My Behavior. Step: 15000. Time Elapsed: 146.098 s. Mean Reward: 2.041. Std of Reward: 0.961. Training.
[INFO] My Behavior. Step: 20000. Time Elapsed: 229.491 s. Mean Reward: 2.242. Std of Reward: 0.550. Training.
[WARNING] Restarting worker[0] after 'The Unity environment took too long to respond. Make sure that :
         The environment does not need user interaction to launch
         The Agents' Behavior Parameters > Behavior Type is set to "Default"
         The environment and the Python interface have compatible versions.
         If you're running on a headless server without graphics support, turn off display by either passing --no-graphics option or build your Unity executable as server build.'
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
[INFO] Exported results\RollerBall_Run1\My Behavior\My Behavior-21491.onnx
[INFO] Copied results\RollerBall_Run1\My Behavior\My Behavior-21491.onnx to results\RollerBall_Run1\My Behavior.onnx.
Traceback (most recent call last):
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Users\musac\miniconda3\envs\mlagents\Scripts\mlagents-learn.exe\__main__.py", line 5, in <module>
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\learn.py", line 270, in main
    run_cli(parse_command_line())
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\learn.py", line 266, in run_cli
    run_training(run_seed, options, num_areas)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\learn.py", line 138, in run_training
    tc.start_learning(env_manager)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents_envs\timers.py", line 305, in wrapped
    return func(*args, **kwargs)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\trainer_controller.py", line 175, in start_learning
    n_steps = self.advance(env_manager)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents_envs\timers.py", line 305, in wrapped
    return func(*args, **kwargs)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\trainer_controller.py", line 233, in advance
    new_step_infos = env_manager.get_steps()
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\env_manager.py", line 124, in get_steps
    new_step_infos = self._step()
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\subprocess_env_manager.py", line 420, in _step
    self._restart_failed_workers(step)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\subprocess_env_manager.py", line 328, in _restart_failed_workers
    self.reset(self.env_parameters)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\env_manager.py", line 68, in reset
    self.first_step_infos = self._reset_env(config)
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\subprocess_env_manager.py", line 446, in _reset_env
    ew.previous_step = EnvironmentStep(ew.recv().payload, ew.worker_id, {}, {})
  File "C:\Users\musac\miniconda3\envs\mlagents\lib\site-packages\mlagents\trainers\subprocess_env_manager.py", line 101, in recv
    raise env_exception
mlagents_envs.exception.UnityTimeOutException: The Unity environment took too long to respond. Make sure that :
         The environment does not need user interaction to launch
         The Agents' Behavior Parameters > Behavior Type is set to "Default"
         The environment and the Python interface have compatible versions.
         If you're running on a headless server without graphics support, turn off display by either passing --no-graphics option or build your Unity executable as server build.

Training session stopped or finished.
Press Enter to exit...