# Bimanual Reinforcement Learning for Catching Large-scale Object

## Test Policy
- Objects used in training phase
  
  <img src="https://github.com/user-attachments/assets/52196e08-3a38-41e2-83c1-28b39f16e5f6" width="400" height="211">
  <img src="https://github.com/user-attachments/assets/ec68f07a-f879-4974-9628-a923b28483ab" width="400" height="211">
  <img src="https://github.com/user-attachments/assets/4874cba9-1d11-4021-ad35-ca9964fb5b4a" width="400" height="211">

- Unseen Objects

  <img src="https://github.com/user-attachments/assets/610eea85-4cc2-4bc5-84f7-63d447a80628" width="400" height="211">
  <img src="https://github.com/user-attachments/assets/19494436-2a39-4269-8e6e-e929fe8cc4be" width="400" height="211">

## Requirements
- [IsaacGym](https://github.com/isaac-sim/IsaacGymEnvs)
- Python 3.8
- skrl Library

## Simulation Setup

`task/bimnual_grasp.py` is based on `VecTask` class of IsaacGym. Therefore, follow the instruction for utilizing custom VecTasks and register this environment.

## Train

`train/PPO_Bimanualgrasp.py` is the configuration file of PPO algorithm. Inferring this configuration, the policy can be trained by commanding this line:

```ruby
  python train.py task=$Task_name$
```

`$Task_name$` can be configured in `IsaacGymEnvs/isaacgymenvs/tasks/__init__.py` file.
