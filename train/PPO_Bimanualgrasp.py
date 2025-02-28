import isaacgym
import isaacgymenvs

import matplotlib.pyplot as plt

import argparse

from datetime import datetime
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
# from skrl.agents.torch.ppo import PPO
# from skrl.envs.torch import wrap_env
from skrl.utils import set_seed
from isaacgymenvs.Trainer_fullstate import TrajectoryConditionedDynamicGraspTrainer
from isaacgymenvs.Tester_fullstate import TrajectoryConditionedDynamicGraspTester
# from isaacgymenvs.imitation_eval import TrajectoryConditionedDynamicGraspTester
from isaacgymenvs.wrapper_for_fullstate import wrap_env
from isaacgymenvs.PPO_for_fullstate import PPO, PPO_DEFAULT_CONFIG

seed = set_seed(826)
# seed = set_seed()

class DynamicGraspNet(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self)

        self.policy_net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                #  nn.LayerNorm(256),
                                 nn.ELU(),
                                 nn.Linear(512,1024),
                                #  nn.LayerNorm(512),
                                 nn.ELU(),
                                 nn.Linear(1024,1024),
                                #  nn.LayerNorm(256),
                                 nn.ELU(),
                                 nn.Linear(1024,512),
                                #  nn.LayerNorm(128),
                                 nn.ELU(),
                                )

        self.value_net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                #  nn.LayerNorm(256),
                                 nn.ELU(),
                                 nn.Linear(512,1024),
                                #  nn.LayerNorm(512),
                                 nn.ELU(),
                                 nn.Linear(1024,1024),
                                #  nn.LayerNorm(256),
                                 nn.ELU(),
                                 nn.Linear(1024,512),
                                #  nn.LayerNorm(128),
                                 nn.ELU(),
                                )

        # self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
        #                         #  nn.LayerNorm(256),
        #                          nn.ELU(),
        #                          nn.Linear(256,512),
        #                         #  nn.LayerNorm(512),
        #                          nn.ELU(),
        #                          nn.Linear(512,256),
        #                         #  nn.LayerNorm(256),
        #                          nn.ELU(),
        #                          nn.Linear(256,128),
        #                         #  nn.LayerNorm(128),
        #                          nn.ELU(),
        #                         )

        # For Policy
        self.mean_action_layer = nn.Linear(512,self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # For Value
        self.value_layer = nn.Linear(512,1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_action_layer(self.policy_net(inputs["states"])), self.log_std_parameter, {}
            # return self.mean_action_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.value_net(inputs["states"])), {}
            # return self.value_layer(self.net(inputs["states"])), {}

parser = argparse.ArgumentParser(description="headless & wandb & test")
parser.add_argument("--headless", default=False, required=False, action="store_true")
parser.add_argument("--wandb", default=False, required=False, action="store_true")
parser.add_argument("--test", default=False, required=False, action="store_true")
args = parser.parse_args()

unwrapped_env = isaacgymenvs.make(seed=seed,
                        task="BimanualGrasp",
                        num_envs=1,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=args.headless)
# unwrapped_env = load_isaacgym_env_preview4(task_name="FullState",
#                                            num_envs=1,)
env = wrap_env(unwrapped_env,"isaacgym-sykim")

device = env.device

memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device,
                      export=False,
                      export_format="csv",
                      export_directory="/home/lr-drgn/rl_env/IsaacGymEnvs/isaacgymenvs/runs/BimanualGrasp/traj")

models_ppo = {}
models_ppo["policy"] = DynamicGraspNet(observation_space=env.observation_space, 
                                action_space=env.action_space,
                                device=device)

models_ppo["value"] = models_ppo["policy"] # same instance: shared model

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 32  # memory_size
cfg_ppo["learning_epochs"] = 4
cfg_ppo["mini_batches"] = 4  # batch_size = memory_size * num_envs / mini_batches
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-4
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016, "kl_factor": 2.0, "min_lr": 1e-6, "lr_factor": 1.5}
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = None
# cfg_ppo["state_preprocessor"] = None
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg_ppo["value_preprocessor"] = None
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints each 200 and 2000 timesteps respectively
cfg_ppo["experiment"]["directory"] = "/home/lr-drgn/rl_env/IsaacGymEnvs/isaacgymenvs/runs/"
cfg_ppo["experiment"]["experiment_name"] = "BimanualGrasp" + "_" + datetime.now().strftime("%m%d_%H%M")
cfg_ppo["experiment"]["write_interval"] = 10
cfg_ppo["experiment"]["checkpoint_interval"] = 1000
cfg_ppo["experiment"]["wandb"] = args.wandb

cfg_ppo["timesteps"] = 50000

agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": cfg_ppo["timesteps"]}
cfg_tester = {"timesteps": 50000}

# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.test == False :
    # Training Code
    cfg_ppo["experiment"]["wandb_kwargs"] = {"project":"Bimanual_Grasp"}
    trainer = TrajectoryConditionedDynamicGraspTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()
else :
    cfg_ppo["experiment"]["write_interval"] = 0
    cfg_ppo["experiment"]["checkpoint_interval"] = 0
    agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    # Testing Code
    trainer = TrajectoryConditionedDynamicGraspTester(cfg=cfg_tester, env=env, agents=agent)
    # trainer = SequentialTrainer(cfg=cfg_tester, env=env, agents=agent)
    # agent.load("/home/lr-drgn/rl_env/IsaacGymEnvs/isaacgymenvs/runs/BimanualGrasp_0612_1403/checkpoints/best_agent.pt") # w/ penalty
    agent.load("/home/lr-drgn/rl_env/IsaacGymEnvs/isaacgymenvs/runs/BimanualGrasp_0614_1424/checkpoints/best_agent.pt") # w/o penalty
    # trainer = TrajectoryConditionedDynamicGraspTester(cfg=cfg_tester, env=env, agents=agent)

    trainer.eval()