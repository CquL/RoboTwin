"""
ACT-frozen + SAC-policy-head fine-tuning.

策略名称: ACT-frozen + SAC-head fine-tuning
    - 冻结 ACT trunk (backbone + Transformer decoder)
    - 替换原 action_head 为 SAC 随机策略头 (μ/logσ, linear_mode)
    - 新增双 Q critic、target network、feature replay buffer
    - BC regularization 使用预计算专家 (h, a_raw) 对
    - 推理: receding horizon, 每步重规划, 执行第一步动作

不是 Residual RL: SAC actor 输出完整动作, 不保留 ACT action_head 作为 base.

模块:
    forward_hidden:  暴露 ACT Transformer hidden states hs
    actor:           TanhGaussianActor (μ/logσ, linear_mode)
    critic:          TwinQCritic + EMA target networks
    replay_buffer:   FeatureReplayBuffer (存 h 向量)
    reward:          Progress-based dense reward
    env_wrapper:     SAPIEN 环境 RL wrapper
    expert_data:     专家数据加载与特征预计算
    sac_trainer:     SAC + BC 联合训练循环
    sac_config:      训练配置
"""

from .forward_hidden import add_forward_hidden_to_detrvae
from .actor import TanhGaussianActor
from .critic import QNet, TwinQCritic
from .replay_buffer import FeatureReplayBuffer, RawReplayBuffer
from .reward import BeatBlockHammerReward, BimanualReward
from .expert_data import ExpertFeatureDataset, setup_expert_data
from .sac_config import SACConfig

__all__ = [
    "add_forward_hidden_to_detrvae",
    "TanhGaussianActor",
    "QNet",
    "TwinQCritic",
    "FeatureReplayBuffer",
    "RawReplayBuffer",
    "BeatBlockHammerReward",
    "BimanualReward",
    "ExpertFeatureDataset",
    "setup_expert_data",
    "SACConfig",
]
