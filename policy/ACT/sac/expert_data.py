"""
专家数据加载器 — 用于 BC Regularization。

从 ACT 训练数据集中加载专家 demonstrations，
预计算 ACT trunk 特征（frozen trunk 模式下），
提供训练时的 BC batch 采样。

关键: HDF5 中存储的是 RAW qpos 和 RAW action，
     需要在过 ACT trunk 前做 z-score 归一化，
     BC target 使用 RAW action（与 SAC actor 输出空间一致）。

两种模式:
    precomputed: 一次性预计算所有专家特征，存 GPU/CPU (适合 frozen trunk, 快)
    online:      训练时实时过 trunk (适合 trainable trunk, 慢)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class ExpertFeatureDataset(Dataset):
    """预计算的专家特征数据集。每条: (h, a_raw)"""

    def __init__(self, features: np.ndarray, actions: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.actions[idx]


def load_expert_episodes(
    dataset_dir: str,
    num_episodes: int,
    camera_names: List[str],
) -> List[Dict]:
    """
    加载专家 episodes 的原始数据。

    返回:
        List[Dict]: 每个 episode:
            "qpos":   (T, 14)   RAW 关节位置
            "images": (T, N_cam, H, W, 3)  图像 uint8
            "action": (T, 14)  RAW 动作
    """
    episodes = []

    for ep_idx in range(num_episodes):
        episode_path = os.path.join(dataset_dir, f"episode_{ep_idx}.hdf5")
        if not os.path.exists(episode_path):
            print(f"[ExpertData] Episode {ep_idx} not found, skipping")
            continue

        try:
            with h5py.File(episode_path, "r") as f:
                qpos = f["observations/qpos"][:].astype(np.float32)   # (T, 14) RAW
                action = f["action"][:].astype(np.float32)            # (T, 14) RAW

                images = []
                for cam_name in camera_names:
                    cam_images = f[f"observations/images/{cam_name}"][:].astype(np.uint8)
                    images.append(cam_images)

                # (T, N_cam, H, W, 3)
                images = np.stack(images, axis=1)

                episodes.append({
                    "qpos": qpos,
                    "images": images,
                    "action": action,
                })
        except Exception as e:
            print(f"[ExpertData] Failed to load episode {ep_idx}: {e}")
            continue

    print(f"[ExpertData] Loaded {len(episodes)} expert episodes (RAW qpos/action)")
    return episodes


def precompute_expert_features(
    episodes: List[Dict],
    act_model: nn.Module,
    stats: Dict,
    camera_names: List[str],
    feat_dim: int,
    device: str = "cuda:0",
    max_frames: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 frozen ACT trunk 预计算所有专家帧的特征。

    关键流程:
        1. qpos_raw → z-score normalize → ACT forward_hidden → h
        2. action_raw 直接作为 BC target（与 SAC actor 输出空间一致）

    参数:
        episodes:      专家 episode 列表 (RAW 数据)
        act_model:     DETRVAE 实例 (frozen, 已添加 forward_hidden)
        stats:         dataset_stats (包含 qpos_mean/std, action_mean/std)
        camera_names:  相机名称列表
        feat_dim:      特征维度
        device:        计算设备
        max_frames:    最大帧数

    返回:
        features: (N, feat_dim)  ACT 特征
        actions:  (N, act_dim)   RAW 专家动作 (BC target)
    """
    from .forward_hidden import extract_actor_feat

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    qpos_mean = stats["qpos_mean"]
    qpos_std = stats["qpos_std"]

    all_features = []
    all_actions = []

    act_model.eval()
    total_frames = 0

    print(f"[ExpertData] Precomputing features (max {max_frames} frames)...")

    for ep_idx, ep in enumerate(episodes):
        if total_frames >= max_frames:
            break

        qpos_raw = ep["qpos"]        # (T, 14) RAW
        images_raw = ep["images"]     # (T, N_cam, H, W, 3) uint8
        action_raw = ep["action"]     # (T, 14) RAW

        T = len(qpos_raw)
        batch_size = 32

        for start in range(0, T, batch_size):
            if total_frames >= max_frames:
                break

            end = min(start + batch_size, T)
            bs = end - start

            # ---- qpos: RAW → z-score normalize ----
            qpos_norm = (qpos_raw[start:end] - qpos_mean) / qpos_std
            qpos_batch = torch.from_numpy(qpos_norm).float().to(device)

            # ---- images: uint8 → [0,1] float → ImageNet normalize ----
            imgs = images_raw[start:end]  # (bs, N_cam, H, W, 3)
            imgs = imgs.transpose(0, 1, 4, 2, 3)  # (bs, N_cam, 3, H, W)
            imgs_batch = torch.from_numpy(imgs.copy()).float().to(device) / 255.0

            for cam_idx in range(imgs_batch.shape[1]):
                imgs_batch[:, cam_idx] = normalize(imgs_batch[:, cam_idx])

            # ---- ACT forward ----
            with torch.no_grad():
                hs = act_model.forward_hidden(qpos_batch, imgs_batch, z_mode="zero")
                h = extract_actor_feat(hs, mode="first")  # (bs, feat_dim)

            all_features.append(h.cpu().numpy())
            # BC target: RAW action (与 SAC actor 输出空间一致)
            all_actions.append(action_raw[start:end])

            total_frames += bs

            if ep_idx % 5 == 0 and start == 0:
                print(f"  Episode {ep_idx}, frame {total_frames}/{min(sum(len(ep['qpos']) for ep in episodes), max_frames)}")

    features = np.concatenate(all_features, axis=0)[:max_frames]
    actions = np.concatenate(all_actions, axis=0)[:max_frames]

    print(f"[ExpertData] Precomputed {len(features)} expert (h, a_raw) pairs")
    print(f"[ExpertData] Feature shape: {features.shape}, Action shape: {actions.shape}")

    return features, actions


def create_expert_loader(
    features: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """从预计算的特征创建 DataLoader。"""
    dataset = ExpertFeatureDataset(features, actions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)


def setup_expert_data(
    act_ckpt_dir: str,
    act_model: nn.Module,
    stats: Dict,
    camera_names: List[str],
    feat_dim: int,
    device: str = "cuda:0",
    max_frames: int = 50000,
    expert_batch_size: int = 64,
) -> Tuple[Optional[DataLoader], int]:
    """
    一站式设置专家数据。

    1. 从 ACT checkpoint 目录匹配 processed_data 路径
    2. 加载 RAW 专家 episodes
    3. 预计算 ACT 特征（qpos 过 z-score 归一化，action 保持 raw）
    4. 创建 DataLoader

    返回:
        expert_loader: DataLoader → (h, a_raw) batch
        num_experts:   int
    """
    from policy.ACT.constants import SIM_TASK_CONFIGS

    dataset_dir = None
    num_episodes = 0

    ckpt_basename = os.path.basename(act_ckpt_dir.rstrip("/"))
    for suffix in ["-ft_from_", "-freeze_backbone_", "-action_head_", "-lora_"]:
        if suffix in ckpt_basename:
            ckpt_basename = ckpt_basename.split(suffix)[0]

    best_match = None
    best_match_len = 0
    for task_name, task_config in SIM_TASK_CONFIGS.items():
        if "beat_block_hammer" not in task_name:
            continue
        task_suffix = task_name.split("sim-beat_block_hammer-")[-1] if "sim-beat_block_hammer-" in task_name else ""
        if task_suffix and task_suffix in ckpt_basename:
            if len(task_suffix) > best_match_len:
                best_match = (task_name, task_config)
                best_match_len = len(task_suffix)

    if best_match is not None:
        task_name, task_config = best_match
        dataset_dir = task_config["dataset_dir"]
        num_episodes = task_config["num_episodes"]
        print(f"[ExpertData] Matched: {task_name} → {dataset_dir}")
    else:
        for task_name, task_config in SIM_TASK_CONFIGS.items():
            if "beat_block_hammer" in task_name and "demo_clean" in ckpt_basename and "demo_clean" in task_name:
                dataset_dir = task_config["dataset_dir"]
                num_episodes = task_config["num_episodes"]
                break

    if dataset_dir is None:
        print("[ExpertData] WARNING: Could not find expert dataset. BC regularization disabled.")
        return None, 0

    # dataset_dir 是相对路径 (相对于 policy/ACT/)
    if dataset_dir.startswith("./"):
        act_policy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.normpath(os.path.join(act_policy_dir, dataset_dir[2:]))

    print(f"[ExpertData] Loading expert data from {dataset_dir} ({num_episodes} episodes)...")

    episodes = load_expert_episodes(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
    )

    if len(episodes) == 0:
        print("[ExpertData] WARNING: No episodes loaded. BC regularization disabled.")
        return None, 0

    features, actions = precompute_expert_features(
        episodes=episodes,
        act_model=act_model,
        stats=stats,
        camera_names=camera_names,
        feat_dim=feat_dim,
        device=device,
        max_frames=max_frames,
    )

    loader = create_expert_loader(features, actions, batch_size=expert_batch_size)
    print(f"[Setup] Expert data ready: {len(features)} frames")
    return loader, len(features)
