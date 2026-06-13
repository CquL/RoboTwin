"""
SAPIEN 环境的 RL wrapper。

将 RoboTwin SAPIEN 环境包装成标准 RL 接口:
    reset() → obs_dict
    step(action) → (obs_dict, reward, done, info)

关键适配:
    1. take_action() 内部包含 TOPP 轨迹优化 + 多步物理仿真
    2. check_success() 在 take_action 内部被调用
    3. 观测包含 qpos + 多相机图像
"""

import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Dict, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class SAPIENRLWrapper:
    """
    SAPIEN 环境的 RL wrapper。
    """

    def __init__(
        self,
        task_name: str = "beat_block_hammer",
        task_config: str = "demo_clean_regen_20260604_144403",
        seed: int = 0,
        max_episode_steps: int = 400,
        headless: bool = True,
        camera_names: Tuple[str, ...] = ("cam_high", "cam_right_wrist", "cam_left_wrist"),
        image_size: Tuple[int, int] = (480, 640),
        device: str = "cuda:0",
    ):
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.camera_names = list(camera_names)
        self.image_size = image_size
        self.device = device

        self._task_env = None
        self._args = None
        self._step_count = 0
        self._current_seed = seed
        self._prev_action = None
        self._prev_dist_xy = None  # 用于 progress reward

    def _build_env(self):
        from envs import CONFIGS_PATH
        import yaml
        import importlib

        with open(f"./task_config/{self.task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        embodiment_type = args.get("embodiment", ["aloha-agilex"])
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(emb_type):
            return _embodiment_types[emb_type]["file_path"]

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        else:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def get_config(robot_file):
            import yaml as _yaml
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return _yaml.load(f.read(), Loader=_yaml.FullLoader)

        args["left_embodiment_config"] = get_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_config(args["right_robot_file"])

        args["eval_mode"] = False
        args["task_name"] = self.task_name
        args["task_config"] = self.task_config
        args["save_data"] = False
        args["render_freq"] = 0
        args.pop("seed", None)

        envs_module = importlib.import_module(f"envs.{self.task_name}")
        task_env = getattr(envs_module, self.task_name)()
        self._args = args
        return task_env

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._current_seed = seed

        max_retries = 3
        for retry in range(max_retries):
            try:
                if self._task_env is not None:
                    try:
                        self._task_env.close_env()
                    except Exception:
                        pass
                self._task_env = self._build_env()
                self._task_env.setup_demo(
                    now_ep_num=0, seed=self._current_seed,
                    is_test=False, **self._args,
                )
                break
            except Exception as e:
                print(f"[Env] setup_demo failed (retry {retry+1}/{max_retries}, seed={self._current_seed}): {e}")
                self._current_seed += 1
                if retry == max_retries - 1:
                    raise RuntimeError(f"Failed to reset after {max_retries} attempts")

        self._step_count = 0
        self._task_env.take_action_cnt = 0
        self._task_env.eval_success = False
        self._prev_action = None
        self._prev_dist_xy = None

        if self._task_env.step_lim is None:
            self._task_env.step_lim = self.max_episode_steps

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._step_count += 1
        self._task_env.take_action(action, action_type="qpos")

        success = self._task_env.eval_success
        task_timeout = (self._task_env.step_lim is not None
                        and self._task_env.take_action_cnt >= self._task_env.step_lim)
        wrapper_timeout = self._step_count >= self.max_episode_steps
        done = success or task_timeout or wrapper_timeout

        reward, reward_info = self._compute_reward(action, success)
        obs = self._get_obs()
        self._prev_action = action.copy()

        info = {"success": success, "timeout": task_timeout or wrapper_timeout,
                "step": self._step_count, "take_action_cnt": self._task_env.take_action_cnt,
                **reward_info}
        return obs, reward, done, info

    def _get_obs(self) -> Dict[str, Any]:
        raw_obs = self._task_env.get_obs()

        qpos = raw_obs["joint_action"]["vector"].copy()

        images_rgb = {}
        for cam_name in ["head_camera", "right_wrist_camera", "left_wrist_camera"]:
            if cam_name in raw_obs["observation"]:
                images_rgb[cam_name] = raw_obs["observation"][cam_name]["rgb"].copy()

        cam_name_map = {
            "cam_high": "head_camera",
            "cam_right_wrist": "right_wrist_camera",
            "cam_left_wrist": "left_wrist_camera",
        }
        act_images = []
        first_shape = None
        for cam_name in self.camera_names:
            sapi_cam = cam_name_map.get(cam_name, cam_name)
            if sapi_cam in images_rgb:
                img = images_rgb[sapi_cam].copy()
                if first_shape is None:
                    first_shape = img.shape[:2]
                if img.shape[0] != first_shape[0] or img.shape[1] != first_shape[1]:
                    try:
                        from PIL import Image
                        img = np.array(Image.fromarray(img).resize(
                            (first_shape[1], first_shape[0]), Image.BILINEAR))
                    except Exception:
                        new_img = np.zeros((*first_shape, 3), dtype=np.uint8)
                        hh = min(img.shape[0], first_shape[0])
                        ww = min(img.shape[1], first_shape[1])
                        new_img[:hh, :ww] = img[:hh, :ww]
                        img = new_img
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            else:
                if first_shape is not None:
                    img = np.zeros((3, *first_shape), dtype=np.float32)
                else:
                    img = np.zeros((3, 240, 320), dtype=np.float32)
            act_images.append(img)

        act_images = np.stack(act_images, axis=0)
        return {"qpos": qpos, "images": act_images,
                "head_cam": images_rgb.get("head_camera"),
                "right_cam": images_rgb.get("right_wrist_camera"),
                "left_cam": images_rgb.get("left_wrist_camera")}

    def _compute_reward(self, action: np.ndarray, success: bool) -> Tuple[float, Dict]:
        """
        Progress-based dense reward for beat_block_hammer.

        r = 10.0 * success
          + 2.0 * (prev_dist - curr_dist)     ← progress toward block
          + 0.5 * hammer_lifted
          + 0.5 * hammer_near_block
          - 0.01 * ||(a - a_prev) / action_std||²
          - 0.001                              ← time penalty
        """
        info = {"success": success}

        try:
            hammer_pose = self._task_env.hammer.get_functional_point(0, "pose")
            hammer_pos = hammer_pose.p
            block_pose = self._task_env.block.get_functional_point(1, "pose")
            block_pos = block_pose.p

            curr_dist = np.linalg.norm(hammer_pos[:2] - block_pos[:2])
            hammer_lifted = hammer_pos[2] > 0.81
            hammer_near_block = curr_dist < 0.08

            info["hammer_pos"] = hammer_pos
            info["block_pos"] = block_pos
            info["dist_xy"] = curr_dist
            info["hammer_lifted"] = hammer_lifted
            info["hammer_near_block"] = hammer_near_block

            # Progress reward
            progress = 0.0
            if self._prev_dist_xy is not None:
                progress = self._prev_dist_xy - curr_dist
            self._prev_dist_xy = curr_dist
            info["progress"] = progress

            # 构建奖励
            reward = 0.0
            if success:
                reward += 10.0
                return reward, info

            reward += 2.0 * progress
            reward += 0.5 * float(hammer_lifted)
            reward += 0.5 * float(hammer_near_block)

            # 动作平滑惩罚（归一化空间）
            if self._prev_action is not None:
                from policy.ACT.sac.reward import _load_stats_once
                import pickle as _pickle
                stats_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "act_ckpt/act-beat_block_hammer/demo_clean_regen_20260604_144403-50/dataset_stats.pkl"
                )
                try:
                    with open(stats_path, "rb") as f:
                        _s = _pickle.load(f)
                    action_std = _s["action_std"]
                    delta_norm = (action - self._prev_action) / (action_std + 1e-6)
                    reward -= 0.01 * np.sum(delta_norm ** 2)
                except Exception:
                    reward -= 0.01 * np.sum((action - self._prev_action) ** 2)

            reward -= 0.001  # 时间惩罚

        except Exception as e:
            reward = 10.0 if success else -0.01
            info["error"] = str(e)

        return reward, info

    def close(self):
        if self._task_env is not None:
            try:
                self._task_env.close_env()
            except Exception:
                pass
            self._task_env = None


class ACTFeatureExtractor:
    """
    从 SAPIEN 观测中提取 ACT trunk 特征。
    用于 head-only 模式: frozen ACT trunk 提取特征 h。
    """

    def __init__(
        self,
        act_model: torch.nn.Module,
        stats: Dict,
        camera_names: Tuple[str, ...] = ("cam_high", "cam_right_wrist", "cam_left_wrist"),
        device: str = "cuda:0",
    ):
        self.act_model = act_model
        self.stats = stats
        self.camera_names = camera_names
        self.device = device
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.act_model.eval()
        for param in self.act_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, obs: Dict[str, Any], z_mode: str = "zero") -> np.ndarray:
        """
        从观测提取 ACT 特征。

        qpos RAW → z-score normalize → ACT forward_hidden → h0
        """
        qpos = obs["qpos"].copy()
        qpos_norm = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        qpos_tensor = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        images = obs["images"].copy()
        images_tensor = torch.from_numpy(images).float().to(self.device).unsqueeze(0)

        for cam_idx in range(images_tensor.shape[1]):
            images_tensor[0, cam_idx] = self.normalize(images_tensor[0, cam_idx])

        from .forward_hidden import extract_actor_feat
        hs = self.act_model.forward_hidden(qpos_tensor, images_tensor, z_mode=z_mode)
        h = extract_actor_feat(hs, mode="first")
        return h.squeeze(0).cpu().numpy()
