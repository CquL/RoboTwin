"""
ACT + SAC 策略评估脚本。

用法:
    python -m policy.ACT.sac.eval_sac \
        --sac_ckpt_path policy/ACT/sac/sac_ckpt/beat_block_hammer-mvp/sac_best.ckpt \
        --task_name beat_block_hammer \
        --task_config demo_randomized \
        --num_episodes 50 \
        --seed 0

注意:
    - 当前评测用 is_test=False，与官方 eval_policy.py (is_test=True) 不完全一致
    - 最终论文比较应与官方 eval 一致，需写 official-compatible eval
    - 训练期 temporal_agg 已关闭
    - eval 使用确定性 mean action
"""

import os
import sys
import argparse
import json
import time

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
import pickle

from policy.ACT.sac.sac_config import SACConfig
from policy.ACT.sac.actor import TanhGaussianActor
from policy.ACT.sac.forward_hidden import add_forward_hidden_to_detrvae
from policy.ACT.sac.sac_env import SAPIENRLWrapper, ACTFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="ACT + SAC Policy Evaluation")
    parser.add_argument("--sac_ckpt_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="beat_block_hammer")
    parser.add_argument("--task_config", type=str, default="demo_randomized")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def load_model(ckpt_path: str, device: str):
    """加载 SAC checkpoint 并重建模型。"""
    print(f"[Eval] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    config_dict = ckpt.get("config", {})
    config = SACConfig.from_dict(config_dict)

    # 加载 ACT 模型
    from policy.ACT.detr.models import build_ACT_model

    class Args:
        pass
    args = Args()
    args.hidden_dim = config.act_hidden_dim
    args.dim_feedforward = 3200
    args.chunk_size = config.act_chunk_size
    args.camera_names = list(config.camera_names)
    args.backbone = "resnet18"
    args.enc_layers = 4
    args.dec_layers = 7
    args.nheads = 8
    args.dropout = 0.1
    args.pre_norm = False
    args.position_embedding = "sine"
    args.dilation = False
    args.masks = False
    args.peft_mode = "none"
    args.lora_r = 8
    args.lora_alpha = 16.0
    args.lora_dropout = 0.0
    args.lr_backbone = 1e-5
    args.lr = 1e-4
    args.state_dim = 14

    act_model = build_ACT_model(args)
    act_model.to(device)

    act_ckpt_dir = config.act_ckpt_dir
    act_ckpt_path = os.path.join(act_ckpt_dir, "policy_best.ckpt")
    if not os.path.exists(act_ckpt_path):
        act_ckpt_path = os.path.join(act_ckpt_dir, "policy_last.ckpt")

    state_dict = torch.load(act_ckpt_path, map_location=device)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    act_model.load_state_dict(state_dict, strict=True)
    act_model.eval()
    add_forward_hidden_to_detrvae(act_model)

    # 加载归一化统计量
    stats_path = os.path.join(act_ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # 构建 actor
    action_mean = torch.from_numpy(stats["action_mean"]).float()
    action_std = torch.from_numpy(stats["action_std"]).float()
    action_low = action_mean - 3 * action_std
    action_high = action_mean + 3 * action_std

    actor = TanhGaussianActor(
        feat_dim=config.feat_dim,
        act_dim=config.state_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=config.actor_hidden_dim,
        init_log_std=config.init_log_std,
        simple_mode=config.actor_simple_mode,
        linear_mode=config.actor_linear_mode,
        action_mean=action_mean if config.actor_linear_mode else None,
        action_std=action_std if config.actor_linear_mode else None,
    ).to(device)

    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    print(f"[Eval] Model loaded. SAC env_step={ckpt.get('env_step','?')}, best_eval={ckpt.get('best_eval_success','?')}")
    return act_model, actor, stats, config


def evaluate(act_model, actor, stats, config: SACConfig, args):
    """运行评估 — receding horizon, 每步重规划, deterministic mean action."""
    print(f"\n[Eval] {args.num_episodes} episodes...")

    feature_extractor = ACTFeatureExtractor(
        act_model=act_model, stats=stats,
        camera_names=config.camera_names, device=args.device,
    )

    env = SAPIENRLWrapper(
        task_name=args.task_name, task_config=args.task_config,
        seed=args.seed, max_episode_steps=args.max_episode_steps,
        headless=True, camera_names=config.camera_names, device=args.device,
    )

    success_count = 0
    total_reward = 0.0
    total_steps = 0
    results = []
    eval_seed_start = 100000 * (1 + args.seed)

    for ep in range(args.num_episodes):
        try:
            env._current_seed = eval_seed_start + ep
            obs = env.reset()
            ep_reward = 0.0
            ep_steps = 0

            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    h = feature_extractor.extract(obs, z_mode="zero")
                    h_t = torch.from_numpy(h).float().to(args.device).unsqueeze(0)
                    _, _, mu_action = actor.sample(h_t, deterministic=True)
                    action = mu_action.squeeze(0).cpu().numpy()

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1

                if done:
                    break

            success = info.get("success", False)
            if success:
                success_count += 1
            total_reward += ep_reward
            total_steps += ep_steps
            results.append({"episode": ep, "success": success, "reward": float(ep_reward), "steps": ep_steps})

            status = "\033[92mOK\033[0m" if success else "\033[91mFAIL\033[0m"
            print(f"[Eval] Ep {ep}: {status} reward={ep_reward:.2f} steps={ep_steps}")

        except Exception as e:
            print(f"[Eval] Episode {ep} error: {e}")
            try:
                env.close()
            except Exception:
                pass
            continue

    env.close()

    success_rate = success_count / max(args.num_episodes, 1)
    avg_reward = total_reward / max(args.num_episodes, 1)
    avg_steps = total_steps / max(args.num_episodes, 1)

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Task:           {args.task_name} ({args.task_config})")
    print(f"Episodes:       {args.num_episodes}")
    print(f"Success rate:   {success_rate:.2%} ({success_count}/{args.num_episodes})")
    print(f"Avg reward:     {avg_reward:.2f}")
    print(f"Avg steps:      {avg_steps:.1f}")
    print(f"{'='*60}")

    if args.output_file:
        output = {"config": config.to_dict(), "results": results,
                  "summary": {"success_rate": success_rate, "avg_reward": avg_reward,
                              "avg_steps": avg_steps, "num_episodes": args.num_episodes}}
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Eval] Results saved to {args.output_file}")

    return success_rate, avg_reward, results


def main():
    args = parse_args()
    act_model, actor, stats, config = load_model(args.sac_ckpt_path, args.device)
    evaluate(act_model, actor, stats, config, args)


if __name__ == "__main__":
    main()
