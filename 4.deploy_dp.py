"""
Diffusion Policy 推理程序
使用训练好的 Diffusion Policy 模型在仿真环境中执行机器人控制任务

功能：
    1. 加载训练好的 Diffusion Policy 模型
    2. 在仿真环境中实时推理
    3. 记录任务成功/失败
    4. 支持多轮重置和重新执行
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

from mujoco_env.y_env import SimpleEnv


# ============================================================================
# 1. 配置参数
# ============================================================================

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 环境配置
xml_path = './asset/example_scene_y.xml'  # 仿真场景配置文件
action_type = 'joint_angle'                # 动作类型：关节角度
task_description = 'Put mug cup on the plate'  # 任务描述

# 模型配置
dataset_repo = 'fr3_pnp'                   # 数据集名称（用于加载统计信息）
root_dir = './demo_data'                  # 数据集路径
checkpoint_path = './ckpt/dp_ee_pose'           # 训练好的模型路径

# Diffusion Policy 推理参数
num_diffusion_steps = 100                  # 扩散步数（应与训练时一致）
beta_schedule = "linear"                   # 噪声调度
prediction_type = "epsilon"                # 预测类型


# ============================================================================
# 2. 图像变换定义
# ============================================================================
# 将 PIL 图像转换为 PyTorch Tensor
img_transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像 (H, W, C) 转换为 Tensor (C, H, W)，值域0-255→0-1
])


# ============================================================================
# 3. 初始化环境
# ============================================================================
print("\n[步骤1] 初始化仿真环境...")
PnPEnv = SimpleEnv(xml_path, action_type=action_type)
print("✓ 环境初始化完成")


# ============================================================================
# 4. 加载训练好的 Diffusion Policy 模型
# ============================================================================
print("\n[步骤2] 加载预训练模型...")

# 加载数据集元信息（用于获取统计信息和特征定义）
dataset_metadata = LeRobotDatasetMetadata(dataset_repo, root=root_dir)
features = dataset_to_policy_features(dataset_metadata.features)

# 分离输入特征（观察）和输出特征（动作）
output_features = {
    key: ft for key, ft in features.items()
    if ft.type is FeatureType.ACTION
}
input_features = {
    key: ft for key, ft in features.items()
    if key not in output_features
}

# 移除手腕视角（可选）
if "observation.wrist_image" in input_features:
    input_features.pop("observation.wrist_image")

print(f"  输入特征: {list(input_features.keys())}")
print(f"  输出特征: {list(output_features.keys())}")

# 配置 Diffusion Policy
cfg = DiffusionConfig(
    # 模型特征配置
    input_features=input_features,
    output_features=output_features,

    # 关键参数
    chunk_size=10,                  # 输出动作的未来步数
    n_action_steps=10,              # 生成的动作步数

    # Diffusion 特定参数
    num_diffusion_steps=num_diffusion_steps,
    beta_schedule=beta_schedule,
    prediction_type=prediction_type,

    # UNet架构参数
    down_block_types=("CrossAttnDownBlock2d", "CrossAttnDownBlock2d", "AttnDownBlock2d", "DownBlock2d"),
    up_block_types=("UpBlock2d", "AttnUpBlock2d", "CrossAttnUpBlock2d", "CrossAttnUpBlock2d"),
    block_out_channels=(320, 640, 1280, 1280),
    cross_attention_dim=256,
)

# 计算时间戳偏移
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# 从预训练模型加载策略
print(f"  从 {checkpoint_path} 加载模型...")
policy = DiffusionPolicy.from_pretrained(
    checkpoint_path,
    config=cfg,
    dataset_stats=dataset_metadata.stats
)
policy.to(device)
policy.eval()  # 设置为评估模式（关闭dropout等）

print("✓ 模型加载完成")
print(f"  模型参数数量: {sum(p.numel() for p in policy.parameters()):,}")


# ============================================================================
# 5. 推理主循环
# ============================================================================
print("\n[步骤3] 开始推理...")
print(f"任务: {task_description}\n")

success_count = 0
episode_count = 0
max_episodes = 5  # 最多运行5个episode

# 初始化环境和策略
step = 0
PnPEnv.reset(seed=0)
policy.reset()  # 重置policy的内部状态

with torch.no_grad():  # 关闭梯度计算，加快推理速度
    while PnPEnv.env.is_viewer_alive() and episode_count < max_episodes:
        # 物理仿真前进一步
        PnPEnv.step_env()

        # 以固定频率（20Hz）执行一次推理和控制
        if PnPEnv.env.loop_every(HZ=20):
            # ---------------------------------------------------------------
            # 检查任务完成
            # ---------------------------------------------------------------
            success = PnPEnv.check_success()
            if success:
                print(f"[Episode {episode_count+1}] ✓ 任务成功！")
                success_count += 1
                episode_count += 1

                # 重置环境和策略，准备下一轮
                policy.reset()
                PnPEnv.reset(seed=0)
                step = 0

                # 如果已完成所有episode，退出
                if episode_count >= max_episodes:
                    break
                continue

            # ---------------------------------------------------------------
            # 准备推理输入
            # ---------------------------------------------------------------
            # 获取末端执行器位姿（6维）
            ee_state = PnPEnv.get_ee_pose()

            # 获取相机图像并预处理
            agent_image, wrist_image = PnPEnv.grab_image()

            # 处理主视角图像
            agent_image = Image.fromarray(agent_image)
            agent_image = agent_image.resize((256, 256))  # 调整到模型输入大小
            agent_image = img_transform(agent_image)      # 转换为Tensor

            # 处理手腕视角图像（保持但在模型中未使用）
            wrist_image = Image.fromarray(wrist_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = img_transform(wrist_image)

            # 构建输入字典
            # 注意：Diffusion Policy 输入与 ACT 相同，但内部处理不同
            inference_input = {
                'observation.state': torch.tensor([ee_state], dtype=torch.float32).to(device),
                'observation.image': agent_image.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                'task': [task_description],
                'timestamp': torch.tensor([step / 20.0], dtype=torch.float32).to(device),
            }

            # ---------------------------------------------------------------
            # Diffusion Policy 推理
            # ---------------------------------------------------------------
            # 通过反向扩散过程生成动作
            # 1. 从纯噪声开始
            # 2. 逐步去噪 num_diffusion_steps 次
            # 3. 输出最终的动作预测
            action = policy.select_action(inference_input)

            # 提取动作并转换为 numpy
            action = action[0].cpu().detach().numpy()

            # ---------------------------------------------------------------
            # 执行动作
            # ---------------------------------------------------------------
            # 将动作发送给环境
            _ = PnPEnv.step(action)

            # 渲染可视化
            PnPEnv.render(teleop=False)

            step += 1

            # 每50步打印一次当前状态
            if step % 50 == 0:
                print(f"  [Episode {episode_count+1}] Step {step}: 运行中...")

            # 安全超时：如果超过300步还未完成，重置
            if step > 300:
                print(f"  [Episode {episode_count+1}] ⚠ 超时，重置环境")
                episode_count += 1
                policy.reset()
                PnPEnv.reset(seed=0)
                step = 0

print("\n" + "=" * 80)
print("推理完成！")
print(f"总Episode数: {episode_count}")
print(f"成功次数: {success_count}")
if episode_count > 0:
    success_rate = 100 * success_count / episode_count
    print(f"成功率: {success_rate:.1f}%")
print("=" * 80)

# 关闭环境
PnPEnv.env.close_viewer()
