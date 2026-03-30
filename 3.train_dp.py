"""
Diffusion Policy 训练程序
基于 LeRobot 框架的扩散策略模型训练
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps


# ============================================================================
# 1. 配置参数
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 训练参数
training_steps = 10000          # 总训练步数
log_freq = 100                  # 日志打印频率
save_freq = 1000                # 模型保存频率
learning_rate = 1e-4            # 学习率
batch_size = 32                 # 批次大小

# 数据集参数
dataset_repo = "fr3_pnp"        # 数据集名称
root_dir = '../lerobot_fr3sim_act/demo_data'        # 数据集根目录
checkpoint_path = './ckpt/dp_ee_pose' # 模型保存路径

# Diffusion Policy 特定参数
num_diffusion_steps = 100       # 扩散过程的步数(越多收敛越慢但效果更好，通常16-100)
beta_schedule = "linear"        # 噪声调度: linear, cosine, quad
prediction_type = "epsilon"     # 预测类型: epsilon(噪声) 或 sample(样本)


# ============================================================================
# 2. 数据增强定义
# ============================================================================
class AddGaussianNoise(object):
    """向图像Tensor添加高斯噪声，增加训练的鲁棒性"""
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# 建立图像增强pipeline
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))  # 确保像素值在[0,1]范围内
])


# ============================================================================
# 3. 采样器定义 (用于推理测试)
# ============================================================================
class EpisodeSampler(torch.utils.data.Sampler):
    """根据情节索引获取该情节的所有数据帧"""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # 根据情节索引获取数据帧的起止范围
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


# ============================================================================
# 4. 主训练函数
# ============================================================================
def main():
    print("=" * 80)
    print("Diffusion Policy 训练程序")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 4.1 初始化策略配置和模型
    # -----------------------------------------------------------------------
    print("\n[步骤1] 初始化策略配置...")

    # 加载数据集元信息
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo, root=root_dir)
    features = dataset_to_policy_features(dataset_metadata.features)

    # 分离输入特征（观察）和输出特征（动作）
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 移除不必要的观测特征（例如手腕视角）
    if "observation.wrist_image" in input_features:
        input_features.pop("observation.wrist_image")

    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")

    # 配置 Diffusion Policy
    # Diffusion Policy 通过逐步去噪来生成动作序列
    cfg = DiffusionConfig(
        # 模型结构参数
        input_features=input_features,
        output_features=output_features,

        # 关键参数：预测任务的动作长度
        horizon=16,                             # 决定了模型的 chunk_size (原代码中的 chunk_size=10)
        n_action_steps=10,                      # 模型实际生成的动作步数

        # Diffusion 特定参数
        num_train_timesteps=num_diffusion_steps, # 扩散链长度
        beta_schedule=beta_schedule,             # 噪声调度方式
        prediction_type=prediction_type,         # 预测噪声还是原始样本
    )

    # 计算需要的历史/未来时间戳
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    # 初始化策略模型
    print("  初始化 Diffusion Policy 模型...")
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()  # 设置为训练模式
    policy.to(device)

    print(f"  模型参数数量: {sum(p.numel() for p in policy.parameters()):,}")

    # -----------------------------------------------------------------------
    # 4.2 加载数据集
    # -----------------------------------------------------------------------
    print("\n[步骤2] 加载数据集...")
    dataset = LeRobotDataset(
        dataset_repo,
        delta_timestamps=delta_timestamps,
        root=root_dir,
        image_transforms=transform
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type != "cpu"),
        drop_last=True,
    )

    print(f"  数据集大小: {len(dataset)} 帧")
    print(f"  批次大小: {batch_size}")
    print(f"  总批数: {len(dataloader)}")

    # -----------------------------------------------------------------------
    # 4.3 训练循环
    # -----------------------------------------------------------------------
    print("\n[步骤3] 开始训练...")
    print(f"预计训练步数: {training_steps}\n")

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    step = 0
    done = False

    while not done:
        for batch in dataloader:
            # 将batch中的Tensor移动到指定设备
            inp_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            # Diffusion Policy 前向传递：
            # - 从真实动作添加噪声（forward diffusion）
            # - 训练网络预测噪声或去噪动作（reverse diffusion）
            loss, _ = policy.forward(inp_batch)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 日志输出
            if step % log_freq == 0:
                print(f"training step: {step:5d} | loss: {loss.item():.4f}")

            # 定期保存模型
            #if step % save_freq == 0 and step > 0:
                #policy.save_pretrained(checkpoint_path)
                #print(f"  → 模型已保存至 {checkpoint_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # -----------------------------------------------------------------------
    # 4.4 保存最终模型
    # -----------------------------------------------------------------------
    print(f"\n[步骤4] 保存最终模型...")
    policy.save_pretrained(checkpoint_path)
    print(f"✓ 模型已保存至 {checkpoint_path}")

    # -----------------------------------------------------------------------
    # 4.5 推理测试和可视化
    # -----------------------------------------------------------------------
    print(f"\n[步骤5] 推理测试...")
    policy.eval()
    actions_list = []
    gt_actions_list = []

    # 使用第一个episode进行推理测试
    episode_index = 0
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.type != "cpu"),
        sampler=episode_sampler,
    )

    # 重置policy的内部状态（如果需要）
    policy.reset()

    # 关闭梯度计算，加快推理速度
    with torch.no_grad():
        for batch in test_dataloader:
            inp_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Diffusion Policy 推理：
            # 通过反向扩散（从纯噪声逐步去噪）生成动作
            action = policy.select_action(inp_batch)
            actions_list.append(action)
            gt_actions_list.append(inp_batch["action"][:, 0, :])

    # 合并所有推理结果
    actions = torch.cat(actions_list, dim=0)
    gt_actions = torch.cat(gt_actions_list, dim=0)

    print(f"actions.shape: {actions.shape}, gt_actions.shape: {gt_actions.shape}")

    # 判断是否需要对齐前几个元素（有可能是返回了horizon，而预测只取一步或相反）
    if len(actions.shape) == 2 and len(gt_actions.shape) == 2 and actions.shape[1] != gt_actions.shape[1]:
        print("维度不一致，尝试对齐...", actions.shape, gt_actions.shape)
        # 例如 actions 可能是直接返回的单步动作向量，但 gt 可能包含了时间维度，或者两者含义不同
        min_dim = min(actions.shape[1], gt_actions.shape[1])
        # 暂时只取前 min_dim 进行测试，确保继续运行
        actions = actions[:, :min_dim]
        gt_actions = gt_actions[:, :min_dim]
    elif len(actions.shape) == 3 and len(gt_actions.shape) == 2:
        # action可能包含时间步 (B, n_steps, action_dim)
        actions = actions[:, 0, :]

    # 计算平均动作误差
    error = torch.mean(torch.abs(actions - gt_actions)).item()
    print(f"✓ 平均动作误差 (Mean Absolute Error): {error:.4f}")

    # -----------------------------------------------------------------------
    # 4.6 可视化预测结果
    # -----------------------------------------------------------------------
    print("\n[步骤6] 绘制推理结果...")

    # 可视化前7个自由度的关节动作
    action_dim = 7
    fig, axs = plt.subplots(action_dim, 1, figsize=(12, 10))

    for i in range(action_dim):
        # 约束输入：预测值（pred）vs 真实值（ground truth）
        axs[i].plot(
            actions[:, i].cpu().numpy(),
            label="预测动作 (Predicted)",
            linewidth=2,
            alpha=0.8
        )
        axs[i].plot(
            gt_actions[:, i].cpu().numpy(),
            label="真实动作 (Ground Truth)",
            linewidth=2,
            alpha=0.8,
            linestyle='--'
        )
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(f"关节 {i} 动作")
        axs[i].grid(True, alpha=0.3)

    axs[-1].set_xlabel("时间步")
    plt.suptitle(f"Diffusion Policy 推理结果 (MAE: {error:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{checkpoint_path}/inference_results.png", dpi=150, bbox_inches='tight')
    print(f"✓ 推理结果图已保存至 {checkpoint_path}/inference_results.png")
    plt.show()

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

