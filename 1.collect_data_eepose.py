import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SEED = None # 随机种子，用于固定(0)或随机(None)物体位置
# 【修改点1】：修改数据集名称，避免和之前omy机器人的数据冲突
REPO_NAME = 'fr3_pnp' 
NUM_DEMO = 10 # 采集的demo数量
ROOT = "./demo_data" # 数据保存的路径
TASK_NAME = 'Put mug cup on the plate' #任务描述文本
xml_path = './asset/example_scene_y.xml'#场景描述文件路径位置
# 设置虚拟环境
PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')#给机器人的信息是关节角度

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

#开始录数据
if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                # 【修改点2】：修改机器人类型标识
                robot_type="fr3",
                fps=20, # 用于后续训练进行时间对齐
                #lerobot的数据结构定义，说明要存什么类型的数据，这里定义了五种：
                #主视角图像、手腕视角图像、机器人当前状态、机器人的动作、物体初始位置
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    # get_ee_pose() 返回 [px, py, pz, roll, pitch, yaw]，共 6 维
                    # FR3 与 omy 的末端执行器位姿维度相同，无需修改
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # 末端执行器的位姿信息x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        # 【修改点3】：维度从 (7,) 改为 (8,)！
                        # omy 机器人：6个关节角 + 1个夹爪 = 7维
                        # FR3 机器人：7个关节角 + 1个夹爪 = 8维（get_joint_state() 返回值）
                        "shape": (8,),
                        "names": ["action"], # 7 joint angles and 1 gripper；这是模型要学习的目标
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # 记录任务开始时物体在哪 训练中不使用.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# 【修改点4】：初始化一个全零的空的七维动作向量
# teleop_robot() 在 eef_pose 模式下返回 7 维：[dx,dy,dz,dr,dp,dy,gripper]
action = np.zeros(7) 
episode_id = 0 # 成功次数计数器，从 0 开始
record_flag = False # 录制开关，等机器人发起动作再打开

while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO: # 循环条件：窗口没关且没录够'NUM_DEMO'次
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done: 
            # 做完了，存档，重置环境，计数器+1
            dataset.save_episode()
            PnPEnv.reset(seed = SEED)
            episode_id += 1
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset  = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0: #有动作开录
            record_flag = True
            print("Start recording")
        if reset:
            # 重置环境清理动作记录池
            # This can be done by pressing 'z' key
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # 记录末端执行器姿态和拍照
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)#执行动作，获取关节角度（7维）+ 夹爪状态（1维）= 8维
        if record_flag:
            # 把数据存进dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,  # 6维末端执行器位姿
                    "action": joint_q,             # 8维：7个关节角 + 1个夹爪状态
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": TASK_NAME,
                }, task = TASK_NAME
            )
        PnPEnv.render(teleop=True)

PnPEnv.env.close_viewer()

# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')
