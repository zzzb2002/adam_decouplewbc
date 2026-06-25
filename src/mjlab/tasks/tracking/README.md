# Adam-SP Motion Tracking 任务

训练 Adam-SP 人形机器人跟踪人类运动数据（Motion Tracking）。

## 快速开始

### 1. 单个运动数据

```bash
python src/mjlab/scripts/train.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --env.commands.motion.motion_file=src/assets/motions/adam_sp/EricCamper04_stageii.npz \
  --env.scene.num_envs=4096 \
  --gpu-ids [0] 
```

### 2. 多个运动数据

```bash
# 方式1：指定多个文件
python src/mjlab/scripts/train.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --env.commands.motion.motion-files="('src/assets/motions/adam_sp/0-strat_runing2_Skeleton_009_z_up_x_forward_gym.npz','src/assets/motions/adam_sp/EricCamper04_stageii.npz')" \
  --env.scene.num_envs=4096 \
  --gpu-ids [0]

# 方式2：指定目录（加载目录下所有npz）
python src/mjlab/scripts/train.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --env.commands.motion.motion_dir=src/assets/motions/adam_sp \
  --env.scene.num_envs=4096 \
  --gpu-ids [0]
```

训练完成后播放查看：
```bash
python src/mjlab/scripts/play.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --motion_file=src/assets/motions/adam_sp/EricCamper04_stageii.npz \
  --checkpoint_file=logs/rsl_rl/adam_sp_tracking/2026-05-10_17-43-43_wo_obs_motion_anchor_b_[adam_sp]/model_29999.pt \
  --video=True \
  --video_length=1000 \
  --video_height=1080 \
  --video_width=1920
```

训练完成后基于 Web 的 3D 可视化：
```bash
python src/mjlab/scripts/play.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --checkpoint_file=logs/rsl_rl/adam_sp_tracking/2026-05-25_11-54-45_wo_obs_motion_anchor_b_[adam_sp_20_data]_[2048_2048_1024_1024_512_512]/model_13000.pt \
  --motion_file=src/assets/motions/adam_sp/EricCamper04_stageii.npz \
  --viewer viser
```

### 3. 多 GPU 训练

`--gpu-ids` 支持指定单个或多个 GPU（按 `CUDA_VISIBLE_DEVICES` 中的索引）：

```bash
# 单 GPU（默认）
--gpu-ids [0]

# 多 GPU
--gpu-ids [0,1]

# 使用所有可用 GPU
--gpu-ids all

# CPU 训练
--gpu-ids None
```

## 数据转换流程

### 原始数据格式

```
原始pkl文件 (GMR) → 转换工具 → CSV → 转换工具 → NPZ
```

### 1. batch_gmr_pkl_to_csv.py

将GMR输出的pkl文件批量转换为CSV格式。

**输入**：pkl文件目录
**输出**：同目录下的csv子目录

```bash
python src/mjlab/scripts/batch_gmr_pkl_to_csv.py --folder <pkl文件所在目录>
```

示例：
```bash
python src/mjlab/scripts/batch_gmr_pkl_to_csv.py --folder src/input_pkl/adam_sp
```

输出保存到：`src/input_pkl/adam_sp/csv/`

### 2. csv_to_npz.py

将CSV文件转换为NPZ格式（训练用）。

**输入**：CSV文件
**输出**：NPZ文件

注意：这里需要指定正确的input-fps。如果CSV是从pkl转换来的：
- pkl > 30fps：CSV会被batch_gmr_pkl_to_csv.py下采样到30fps
- pkl ≤ 30fps：CSV保持原始fps

```bash
python src/mjlab/scripts/csv_to_npz.py \
  --input-file <输入CSV> \
  --output-name <输出文件名> \
  --input-fps <输入FPS> \
  --output-fps <输出FPS> \
  --robot adam_sp
```

示例：
```bash
# CSV是30fps -> 转为50fps
python src/mjlab/scripts/csv_to_npz.py \
  --input-file src/output_pkl/zb_GMR/csv/EricCamper04_stageii.csv \
  --output-name EricCamper04_stageii.npz \
  --input-fps 30 \
  --output-fps 50 \
  --robot adam_sp

# CSV是60fps原始 -> 转为50fps（记得改成60）
python src/mjlab/scripts/csv_to_npz.py \
  --input-file src/assets/motions/adam_sp/motion.csv \
  --output-name motion.npz \
  --input-fps 60 \
  --output-fps 50 \
  --robot adam_sp
```

输出保存到：`src/assets/motions/adam_sp/`

### 3. read_pkl.py

读取并查看pkl文件的结构（用于调试）。

```bash
python src/mjlab/scripts/read_pkl.py
```

默认读取：`src/input_pkl/CLOT2GMR/0-dance2_Skeleton_007_z_up_x_forward_gym.pkl`

修改文件路径：
```python
file_path = "src/input_pkl/CLOT2GMR/0-dance2_Skeleton_007_z_up_x_forward_gym.pkl"
```

### 4. visualize.py

可视化pkl/npz运动数据。

```bash
python src/mjlab/scripts/visualize.py --motion_file <pkl/npz文件>
```

示例：
```bash
python src/mjlab/scripts/visualize.py --motion_file src/output_pkl/zb_GMR/EricCamper04_stageii.pkl
```

## 数据存放位置

### 输入

| 类型 | 位置 |
|---|---|
| GMR原始pkl | `src/input_pkl/adam_sp/` 或 `src/input_pkl/CLOT2GMR/` |
| 中间CSV | `src/input_pkl/adam_sp/csv/` 或指定目录 |

### 输出

| 类型 | 位置 |
|---|---|
| NPZ运动数据 | `src/assets/motions/adam_sp/` |

## 配置参数

### 采样模式

在 `MotionCommandCfg` 中配置：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `sampling_mode` | 采样模式：`adaptive`（自适应）, `uniform`（均匀）, `start`（从头开始） | `adaptive` |
| `adaptive_kernel_size` | 自适应采样的kernel大小 | 1 |
| `adaptive_lambda` | 平滑kernel的衰减系数 | 0.8 |
| `adaptive_uniform_ratio` | 均匀采样的比例 | 0.1 |
| `adaptive_alpha` | 指数移动平均系数 | 0.001 |

### 姿态扰动范围

```python
pose_range={
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
},
velocity_range={
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
},
joint_position_range=(-0.1, 0.1),  # 关节位置扰动范围
```

## 运动数据格式

参考运动数据为 `.npz` 文件，包含以下数组：

| 键 | 形状 | 说明 |
|---|---|---|
| `joint_pos` | (N, D) | 关节位置 |
| `joint_vel` | (N, D) | 关节速度 |
| `body_pos_w` | (N, B, 3) | 身体部位世界位置 |
| `body_quat_w` | (N, B, 4) | 身体部位世界四元数 |
| `body_lin_vel_w` | (N, B, 3) | 身体部位线速度 |
| `body_ang_vel_w` | (N, B, 3) | 身体部位角速度 |

其中 N 为帧数，D 为关节数，B 为身体部位数。

## 自适应采样

当使用多个运动数据时，系统按时间将每个 clip 均匀划分为多个 bin，统计每个 bin 内的失败（termination）次数，并让失败率高的 bin 获得更高的采样概率，从而使 agent 专注于学习更难的动作片段。

目前有两种多 clip 自适应采样策略，通过切换 `commands` 文件来选择：

| 策略 | 对应文件 | 核心思想 |
|---|---|---|
| **Per-Clip Adaptive** | `src/mjlab/tasks/tracking/mdp/commands_per_clip.py` | 每个 clip 独立统计失败、独立采样；clip 之间信息不互通，所有 clip 被选中的概率始终相等。 |
| **Global Adaptive** | `src/mjlab/tasks/tracking/mdp/commands_sonic.py` | 所有 clip 的 bin 拼接成一个全局空间，统一统计失败、统一采样；难度高的 clip/时间段会获得更高的全局采样概率。 |

### 如何选择

- ** motions 异构**（如走路、转身、侧步等不同动作类型）→ **Per-Clip Adaptive** 更稳妥。它能保证每个动作类型都有足够的训练样本，避免难的动作战利品（starve）简单动作。
- ** motions 同构**（如多个同类型走路的捕捉片段）→ **Global Adaptive** 更有价值。难度信息可以在不同录制之间共享，全局统一聚焦于所有录制中最难的时间段。

切换方式：在环境配置中把 `commands_per_clip.py` 或者 `commands_sonic.py` 代码粘贴到 `commands.py` 即可。

## 查看日志

```bash
# TensorBoard
tensorboard --logdir logs/rsl_rl/adam_sp_tracking/
```

## 常用命令

```bash
# 训练
python src/mjlab/scripts/train.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation

# 播放并录制视频
python src/mjlab/scripts/play.py Mjlab-Tracking-Flat-Adam-Sp-No-State-Estimation \
  --motion_file=src/assets/motions/adam_sp/EricCamper04_stageii.npz \
  --checkpoint_file=logs/rsl_rl/adam_sp_tracking/xxx/model_29999.pt \
  --video=True \
  --video_length=1000 \
  --video_height=1080 \
  --video_width=1920

# 调整环境数量
--env.scene.num_envs=4096

# 调整训练迭代次数
--agent.max_iterations=50000
```