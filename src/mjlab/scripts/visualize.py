import os
import sys
import time
import argparse
sys.path.append(os.getcwd())
import numpy as np
import mujoco
import mujoco.viewer
import joblib


def add_visual_capsule(scene, point1, point2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom - 1],
                         mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         point1, point2)


def key_call_back(keycode):
    global time_step, paused
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused


def main():
    parser = argparse.ArgumentParser(description="Visualize motion data")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to motion pkl/npz file")
    parser.add_argument("--asset_path", type=str, default="src/mjlab/asset_zoo/robots/adam_sp/adam_sp.xml", help="Path to robot XML")
    args = parser.parse_args()

    motion_file = args.motion_file
    asset_path = args.asset_path

    global time_step, dt, paused, fps, root_pos, root_rot, dof_pos
    time_step = 0
    paused = False

    # 自动检测文件格式并加载
    if motion_file.endswith(".npz"):
        # npz格式 (mjlab)
        motion_data = np.load(motion_file)
        fps = 50  # 默认50fps
        dt = 1 / fps
        root_pos = motion_data["body_pos_w"][:, 0]  # 第一个body的位置
        # print(root_pos)
        # print(root_pos.shape)
        root_rot = motion_data["body_quat_w"][:, 0]  # 第一个body的旋转 (wxyz格式，无需重排)
        dof_pos = motion_data["joint_pos"]
    else:
        # pkl格式
        motion_data = joblib.load(motion_file)
        fps = motion_data["fps"]
        dt = 1 / fps
        root_pos = motion_data["root_pos"]
        root_rot = motion_data["root_rot"]
        dof_pos = motion_data["dof_pos"]

    print(f"Playing motion")
    print(f"FPS: {fps}, Frames: {dof_pos.shape[0]}")

    # 加载 MuJoCo 模型
    mj_model = mujoco.MjModel.from_xml_path(asset_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        while viewer.is_running():
            step_start = time.time()
            curr_time = int(time_step / dt) % dof_pos.shape[0]

            mj_data.qpos[:3] = root_pos[curr_time]
            if motion_file.endswith(".npz"):
                mj_data.qpos[3:7] = root_rot[curr_time]  # npz: wxyz
            else:
                mj_data.qpos[3:7] = root_rot[curr_time][[3, 0, 1, 2]]  # pkl: xyzw -> wxyz
            mj_data.qpos[7:] = dof_pos[curr_time]

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()