# manual_camera_lidar_calib

A ROS (Noetic) package for **manually calibrating the rigid transform between a camera and a LiDAR** using a real-time PyQt5 GUI. The user adjusts X, Y, Z, Roll, Pitch, and Yaw in the LiDAR frame while observing the alignment in RViz, then saves the result as a URDF.

---

## Overview

The package provides two complementary workflows for camera–LiDAR extrinsic calibration:

1. **GUI-based adjustment** (`manual_adjustment.launch`) — A custom PyQt5 GUI (`move_camera_frame.py`) lets you fine-tune 6-DOF parameters with adjustable step sizes. The GUI publishes `/joint_states` that drive a kinematic chain in a companion URDF, so changes are reflected live in RViz.

2. **Slider-based adjustment** (`manual_adjustment_joint_state_gui.launch`) — Uses the standard `joint_state_publisher_gui` for quick exploration via sliders.

Both modes use `robot_state_publisher` to broadcast TF from the kinematic-chain URDF (`camera_rig_static_joints.urdf`), giving real-time visual feedback of the camera↔LiDAR alignment.

---

## Package Structure

```
manual_camera_lidar_calib/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/                          # (reserved for RViz / parameter configs)
├── launch/
│   ├── manual_adjustment.launch                # PyQt5 GUI + robot_state_publisher
│   └── manual_adjustment_joint_state_gui.launch # joint_state_publisher_gui variant
├── scripts/
│   └── move_camera_frame.py         # Main calibration GUI node
├── src/
├── urdf/
|   ├── camera_rig_static_joints.urdf   # Kinematic chain URDF (used at runtime)
|   ├── camera_rig_ideal.urdf           # Source/template URDF with fixed joints
|   └── camera_rig_calibrated.urdf      # Output: saved calibration result
└── rviz/
    └── camera_lidar_manual_calib_config.rviz   # Rviz configuration file to visualize camera lidar alignment 
```

---

## URDF Files

### `camera_rig_static_joints.urdf` (runtime kinematic chain)

Defines the adjustable chain driven by `/joint_states`:

```
lidar_frame
  └─ [prismatic X] cam_x_link
       └─ [prismatic Y] cam_y_link
            └─ [prismatic Z] cam_z_link
                 └─ [revolute X] cam_roll_link
                      └─ [revolute Y] cam_pitch_link
                           └─ [revolute Z] cam_yaw_link
                                └─ [fixed] camera
```

The fixed joint `cam_yaw_to_camera` applies the convention change from LiDAR axes to camera optical axes via `rpy="-π/2  0  -π/2"` (static axes urdf convention).

Also contains a static `lidar_to_imu` joint for the Livox MID-360 IMU frame (`livox_frame`).

### `camera_rig_ideal.urdf` (template for save/load)

A flat URDF with fixed joints used as the template. Contains:

- `camera_lidar_rig` → `camera` (hand-eye calibration from Vicon)
- `camera` → `lidar_frame` (the `camera_to_lidar` joint — **this is what the GUI edits**)
- `lidar_frame` → `livox_frame` (Livox IMU extrinsics)

When you press **Save URDF**, the `camera_to_lidar` joint is updated and written to `camera_rig_calibrated.urdf`.

### `camera_rig_calibrated.urdf` (output)

A copy of `camera_rig_ideal.urdf` with the `camera_to_lidar` joint replaced by the calibrated values. Load this into a `robot_state_publisher` for your final pipeline.

---

## Frame Conventions

| Frame | Convention | Axes |
|-------|-----------|------|
| `lidar_frame` | LiDAR / ROS standard | X-forward, Y-left, Z-up |
| `camera` | Camera optical | X-right, Y-down, Z-forward |

The convention change between them is encoded in the fixed rotation matrix:

```
R_lc (camera axes → lidar axes) = [ 0  0  1 ]
                                   [-1  0  0 ]
                                   [ 0 -1  0 ]
```

### Internal GUI state vs. URDF convention

| | GUI (internal state) | URDF (`camera_to_lidar` joint) |
|---|---|---|
| **Frame** | LiDAR frame | Camera frame |
| **Direction** | LiDAR → Camera (adjustment) | Camera → LiDAR (parent → child) |
| **Rotation** | Intrinsic XYZ: Rx(roll) · Ry(pitch) · Rz(yaw) | Static XYZ euler angles (URDF standard) |
| **Translation** | Position of camera in LiDAR frame | Position of LiDAR in camera frame |

The `save_urdf()` and `load_initial_from_urdf()` methods handle all conversions between these two representations.

---

## Dependencies

### ROS packages

- `rospy`, `roscpp`, `sensor_msgs`, `std_msgs`
- `robot_state_publisher`
- `joint_state_publisher_gui` (only for the slider launch variant)
- `tf`

### Python

- `PyQt5`
- `numpy`
- `scipy`

---

## Usage

### 1. Launch the GUI calibrator

```bash
roslaunch manual_camera_lidar_calib manual_adjustment.launch
```

This starts:
- `move_camera_frame.py` — the PyQt5 calibration GUI
- `robot_state_publisher` — publishes TF from the kinematic chain URDF

### 2. Open RViz

Open RViz separately (or add it to the launch file) and load the rviz config file present in this pakage rviz folder.

### 3. Adjust the transform

The GUI provides:

| Control | Description |
|---------|-------------|
| **XYZ step (m)** | Log-scale slider to set translation increment (10⁻⁶ to 10⁻¹ m) |
| **RPY step (rad)** | Log-scale slider to set rotation increment (10⁻⁶ to 10⁻¹ rad) |
| **X / Y / Z** | Translation of the camera origin in the LiDAR frame (meters) |
| **Roll / Pitch / Yaw** | Rotation about X / Y / Z axes in the LiDAR frame (radians) |
| **+/−** buttons | Increment/decrement by the current step size |
| **Save URDF** | Write the calibration to `camera_rig_calibrated.urdf` |
| **Reload from URDF** | Re-read the ideal URDF and reset the GUI |

### 4. Save

Click **Save URDF**. The result is written to:

```
urdf/camera_rig_calibrated.urdf
```

The saved file includes an XML comment with the full rotation matrix and translation vector for easy inspection.

### 5. Use the calibrated URDF

Load the calibrated URDF into your pipeline's `robot_state_publisher`:

```xml
<param name="robot_description"
       textfile="$(find manual_camera_lidar_calib)/urdf/camera_rig_calibrated.urdf"/>
<node name="robot_state_publisher" pkg="robot_state_publisher"
      type="robot_state_publisher" output="screen"/>
```

---

## Alternative: Joint State Publisher GUI

For quick exploration without the custom GUI:

```bash
roslaunch manual_camera_lidar_calib manual_adjustment_joint_state_gui.launch
```

This uses `joint_state_publisher_gui` with sliders for all 6 joints. Note: this mode does **not** support saving to URDF.

---

## ROS API

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Joint positions for the 6-DOF kinematic chain |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~ideal_urdf_path` | `<pkg>/urdf/camera_rig_ideal.urdf` | Path to the template URDF for load/save |

---

## Mathematical Details

### Kinematic chain rotation

The three revolute joints apply rotations in **intrinsic XYZ** order (each rotation is about the current body-frame axis):

$$R_{\text{chain}} = R_x(\text{roll}) \cdot R_y(\text{pitch}) \cdot R_z(\text{yaw})$$

### Full camera-to-LiDAR rotation

The fixed joint at the end of the chain applies the convention change $R_{lc}$:

$$R_{\text{cam→lid}} = R_{\text{chain}} \cdot R_{lc}$$

### URDF joint rotation

The URDF `camera_to_lidar` joint (parent=camera, child=lidar) stores the **inverse** (child-to-parent = lidar-to-camera):

$$R_{\text{urdf}} = R_{\text{cam→lid}}^T$$

### URDF joint translation

$$t_{\text{urdf}} = -R_{\text{urdf}} \cdot t_{\text{chain}}$$

where $t_{\text{chain}} = (x, y, z)$ is the camera position in the LiDAR frame.
