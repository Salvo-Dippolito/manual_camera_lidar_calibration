#!/usr/bin/env python3
import sys
import os
import rospy
from xml.etree import ElementTree as ET
from PyQt5 import QtWidgets, QtCore
import tf.transformations as tft
import tf
from sensor_msgs.msg import JointState
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.optimize import root

def rotvec_from_matrix(R_mat):
    """Compute rotation vector from rotation matrix (compatible with old scipy)."""
    # Compute angle
    angle = np.arccos(np.clip((np.trace(R_mat) - 1) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    # Compute rotation axis
    rx = (R_mat[2,1] - R_mat[1,2]) / (2*np.sin(angle))
    ry = (R_mat[0,2] - R_mat[2,0]) / (2*np.sin(angle))
    rz = (R_mat[1,0] - R_mat[0,1]) / (2*np.sin(angle))
    axis = np.array([rx, ry, rz])
    return axis * angle

def rotation_error(angles, R_target, axes):
    a1, a2, a3 = axes
    R1 = R.from_rotvec(angles[0]*a1).as_dcm()  # old scipy
    R2 = R.from_rotvec(angles[1]*a2).as_dcm()
    R3 = R.from_rotvec(angles[2]*a3).as_dcm()
    R_est = R1 @ R2 @ R3
    R_diff = R_est.T @ R_target
    return rotvec_from_matrix(R_diff)  # returns shape (3,)

def decompose_rotation_arbitrary_axes(R_target, axes, initial_guess=None):
    """
    Decompose a 3x3 rotation matrix into three rotations about arbitrary axes.
    Returns angles [theta1, theta2, theta3] in radians.
    
    R_target: 3x3 rotation matrix
    axes: list of 3 unit vectors in LiDAR frame
    initial_guess: optional starting guess [theta1, theta2, theta3]
    """
    if initial_guess is None:
        initial_guess = [0.0, 0.0, 0.0]
    
    sol = root(rotation_error, initial_guess, args=(R_target, axes), method='hybr')
    
    if not sol.success:
        raise RuntimeError(f"Rotation decomposition did not converge: {sol.message}")
    
    return sol.x  # theta1, theta2, theta3
    

class CalibratorGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        rospy.init_node("camera_lidar_calibrator_gui")

        # Joint and link chain
        self.joint_names = [
            "lidar_to_cam_x",
            "cam_x_to_cam_y",
            "cam_y_to_cam_z",
            "cam_z_to_cam_roll",
            "cam_roll_to_cam_pitch",
            "cam_pitch_to_cam_yaw",
        ]
        self.link_chain = [
            "lidar_frame", 
            "cam_x_link",
            "cam_y_link",
            "cam_z_link",
            "cam_roll_link",
            "cam_pitch_link",
            "cam_yaw_link",
            "camera",
        ]

        # User adjustment angles (applied before of convention conversion in static_joints.urdf)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.step_xyz = 0.0001  #[m]
        self.step_rpy = 0.00005 #[rad]

        self.parent = "lidar_frame"
        self.child = "camera"

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

        self.load_initial_from_urdf()
        self.build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_all)
        self.timer.start(20)

    # ---------------- UI ----------------
    def build_ui(self):
        self.setWindowTitle("Camera–LiDAR TF Calibrator")
        layout = QtWidgets.QVBoxLayout()
        self.spinboxes = {}

        def add_row(label, key, step):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(step)

            minus_btn = QtWidgets.QPushButton("-")
            plus_btn = QtWidgets.QPushButton("+")

            minus_btn.clicked.connect(lambda: self.increment_value(key, -step))
            plus_btn.clicked.connect(lambda: self.increment_value(key, step))
            spin.valueChanged.connect(lambda val, k=key: self.set_value(k, val))

            row.addWidget(lbl)
            row.addWidget(minus_btn)
            row.addWidget(spin)
            row.addWidget(plus_btn)
            layout.addLayout(row)
            self.spinboxes[key] = spin

        # X, Y, Z, Roll, Pitch, Yaw
        add_row("X (m)", "x", self.step_xyz)
        add_row("Y (m)", "y", self.step_xyz)
        add_row("Z (m)", "z", self.step_xyz)
        add_row("Roll (rad)", "roll", self.step_rpy)
        add_row("Pitch (rad)", "pitch", self.step_rpy)
        add_row("Yaw (rad)", "yaw", self.step_rpy)

        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save URDF")
        save_btn.clicked.connect(self.save_urdf)
        reload_btn = QtWidgets.QPushButton("Reload from URDF")
        reload_btn.clicked.connect(self.reload_from_urdf)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(reload_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.refresh_spinboxes()

    # ---------------- State helpers ----------------
    def set_value(self, key, value):
        setattr(self, key, value)

    def increment_value(self, key, delta):
        setattr(self, key, getattr(self, key) + delta)
        self.refresh_spinboxes()

    def refresh_spinboxes(self):
        for key, spin in self.spinboxes.items():
            spin.blockSignals(True)
            spin.setValue(getattr(self, key))
            spin.blockSignals(False)

    # ---------------- ROS publishing ----------------
    def publish_all(self):
        if rospy.is_shutdown():
            return


        joint_positions = [
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,  
            self.yaw,    
        ]

        # Publish JointState
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = joint_positions
        self.joint_pub.publish(msg)

        # Publish TF chain
        # parents = self.link_chain[:-1]
        # children = self.link_chain[1:]
        # rospy.loginfo_once(f"Publishing TF chain: {parents} -> {children}")
        # now = rospy.Time.now()

        # # Prismatic links
        # self.tf_broadcaster.sendTransform((joint_positions[0], 0, 0),
        #                                   tft.quaternion_from_euler(0, 0, 0),
        #                                   now, children[0], parents[0])
        # self.tf_broadcaster.sendTransform((0, joint_positions[1], 0),
        #                                   tft.quaternion_from_euler(0, 0, 0),
        #                                   now, children[1], parents[1])
        # self.tf_broadcaster.sendTransform((0, 0, joint_positions[2]),
        #                                   tft.quaternion_from_euler(0, 0, 0),
        #                                   now, children[2], parents[2])

        # # Revolute links
        # self.tf_broadcaster.sendTransform((0, 0, 0),
        #                                   tft.quaternion_from_euler(joint_positions[3], 0, 0, axes="sxyz"),
        #                                   now, children[3], parents[3])
        # self.tf_broadcaster.sendTransform((0, 0, 0),
        #                                   tft.quaternion_from_euler(0, joint_positions[4], 0, axes="sxyz"),
        #                                   now, children[4], parents[4])
        # self.tf_broadcaster.sendTransform((0, 0, 0),
        #                                   tft.quaternion_from_euler(0, 0, joint_positions[5], axes="sxyz"),
        #                                   now, children[5], parents[5])
        # Fixed camera
        # self.tf_broadcaster.sendTransform((0, 0, 0),
        #                                   tft.quaternion_from_euler(0, 0, 0),
        #                                   now, children[6], parents[6])

    # ---------------- URDF handling ----------------
    def load_initial_from_urdf(self):
        default_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "urdf", "camera_rig_ideal.urdf")
        )
        urdf_path = rospy.get_param("~ideal_urdf_path", default_path)
        if not os.path.exists(urdf_path):
            rospy.logwarn("Ideal URDF not found: %s", urdf_path)
            return

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            def read_joint_origin(j):
                origin = j.find("origin")
                xyz_vals = [float(v) for v in origin.attrib["xyz"].split()]
                rpy_vals = [float(v) for v in origin.attrib["rpy"].split()]
                return xyz_vals, rpy_vals

            joint = root.find("./joint[@name='camera_to_lidar']")

            if joint is not None:

                xyz_cl, rpy_cl = read_joint_origin(joint)

                # Convert cam->lidar into lidar->cam with convention change (switch from camera frame convention to lidar frame convention)
                #cam->lidar means xyz are lidar position with respect to camera frame
                #xyz_cl= - R_cl * xyz_lc  

                self.x = -xyz_cl[2]
                self.y = xyz_cl[0]
                self.z = xyz_cl[1]

                #switch from camera frame convention to lidar frame convention but using urdf rpy angles (sxyz) and not the convention used in static_joints.urdf (zyx)
                self.roll = rpy_cl[0] - np.pi/2
                self.pitch = rpy_cl[1] + np.pi/2
                self.yaw = rpy_cl[2] 

                rospy.loginfo("Loaded initial transform from URDF joint 'camera_to_lidar'")


            if joint is None:
                rospy.logwarn("Neither joint 'lidar_to_camera' nor 'camera_to_lidar' was found")
                return

            # Invert camera -> lidar to get lidar -> camer
            rospy.loginfo("Loaded initial transform from URDF joint 'camera_to_lidar' (inverted)")
        except Exception as e:
            rospy.logwarn(f"URDF load failed: {e}")

    def reload_from_urdf(self):
        self.load_initial_from_urdf()
        self.refresh_spinboxes()

    def save_urdf(self):

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll), np.cos(self.roll)]
        ])
        
        Ry = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])
        
        Rz = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        R_adjust = Rz @ Ry @ Rx

        #rotates from camera frame to lidar frame
        R_lc=np.array([[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]]) 
        
        #rotates from lidar frame to camera frame
        R_cl=R_lc.T

        rospy.loginfo("R_adjust:\n%s, R_lc:\n%s\n\n", R_adjust, R_lc)


        R_target = R_cl @ R_adjust #rotation from lidar to camera frame, including gui adjustment

        rospy.loginfo("R_target:\n%s\n\n", R_target)
        # R_target=R_target.T
        # Define three arbitrary axes in LiDAR frame
        axes = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]

        # Decompose
        # rpy_lc = decompose_rotation_arbitrary_axes(R_target, axes)
        rpy_cl = tft.euler_from_matrix(R_target, axes="sxyz")  # returns roll, pitch, yaw in radians
        print("Arbitrary axes angles (radians):", rpy_cl)

        # # Verify
        # R_check = R.from_rotvec(rpy_lc[0]*axes[0]).as_dcm() @ \
        #         R.from_rotvec(rpy_lc[1]*axes[1]).as_dcm() @ \
        #         R.from_rotvec(rpy_lc[2]*axes[2]).as_dcm()

        # print("Check close to target:", np.allclose(R_check, R_target, atol=1e-8))

        # switch from lidar frame convention to camera frame convention
        # rpy_cl = (rpy_lc[0] + np.pi/2, rpy_lc[1] - np.pi/2, rpy_lc[2])

        #print warning message with final angles and convention used in URDF (camera->lidar) and that internal state is lidar->camera
        rospy.logwarn(f"Saving URDF with camera->lidar convention. Internal state is lidar->camera. Final angles: roll={rpy_cl[0]:.6f}, pitch={rpy_cl[1]:.6f}, yaw={rpy_cl[2]:.6f}")

        #O_cl=-R_cl * O_lc
        xyz_cl = (self.y, self.z, -self.x)


        default_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "urdf", "camera_rig_ideal.urdf")
        )
        urdf_path = rospy.get_param("~ideal_urdf_path", default_path)
        if not os.path.exists(urdf_path):
            rospy.logwarn("Ideal URDF not found, cannot save.")
            return
        save_path = os.path.join(os.path.dirname(urdf_path), "camera_rig_calibrated.urdf")

        # Copy the original URDF to the new file
        import shutil
        shutil.copyfile(urdf_path, save_path)

        # Read the copied URDF
        with open(save_path, "r") as f:
            urdf_lines = f.readlines()

        # Prepare the new joint string (camera -> lidar)
        new_joint = f'''<joint name="camera_to_lidar" type="fixed">
  <parent link="camera"/>
  <child link="lidar_frame"/>
  <origin xyz="{xyz_cl[0]:.6f} {xyz_cl[1]:.6f} {xyz_cl[2]:.6f}" rpy="{rpy_cl[0]:.6f} {rpy_cl[1]:.6f} {rpy_cl[2]:.6f}"/>
</joint>\n'''

        # Rewrite only the camera_to_lidar joint in the copy
        inside_joint = False
        output_lines = []
        replaced = False
        for line in urdf_lines:
            if '<joint' in line and 'name="camera_to_lidar"' in line:
                inside_joint = True
                output_lines.append(new_joint)
                replaced = True
                continue
            if inside_joint:
                if '</joint>' in line:
                    inside_joint = False
                continue
            output_lines.append(line)

        # If the joint was not found, append it at the end
        if not replaced:
            output_lines.append(new_joint)

        with open(save_path, "w") as f:
            f.writelines(output_lines)
        rospy.loginfo(f"URDF saved to {save_path}")


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CalibratorGUI()
    window.show()
    sys.exit(app.exec_())
