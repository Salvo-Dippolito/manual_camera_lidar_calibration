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
            joint = root.find("./joint[@name='lidar_to_camera']")
            if joint is None:
                joint = root.find("./joint[@name='camera_to_lidar']")
                if joint is None:
                    rospy.logwarn("Joint 'camera_to_lidar' not found either")
                    return
                else:
                    origin = joint.find("origin")
                    xyz = [float(v) for v in origin.attrib["xyz"].split()]
                    rpy = [float(v) for v in origin.attrib["rpy"].split()]

                    self.roll, self.pitch, self.yaw = rpy[0]+np.pi/2, rpy[1]+np.pi/2, rpy[2]-np.pi
                    rospy.loginfo("Loaded initial transform from URDF (inverted)")
                    return
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.attrib["xyz"].split()]
            rpy = [float(v) for v in origin.attrib["rpy"].split()]
            self.x, self.y, self.z = xyz
            self.roll, self.pitch, self.yaw = rpy[0]+np.pi/2, rpy[1]+np.pi/2, rpy[2]-np.pi
            rospy.loginfo("Loaded initial transform from URDF")
        except Exception as e:
            rospy.logwarn(f"URDF load failed: {e}")

    def reload_from_urdf(self):
        self.load_initial_from_urdf()
        self.refresh_spinboxes()

    def save_urdf(self):

        rpy = self.roll - np.pi/2, self.pitch - np.pi/2, self.yaw + np.pi

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

        # Prepare the new joint string
        new_joint = f'''<joint name="lidar_to_camera" type="fixed">
  <parent link="{self.parent}"/>
  <child link="{self.child}"/>
  <origin xyz="{self.x:.6f} {self.y:.6f} {self.z:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>
</joint>\n'''

        # Rewrite only the lidar_to_camera joint in the copy
        inside_joint = False
        output_lines = []
        replaced = False
        for line in urdf_lines:
            if '<joint' in line and 'name="lidar_to_camera"' in line:
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
