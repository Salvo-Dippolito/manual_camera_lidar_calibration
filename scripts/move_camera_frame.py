#!/usr/bin/env python3
import sys
import os
import rospy
from xml.etree import ElementTree as ET
from PyQt5 import QtWidgets, QtCore, QtGui
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
        self.step_groups = {}  # key -> step_attr name ("step_xyz" or "step_rpy")

        # --- Step-size controls ---
        def add_step_slider(label, attr, min_exp, max_exp, layout_parent):
            """Add a log-scale slider + editable text box for step size.
            attr: attribute name on self (e.g. 'step_xyz')
            min_exp / max_exp: exponents of 10 (e.g. -6 .. -1)
            """
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            n_steps = (max_exp - min_exp) * 10  # 10 subdivisions per decade
            slider.setRange(0, n_steps)
            cur_exp = np.log10(getattr(self, attr))
            slider.setValue(int(round((cur_exp - min_exp) * 10)))

            line_edit = QtWidgets.QLineEdit(f"{getattr(self, attr):.6g}")
            line_edit.setFixedWidth(100)
            validator = QtGui.QDoubleValidator(10.0**min_exp, 10.0**max_exp, 8)
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            line_edit.setValidator(validator)

            # Prevent mutual signal loops
            _updating = [False]

            def on_slider(pos):
                if _updating[0]:
                    return
                _updating[0] = True
                exp = min_exp + pos / 10.0
                new_step = 10.0 ** exp
                setattr(self, attr, new_step)
                line_edit.setText(f"{new_step:.6g}")
                self._update_spinbox_steps()
                _updating[0] = False

            def on_text_edited():
                if _updating[0]:
                    return
                _updating[0] = True
                try:
                    val = float(line_edit.text())
                    if val > 0:
                        setattr(self, attr, val)
                        exp = np.log10(val)
                        pos = int(round((exp - min_exp) * 10))
                        pos = max(0, min(n_steps, pos))
                        slider.setValue(pos)
                        self._update_spinbox_steps()
                except ValueError:
                    pass
                _updating[0] = False

            slider.valueChanged.connect(on_slider)
            line_edit.editingFinished.connect(on_text_edited)

            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(line_edit)
            layout_parent.addLayout(row)

        add_step_slider("XYZ step (m)", "step_xyz", -6, -1, layout)
        add_step_slider("RPY step (rad)", "step_rpy", -6, -1, layout)

        # --- Value rows ---
        def add_row(label, key, step_attr):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(getattr(self, step_attr))

            minus_btn = QtWidgets.QPushButton("-")
            plus_btn = QtWidgets.QPushButton("+")

            minus_btn.clicked.connect(lambda _, k=key, sa=step_attr: self.increment_value(k, -getattr(self, sa)))
            plus_btn.clicked.connect(lambda _, k=key, sa=step_attr: self.increment_value(k, getattr(self, sa)))
            spin.valueChanged.connect(lambda val, k=key: self.set_value(k, val))

            row.addWidget(lbl)
            row.addWidget(minus_btn)
            row.addWidget(spin)
            row.addWidget(plus_btn)
            layout.addLayout(row)
            self.spinboxes[key] = spin
            self.step_groups[key] = step_attr

        # X, Y, Z, Roll, Pitch, Yaw
        add_row("X (m)", "x", "step_xyz")
        add_row("Y (m)", "y", "step_xyz")
        add_row("Z (m)", "z", "step_xyz")
        add_row("Roll (rad)", "roll", "step_rpy")
        add_row("Pitch (rad)", "pitch", "step_rpy")
        add_row("Yaw (rad)", "yaw", "step_rpy")

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

    def _update_spinbox_steps(self):
        """Sync spinbox singleStep with current step_xyz / step_rpy values."""
        for key, spin in self.spinboxes.items():
            step_attr = self.step_groups[key]
            spin.setSingleStep(getattr(self, step_attr))

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

                # Convention change matrices
                R_lc_ = np.array([[0, 0, 1],
                                  [-1, 0, 0],
                                  [0, -1, 0]])  # camera axes -> lidar axes
                R_cl_ = R_lc_.T                   # lidar axes -> camera axes

                # R_urdf from URDF (maps lidar->camera, i.e. child->parent)
                R_urdf = tft.euler_matrix(rpy_cl[0], rpy_cl[1], rpy_cl[2], axes="sxyz")[:3, :3]

                # Recover camera->lidar rotation and kinematic chain rotation
                R_cam2lid = R_urdf.T                      # camera->lidar
                R_chain_rot = R_cam2lid @ R_cl_           # remove convention change at end of chain

                # Decompose intrinsic XYZ (matching kinematic chain revolute joint order)
                self.roll, self.pitch, self.yaw = tft.euler_from_matrix(R_chain_rot, axes="rxyz")

                # Recover translation: t_chain = -R_cam2lid @ t_urdf
                t_urdf = np.array(xyz_cl)
                t_chain = -R_cam2lid @ t_urdf
                self.x, self.y, self.z = t_chain[0], t_chain[1], t_chain[2]


                rospy.loginfo("Loaded initial transform from URDF joint 'camera_to_lidar'")


            if joint is None:
                rospy.logwarn("Neither joint 'lidar_to_camera' nor 'camera_to_lidar' was found")
                return


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
        # Build chain rotation matching kinematic chain (intrinsic XYZ: Rx @ Ry @ Rz)
        R_chain_rot = Rx @ Ry @ Rz

        # Convention change: camera axes -> lidar axes
        R_lc = np.array([[0, 0, 1],
                         [-1, 0, 0],
                         [0, -1, 0]])

        # Full camera->lidar rotation (matches kinematic chain including fixed joint)
        R_cam2lid = R_chain_rot @ R_lc

        # URDF joint (parent=camera, child=lidar) needs lidar->camera rotation
        R_urdf = R_cam2lid.T

        rospy.loginfo("R_chain_rot:\n%s\nR_urdf:\n%s\n", R_chain_rot, R_urdf)

        rpy_cl = tft.euler_from_matrix(R_urdf, axes="sxyz")
        print("URDF rpy (radians):", rpy_cl)

        rospy.logwarn(f"Saving URDF camera_to_lidar joint. rpy={rpy_cl[0]:.6f}, {rpy_cl[1]:.6f}, {rpy_cl[2]:.6f}")

        # Translation: t_urdf = -R_urdf @ t_chain  (lidar origin in camera frame)
        t_chain = np.array([self.x, self.y, self.z])
        t_urdf = -R_urdf @ t_chain
        xyz_cl = (t_urdf[0], t_urdf[1], t_urdf[2])


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

        # Build full 3x3 rotation matrix (camera -> lidar) for the comment
        R_cl_full = tft.euler_matrix(rpy_cl[0], rpy_cl[1], rpy_cl[2], axes="sxyz")[:3, :3]

        # Prepare the new joint string (camera -> lidar) with rotation matrix & translation comment
        def fmt_row(row):
            return ", ".join(f"{v: .6f}" for v in row)

        new_joint = f'''    <!-- Camera to LiDAR  manual calibration
         R_cl (rotation matrix, camera to lidar)
           [{fmt_row(R_cl_full[0])}]
           [{fmt_row(R_cl_full[1])}]
           [{fmt_row(R_cl_full[2])}]
         t_cl (translation, camera to lidar)
           [{xyz_cl[0]:.6f}, {xyz_cl[1]:.6f}, {xyz_cl[2]:.6f}]
    -->
    <joint name="camera_to_lidar" type="fixed">
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
