import gym
from gym import spaces
import numpy as np
import rospy
import subprocess
import os
import time
import random
from std_msgs.msg import String, Int8
from px4_offb_ctrl.srv import setInitGlobalPose, setInitGlobalPoseRequest
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from quadrotor_msgs.msg import PositionCommand
import ros_numpy


class PX4OffbSingle(gym.Env):
    def __init__(self, takeoff_height=1.0, px4_offb_pkg_name="px4_offb_ctrl", launch_file="px4_offb_ctrl.launch", use_fastplanner=False, 
        use_px4_ctrl=False, px4_ctrl_mode="pva"):
        self.takeoff_height = takeoff_height
        self.px4_offb_pkg_path = None
        self.ros_subproc = None
        self.reset_init_pose_client = rospy.ServiceProxy("/reset_init_global_nwu_pose", setInitGlobalPose)
        try:
            p = subprocess.run(["rospack", "find", px4_offb_pkg_name], check=True, capture_output=True, encoding="utf-8")
            self.px4_offb_pkg_path = p.stdout.strip('\n')
        except subprocess.CalledProcessError:
            print(f"ROS pkg - {px4_offb_pkg_name} does not exist!")

        launch_file_path = os.path.join(self.px4_offb_pkg_path, "launch", launch_file)
        if not os.path.exists(launch_file_path):
            raise Exception(f"{launch_file_path} does not exist!")
        
        self.ros_subproc = subprocess.Popen(
            f"xterm -e roslaunch px4_offb_ctrl {launch_file} use_local_frame:=false use_fastplanner:={use_fastplanner} \
                px4_ctrl_mode:={px4_ctrl_mode} use_px4_ctrl:={use_px4_ctrl} takeoff_height:={self.takeoff_height}", 
            shell=True
            )

        is_ros_running = False
        while not is_ros_running:
            p = subprocess.run(["rosnode", "list"], capture_output=True)
            if p.returncode == 0:
                is_ros_running = True
            time.sleep(0.1) # wait for roscore to be running
        rospy.init_node("px4_gym_node")
        rospy.loginfo("Preparing for the env ...")
        rospy.wait_for_service("/reset_init_global_nwu_pose")

        self.rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            uav_state_msg = rospy.wait_for_message("/uav_state", String, timeout=5.0)
            if (uav_state_msg.data == "MISSION"):
                break
            self.rate.sleep()

        while not rospy.is_shutdown():
            planner_state_msg = rospy.wait_for_message("/planning/exec_state", Int8, timeout=5.0)
            if (planner_state_msg.data == 0 or planner_state_msg.data == 1):
                break
            self.rate.sleep()

        self._depth = None
        self._pos = None
        self._orient = None
        self._vel = None
        self._v_ref_pub = rospy.Publisher("/global_nwu_vel_ref", TwistStamped, queue_size=10)
        self._v_ref_msg = TwistStamped()
        self._pva_ref_pub = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        self._pva_cmd_msg = PositionCommand()

        rospy.loginfo("Env is ready!")

    def reset(self, seed=None):
        super().reset(seed=seed)

        while not rospy.is_shutdown():
            planner_state_msg = rospy.wait_for_message("/planning/exec_state", Int8, timeout=5.0)
            if (planner_state_msg.data == 0 or planner_state_msg.data == 1):
                break
            self.rate.sleep()

        req = setInitGlobalPoseRequest()

        new_pos = (2 * np.random.rand(2) - 1) * 10
        req.pose.position.x = new_pos[0]
        req.pose.position.y = new_pos[1]
        req.pose.position.z = self.takeoff_height

        # req.pose.position.x = -15
        # req.pose.position.y = 0
        # req.pose.position.z = self.takeoff_height

        while not rospy.is_shutdown():
            try:
                res = self.reset_init_pose_client(req)
                if res.success:
                    break
            except rospy.ServiceException as e:
                print(f"Service call failed: {e}")
            
            self.rate.sleep()

        return self._get_obs()

    def step(self, action):
        self._v_ref_msg.header.stamp = rospy.Time.now()
        self._v_ref_msg.twist.linear.x = action[0]
        self._v_ref_msg.twist.linear.y = action[1]
        self._v_ref_msg.twist.linear.z = action[2]
        self._v_ref_pub.publish(self._v_ref_msg)

        return self._get_obs(), *self._get_reward_and_done(), self._get_info()
    
    def policy(self, obs):
        raise NotImplementedError
    
    def _get_obs(self):
        depth_msg = rospy.wait_for_message("/pcl_render_node/cloud", PointCloud2, timeout=5)
        uav_odom_msg = rospy.wait_for_message("/global_nwu_odom", Odometry, timeout=5)
        self._depth = self._convert_to_pcl(depth_msg)
        self._pos = ros_numpy.numpify(uav_odom_msg.pose.pose.position)
        self._orient = ros_numpy.numpify(uav_odom_msg.pose.pose.orientation)
        self._vel = ros_numpy.numpify(uav_odom_msg.twist.twist.linear)
        return self._depth, self._pos, self._orient, self._vel

    def _get_reward_and_done(self):
        return np.random.randn(), bool(random.randint(0, 1))

    def _get_info(self):
        return {}

    def _kill(self):
        if self.ros_subproc:
            subprocess.run(["rosnode", "kill", "-a"])
            self.ros_subproc.kill()

    @staticmethod
    def _convert_to_pcl(pc2_msg):
        np_pc = ros_numpy.numpify(pc2_msg)
        points = np.zeros((np_pc.shape[0], 3))
        points[:, 0] = np_pc['x']
        points[:, 1] = np_pc['y']
        points[:, 2] = np_pc['z']
        return points

    def close(self):
        self._kill()