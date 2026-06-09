#!/usr/bin/env python3

import math
import random

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

from duckietown_msgs.msg import Pose2DStamped

import os


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw):
    pose = Pose()
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(yaw * 0.5)
    pose.orientation.w = math.cos(yaw * 0.5)
    return pose.orientation


class DuckiebotParticleFilter:
    def __init__(self):
        rospy.init_node("duckiebot_particle_filter")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.robot_name = rospy.get_param("~robot_name", "bear")
        self.world_frame = rospy.get_param("~world_frame", "map")

        self.image_topic = rospy.get_param(
            "~image_topic",
            f"/{self.robot_name}/camera_node/image/compressed"
        )

        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            f"/{self.robot_name}/camera_node/camera_info"
        )

        self.pose_topic = rospy.get_param(
            "~pose_topic",
            f"/{self.robot_name}/velocity_to_pose_node/pose"
        )

        self.num_particles = int(rospy.get_param("~num_particles", 500))

        # 400 mm = 0.4 m
        self.tag_size = float(rospy.get_param("~tag_size", 0.4))

        # Eğer camera_info gelirse bu değerler otomatik güncellenecek.
        self.image_width = int(rospy.get_param("~image_width", 640))
        self.image_height = int(rospy.get_param("~image_height", 480))
        self.horizontal_fov = float(rospy.get_param("~horizontal_fov", 1.047))

        # Gerçek düzenekte bu sınırları kendi alanına göre ayarla.
        self.x_min = float(rospy.get_param("~x_min", -3.9))
        self.x_max = float(rospy.get_param("~x_max", 3.9))
        self.y_min = float(rospy.get_param("~y_min", -2.9))
        self.y_max = float(rospy.get_param("~y_max", 2.9))

        self.motion_sigma_xy = float(rospy.get_param("~motion_sigma_xy", 0.015))
        self.motion_sigma_theta = float(rospy.get_param("~motion_sigma_theta", 0.03))

        self.sensor_sigma_dist = float(rospy.get_param("~sensor_sigma_dist", 0.35))
        self.sensor_sigma_angle = float(rospy.get_param("~sensor_sigma_angle", 0.18))

        # GERÇEK DÜZENEKTE BUNLARI METRE CİNSİNDEN ÖLÇMEN GEREKİYOR.
        # Format: [x1, y1, x2, y2, ..., x8, y8]
        tag_map_flat = rospy.get_param(
            "~tag_map",
            [
                0.33, 0.00,
                0.50, 0.00,
                0.90, 0.25,
                0.90, 0.45,
                0.00, 0.42,
                0.00, 0.71,
                0.05, 0.90,
                0.55, 0.90,
            ],
        )

        self.tag_map = []
        for i in range(0, len(tag_map_flat), 2):
            self.tag_map.append((float(tag_map_flat[i]), float(tag_map_flat[i + 1])))

        # -----------------------------
        # Camera model
        # -----------------------------
        fx = (self.image_width / 2.0) / math.tan(self.horizontal_fov / 2.0)
        fy = fx
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0

        self.camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # -----------------------------
        # AprilTag detector
        # -----------------------------
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco bulunamadı. OpenCV contrib/aruco desteği yok.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
            self.use_new_aruco_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None
            self.use_new_aruco_api = False

        # -----------------------------
        # State
        # -----------------------------
        self.particles = []
        self.last_pose = None

        self.pf_path = Path()
        self.pf_path.header.frame_id = self.world_frame

        self.odom_path = Path()
        self.odom_path.header.frame_id = self.world_frame

        # -----------------------------
        # Publishers
        # -----------------------------
        self.particles_pub = rospy.Publisher(
            "/particles",
            PoseArray,
            queue_size=10
        )

        self.particle_markers_pub = rospy.Publisher(
            "/particle_markers",
            MarkerArray,
            queue_size=10
        )

        self.pf_estimate_pub = rospy.Publisher(
            "/pf_estimate",
            PoseStamped,
            queue_size=10
        )

        self.pf_path_pub = rospy.Publisher(
            "/pf_path",
            Path,
            queue_size=10
        )

        self.odom_path_pub = rospy.Publisher(
            "/odom_path",
            Path,
            queue_size=10
        )

        self.tag_markers_pub = rospy.Publisher(
            "/tag_markers",
            MarkerArray,
            queue_size=10
        )

        # -----------------------------
        # Subscribers
        # -----------------------------
        rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.pose_topic,
            Pose2DStamped,
            self.pose_callback,
            queue_size=10
        )

        self.init_particles()

        self.tag_marker_timer = rospy.Timer(
            rospy.Duration(1.0),
            self.publish_tag_markers
        )

        rospy.loginfo("Duckiebot particle filter started.")
        rospy.loginfo(f"Image topic: {self.image_topic}")
        rospy.loginfo(f"Camera info topic: {self.camera_info_topic}")
        rospy.loginfo(f"Pose topic: {self.pose_topic}")
        rospy.loginfo(f"Tag size: {self.tag_size} m")
        rospy.loginfo(f"Number of tags in map: {len(self.tag_map)}")

    def init_particles(self):
        self.particles = []

        for _ in range(self.num_particles):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            theta = random.uniform(-math.pi, math.pi)
            weight = 1.0 / self.num_particles
            self.particles.append([x, y, theta, weight])

        self.publish_particles()

    def camera_info_callback(self, msg):
        if msg.K[0] <= 0.0:
            return

        self.camera_matrix = np.array(msg.K, dtype=np.float32).reshape(3, 3)

        if len(msg.D) >= 5:
            self.dist_coeffs = np.array(msg.D[:5], dtype=np.float32).reshape(5, 1)

    def pose_callback(self, msg):
        current_x = float(msg.x)
        current_y = float(msg.y)
        current_theta = float(msg.theta)

        if self.last_pose is None:
            self.last_pose = (current_x, current_y, current_theta)
            return

        dx_world = current_x - self.last_pose[0]
        dy_world = current_y - self.last_pose[1]
        dtheta = wrap_angle(current_theta - self.last_pose[2])

        if abs(dx_world) < 1e-5 and abs(dy_world) < 1e-5 and abs(dtheta) < 1e-5:
            return

        cos_last = math.cos(self.last_pose[2])
        sin_last = math.sin(self.last_pose[2])

        dx_local = dx_world * cos_last + dy_world * sin_last
        dy_local = -dx_world * sin_last + dy_world * cos_last

        self.predict(dx_local, dy_local, dtheta)
        self.update_odom_path(current_x, current_y, current_theta)

        self.last_pose = (current_x, current_y, current_theta)

    def predict(self, dx_local, dy_local, dtheta):
        for p in self.particles:
            p_cos = math.cos(p[2])
            p_sin = math.sin(p[2])

            p[0] += (
                dx_local * p_cos
                - dy_local * p_sin
                + random.gauss(0.0, self.motion_sigma_xy)
            )

            p[1] += (
                dx_local * p_sin
                + dy_local * p_cos
                + random.gauss(0.0, self.motion_sigma_xy)
            )

            p[2] = wrap_angle(
                p[2] + dtheta + random.gauss(0.0, self.motion_sigma_theta)
            )

            p[0] = max(self.x_min, min(self.x_max, p[0]))
            p[1] = max(self.y_min, min(self.y_max, p[1]))

        self.publish_particles()

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            rospy.logwarn("Compressed image decode failed.")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if self.use_new_aruco_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is None or len(corners) == 0:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.tag_size,
            self.camera_matrix,
            self.dist_coeffs
        )

        measurements = []

        for i in range(len(corners)):
            # ID localization'da kullanılmıyor.
            # Çünkü ödevde tüm fiziksel tag'lerin aynı ID'ye sahip olması isteniyor.
            tvec = tvecs[i][0]

            x_cam = float(tvec[0])
            z_cam = float(tvec[2])

            distance = math.sqrt(x_cam ** 2 + z_cam ** 2)
            bearing = -math.atan2(x_cam, z_cam)

            measurements.append((distance, bearing))

        rospy.loginfo_throttle(
            1.0,
            f"Detected AprilTags: {len(measurements)}, ids={ids.flatten().tolist()}"
        )

        self.update_weights(measurements)

    def update_weights(self, measurements):
        weights = []

        for p in self.particles:
            prob_particle = 1.0

            for measured_dist, measured_bearing in measurements:
                prob_measurement = 0.0

                for tag_x, tag_y in self.tag_map:
                    expected_dx = tag_x - p[0]
                    expected_dy = tag_y - p[1]

                    expected_dist = math.sqrt(expected_dx ** 2 + expected_dy ** 2)

                    expected_bearing = wrap_angle(
                        math.atan2(expected_dy, expected_dx) - p[2]
                    )

                    dist_error = measured_dist - expected_dist
                    bearing_error = wrap_angle(measured_bearing - expected_bearing)

                    prob_d = math.exp(
                        -0.5 * (dist_error / self.sensor_sigma_dist) ** 2
                    )

                    prob_a = math.exp(
                        -0.5 * (bearing_error / self.sensor_sigma_angle) ** 2
                    )

                    prob_measurement += prob_d * prob_a

                prob_particle *= max(prob_measurement, 1e-300)

            weights.append(prob_particle)

        sum_w = sum(weights)

        if sum_w < 1e-300:
            rospy.logwarn("All weights collapsed. Resetting weights.")
            weights = [1.0 / self.num_particles] * self.num_particles
        else:
            weights = [w / sum_w for w in weights]

        for i in range(self.num_particles):
            self.particles[i][3] = weights[i]

        self.publish_particle_markers()
        self.publish_pf_estimate_and_path()

        self.resample()
        self.publish_particles()

    def resample(self):
        new_particles = []

        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        max_w = max(p[3] for p in self.particles)

        if max_w <= 0.0:
            self.init_particles()
            return

        for _ in range(self.num_particles):
            beta += random.uniform(0.0, 2.0 * max_w)

            while beta > self.particles[index][3]:
                beta -= self.particles[index][3]
                index = (index + 1) % self.num_particles

            p = self.particles[index]
            new_particles.append([p[0], p[1], p[2], 1.0 / self.num_particles])

        self.particles = new_particles

    def compute_weighted_estimate(self):
        sum_w = sum(p[3] for p in self.particles)

        if sum_w <= 0.0:
            weights = [1.0 / self.num_particles] * self.num_particles
        else:
            weights = [p[3] / sum_w for p in self.particles]

        x = sum(w * p[0] for w, p in zip(weights, self.particles))
        y = sum(w * p[1] for w, p in zip(weights, self.particles))

        sin_sum = sum(w * math.sin(p[2]) for w, p in zip(weights, self.particles))
        cos_sum = sum(w * math.cos(p[2]) for w, p in zip(weights, self.particles))

        theta = math.atan2(sin_sum, cos_sum)

        return x, y, theta

    def publish_pf_estimate_and_path(self):
        x, y, theta = self.compute_weighted_estimate()

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.world_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = float(x)
        pose_stamped.pose.position.y = float(y)
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = yaw_to_quaternion(theta)

        self.pf_estimate_pub.publish(pose_stamped)

        self.pf_path.header.stamp = rospy.Time.now()
        self.pf_path.poses.append(pose_stamped)

        if len(self.pf_path.poses) > 2000:
            self.pf_path.poses = self.pf_path.poses[-2000:]

        self.pf_path_pub.publish(self.pf_path)

    def update_odom_path(self, x, y, theta):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.world_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = float(x)
        pose_stamped.pose.position.y = float(y)
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = yaw_to_quaternion(theta)

        self.odom_path.header.stamp = rospy.Time.now()
        self.odom_path.poses.append(pose_stamped)

        if len(self.odom_path.poses) > 2000:
            self.odom_path.poses = self.odom_path.poses[-2000:]

        self.odom_path_pub.publish(self.odom_path)

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.world_frame
        pose_array.header.stamp = rospy.Time.now()

        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0
            pose.orientation = yaw_to_quaternion(p[2])
            pose_array.poses.append(pose)

        self.particles_pub.publish(pose_array)

    def publish_particle_markers(self):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        max_w = max(p[3] for p in self.particles)
        if max_w <= 0.0:
            max_w = 1.0

        now = rospy.Time.now()

        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = 0.03

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            score = max(0.0, min(1.0, p[3] / max_w))

            marker.color.r = float(score)
            marker.color.g = 0.1
            marker.color.b = float(1.0 - score)
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.particle_markers_pub.publish(marker_array)

    def publish_tag_markers(self, event=None):
        marker_array = MarkerArray()
        now = rospy.Time.now()

        for i, (x, y) in enumerate(self.tag_map):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "tags"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.05

            marker.scale.x = self.tag_size
            marker.scale.y = 0.03
            marker.scale.z = self.tag_size

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = self.world_frame
            text.header.stamp = now
            text.ns = "tag_labels"
            text.id = 100 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = 0.35
            text.scale.z = 0.18
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Tag {i}"

            marker_array.markers.append(text)

        self.tag_markers_pub.publish(marker_array)


if __name__ == "__main__":
    try:
        node = DuckiebotParticleFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass