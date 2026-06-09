#!/usr/bin/env python3

import math
import random
import threading

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage, CameraInfo, Image
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

        self.lock = threading.RLock()

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

        # 60 mm = 0.06 m
        self.tag_size = float(rospy.get_param("~tag_size", 0.06))

        # Eğer camera_info gelirse bu değerler otomatik güncellenecek.
        self.image_width = int(rospy.get_param("~image_width", 640))
        self.image_height = int(rospy.get_param("~image_height", 480))
        self.horizontal_fov = float(rospy.get_param("~horizontal_fov", 2.79))

        # Sınırlar alana göre ayarlanacak
        self.x_min = float(rospy.get_param("~x_min", -0.05))
        self.x_max = float(rospy.get_param("~x_max", 0.95))
        self.y_min = float(rospy.get_param("~y_min", -0.05))
        self.y_max = float(rospy.get_param("~y_max", 0.95))

        self.motion_sigma_xy = float(rospy.get_param("~motion_sigma_xy", 0.015))
        self.motion_sigma_theta = float(rospy.get_param("~motion_sigma_theta", 0.03))

        self.sensor_sigma_dist = float(rospy.get_param("~sensor_sigma_dist", 0.35))
        self.sensor_sigma_angle = float(rospy.get_param("~sensor_sigma_angle", 0.18))

        # Odometry frame -> map frame offset.
        self.odom_map_x0 = float(rospy.get_param("~odom_map_x0", 0.85))
        self.odom_map_y0 = float(rospy.get_param("~odom_map_y0", 0.85))
        self.odom_map_theta0 = float(rospy.get_param("~odom_map_theta0", 0.0))

        # Alandaki tag'lere göre ayarlanacak
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
        self.odom_origin = None


        self.map1 = None
        self.map2 = None
        self.rectified_camera_matrix = None

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
        # Debug Publisher
        # -----------------------------
        self.debug_map_pub = rospy.Publisher(
            "/pf_debug_map/compressed",
            CompressedImage,
            queue_size=1
        )

        self.tag_debug_image_pub = rospy.Publisher(
            "/tag_debug_image/compressed",
            CompressedImage,
            queue_size=1
        )

        self.debug_view_pub = rospy.Publisher(
            "/pf_debug_view/image",
            Image,
            queue_size=1
        )

        self.tag_debug_raw_pub = rospy.Publisher(
            "/tag_debug_image/image",
            Image,
            queue_size=1
        )

        self.latest_camera_debug = None
        self.latest_map_debug = None

        self.save_debug_frames = rospy.get_param("~save_debug_frames", False)
        self.debug_frame_dir = rospy.get_param("~debug_frame_dir", "/tmp/pf_debug")
        self.debug_every_n_frames = int(rospy.get_param("~debug_every_n_frames", 10))
        self.debug_frame_count = 0

        self.publish_debug_topic = rospy.get_param("~publish_debug_topic", True)
        self.show_debug_window = rospy.get_param("~show_debug_window", False)
        self.debug_fps = float(rospy.get_param("~debug_fps", 5.0))

        # particle_style: "point" veya "arrow"
        self.particle_style = rospy.get_param("~particle_style", "arrow")
        self.max_particle_arrows = int(rospy.get_param("~max_particle_arrows", 80))

        self.max_path_points = int(rospy.get_param("~max_path_points", 150))

        # İlk kamera görüntüsü gelmeden boş ekran üretmek için
        self.latest_camera_debug = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        cv2.putText(
            self.latest_camera_debug,
            "Waiting for camera image...",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        rospy.on_shutdown(self.on_shutdown)

        if self.save_debug_frames:
            os.makedirs(self.debug_frame_dir, exist_ok=True)

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

        self.debug_view_timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.debug_fps)),
            self.debug_view_timer_callback
        )

        

        rospy.loginfo("Duckiebot particle filter started.")
        rospy.loginfo(f"Image topic: {self.image_topic}")
        rospy.loginfo(f"Camera info topic: {self.camera_info_topic}")
        rospy.loginfo(f"Pose topic: {self.pose_topic}")
        rospy.loginfo(f"Tag size: {self.tag_size} m")
        rospy.loginfo(f"Number of tags in map: {len(self.tag_map)}")

    def on_shutdown(self):
            rospy.loginfo("Shutting down Duckiebot Particle Filter...")
            if self.show_debug_window:
                cv2.destroyAllWindows()

    def init_particles(self):
        self.particles = []

        for _ in range(self.num_particles):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            theta = random.uniform(-math.pi, math.pi)
            weight = 1.0 / self.num_particles
            self.particles.append([x, y, theta, weight])

        self.publish_particles()
        self.publish_pf_estimate_and_path()
        self.publish_debug_map()

    def camera_info_callback(self, msg):
        if msg.K[0] <= 0.0:
            return

        with self.lock:
            self.camera_matrix = np.array(msg.K, dtype=np.float32).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D, dtype=np.float32).reshape(-1, 1)
            
            # Initialize rectification lookup maps once to protect embedded CPU overhead
            if self.map1 is None:
                w, h = msg.width, msg.height
                self.image_width = w
                self.image_height = h

                if getattr(msg, 'distortion_model', 'equidistant') == 'equidistant':
                    # Rectification logic tailored for Duckiebot fisheye lenses
                    self.rectified_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        self.camera_matrix, self.dist_coeffs, (w, h), np.eye(3), balance=0.0
                    )
                    self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                        self.camera_matrix, self.dist_coeffs, np.eye(3), 
                        self.rectified_camera_matrix, (w, h), cv2.CV_16SC2
                    )
                else:
                    # Fallback for standard pinhole cameras
                    self.rectified_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                        self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
                    )
                    self.map1, self.map2 = cv2.initUndistortRectifyMap(
                        self.camera_matrix, self.dist_coeffs, np.eye(3), 
                        self.rectified_camera_matrix, (w, h), cv2.CV_16SC2
                    )

    def pose_callback(self, msg):
        current_x = float(msg.x)
        current_y = float(msg.y)
        current_theta = float(msg.theta)

        if self.last_pose is None:
            self.last_pose = (current_x, current_y, current_theta)
            self.odom_origin = (current_x, current_y, current_theta)

            # Başlangıç odometry noktasını da haritada göster
            self.update_odom_path(current_x, current_y, current_theta)
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

        with self.lock:
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
        self.publish_pf_estimate_and_path()

    def image_callback(self, msg):
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logwarn("Compressed image decode failed.")
                return

            # ---- unwarp the raw fisheye frame into a linear projection space ----
            with self.lock:
                if self.map1 is not None and self.map2 is not None:
                    cv_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                    cam_matrix = self.rectified_camera_matrix
                else:
                    cam_matrix = self.camera_matrix

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if self.use_new_aruco_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.aruco_params
                )

            debug_image = cv_image.copy()

            if ids is None or len(corners) == 0:
                cv2.putText(
                    debug_image,
                    "No AprilTag detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                self.latest_camera_debug = debug_image
                self.publish_compressed_image(self.tag_debug_image_pub, debug_image)
                self.publish_raw_image(self.tag_debug_raw_pub, debug_image)
                self.publish_combined_debug_view()
                return

            cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)

            cv2.putText(
                debug_image,
                f"Detected tags: {len(corners)}, ids={ids.flatten().tolist()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            self.latest_camera_debug = debug_image
            self.publish_compressed_image(self.tag_debug_image_pub, debug_image)
            self.publish_raw_image(self.tag_debug_raw_pub, debug_image)
            self.publish_combined_debug_view()

            # Run pose estimation using the flattened matrix with ZERO distortion coefficients
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.tag_size,
                cam_matrix,
                np.zeros((5, 1), dtype=np.float32)  # Set to zero because the frame is already unwarped
            )

            measurements = []

            for i in range(len(corners)):
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
            if not measurements:
                return

            with self.lock:
                particles = np.array(self.particles)  # Shape: (N, 4)
                N = len(particles)

                particle_positions = particles[:, :2]  # Shape: (N, 2)
                particle_thetas = particles[:, 2]      # Shape: (N,)
                tags = np.array(self.tag_map)          # Shape: (8, 2)

                # Reshape for matrix broadcasting -> Shape: (N, 8)
                tags_x = tags[:, 0][np.newaxis, :]
                tags_y = tags[:, 1][np.newaxis, :]
                p_x = particle_positions[:, 0][:, np.newaxis]
                p_y = particle_positions[:, 1][:, np.newaxis]

                dx = tags_x - p_x
                dy = tags_y - p_y

                # Compute expected distances and bearings for all combinations
                expected_dists = np.sqrt(dx**2 + dy**2)
                raw_bearings = np.arctan2(dy, dx)
                expected_bearings = raw_bearings - particle_thetas[:, np.newaxis]
                expected_bearings = np.arctan2(np.sin(expected_bearings), np.cos(expected_bearings))

                prob_particles = np.ones(N)

                # Loop through active measurements, calculating tag probabilities in parallel
                for measured_dist, measured_bearing in measurements:
                    dist_error = measured_dist - expected_dists
                    bearing_error = measured_bearing - expected_bearings
                    bearing_error = np.arctan2(np.sin(bearing_error), np.cos(bearing_error))

                    prob_d = np.exp(-0.5 * (dist_error / self.sensor_sigma_dist) ** 2)
                    prob_a = np.exp(-0.5 * (bearing_error / self.sensor_sigma_angle) ** 2)

                    # Sum probabilities across all 8 tag hypotheses
                    prob_measurement = np.sum(prob_d * prob_a, axis=1)
                    prob_particles *= np.maximum(prob_measurement, 1e-300)

                sum_w = np.sum(prob_particles)

                if sum_w < 1e-300:
                    rospy.logwarn("All weights collapsed. Resetting weights.")
                    weights = [1.0 / N] * N
                    for i in range(N):
                        self.particles[i][3] = weights[i]
                else:
                    weights = prob_particles / sum_w
                    for i in range(N):
                        self.particles[i][3] = float(weights[i])

            self.publish_particle_markers()
            self.publish_pf_estimate_and_path()

            self.resample()
            self.publish_particles()


    def resample(self):

        with self.lock:
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

        with self.lock:
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

    def odom_to_map(self, odom_x, odom_y, odom_theta):
        if self.odom_origin is None:
            rel_x = 0.0
            rel_y = 0.0
            rel_theta = 0.0
        else:
            origin_x, origin_y, origin_theta = self.odom_origin

            dx = odom_x - origin_x
            dy = odom_y - origin_y
            dtheta = wrap_angle(odom_theta - origin_theta)

            # Duckiebot odom frame'inden başlangıç frame'ine göre local delta
            c0 = math.cos(origin_theta)
            s0 = math.sin(origin_theta)

            rel_x = dx * c0 + dy * s0
            rel_y = -dx * s0 + dy * c0
            rel_theta = dtheta

        # Başlangıçtaki gerçek harita pozu + odometry hareketi
        c = math.cos(self.odom_map_theta0)
        s = math.sin(self.odom_map_theta0)

        map_x = self.odom_map_x0 + c * rel_x - s * rel_y
        map_y = self.odom_map_y0 + s * rel_x + c * rel_y
        map_theta = wrap_angle(self.odom_map_theta0 + rel_theta)

        return map_x, map_y, map_theta

    def update_odom_path(self, x, y, theta):
        map_x, map_y, map_theta = self.odom_to_map(x, y, theta)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.world_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = float(map_x)
        pose_stamped.pose.position.y = float(map_y)
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = yaw_to_quaternion(map_theta)

        self.odom_path.header.stamp = rospy.Time.now()
        self.odom_path.poses.append(pose_stamped)

        if len(self.odom_path.poses) > 2000:
            self.odom_path.poses = self.odom_path.poses[-2000:]

        self.odom_path_pub.publish(self.odom_path)

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.world_frame
        pose_array.header.stamp = rospy.Time.now()

        with self.lock:

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

        with self.lock:
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


    def publish_compressed_image(self, publisher, image):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"

        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )

        if not success:
            return

        msg.data = encoded.tobytes()
        publisher.publish(msg)

    def publish_raw_image(self, publisher, image):
        image = np.ascontiguousarray(image)

        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = image.shape[1] * 3
        msg.data = image.tobytes()

        publisher.publish(msg)


    def world_to_pixel(self, x, y, img_w, img_h, margin, scale):
        px = int(margin + (x - self.x_min) * scale)
        py = int(img_h - margin - (y - self.y_min) * scale)
        return px, py


    def draw_path(self, image, path_msg, color, thickness, img_w, img_h, margin, scale):
        points = []

        for pose_stamped in path_msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            px, py = self.world_to_pixel(x, y, img_w, img_h, margin, scale)
            points.append((px, py))

        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(image, points[i - 1], points[i], color, thickness)


    def draw_robot_arrow(self, image, x, y, theta, color, img_w, img_h, margin, scale):
        px, py = self.world_to_pixel(x, y, img_w, img_h, margin, scale)

        length = 35
        end_x = int(px + length * math.cos(theta))
        end_y = int(py - length * math.sin(theta))

        cv2.arrowedLine(
            image,
            (px, py),
            (end_x, end_y),
            color,
            3,
            tipLength=0.35
        )


    def draw_recent_path(self, image, path_msg, color, thickness, img_w, img_h, margin, scale, max_points=250):
            points = []
            recent_poses = path_msg.poses[-max_points:]

            for pose_stamped in recent_poses:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                px, py = self.world_to_pixel(x, y, img_w, img_h, margin, scale)

                if 0 <= px < img_w and 0 <= py < img_h:
                    points.append((px, py))

            if len(points) >= 2:
                for i in range(1, len(points)):
                    # Added cv2.LINE_AA for smoother path rendering
                    cv2.line(image, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    def debug_view_timer_callback(self, event=None):
        self.publish_debug_map()

    def publish_debug_map(self):
            img_w = 620
            img_h = 520
            margin = 55

            # Use a darker, sleeker background to make colors pop
            image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30

            scale_x = (img_w - 2 * margin) / max(1e-6, (self.x_max - self.x_min))
            scale_y = (img_h - 2 * margin) / max(1e-6, (self.y_max - self.y_min))
            scale = min(scale_x, scale_y)

            # Draw the Room Boundary (Arena)
            x1, y1 = self.world_to_pixel(self.x_min, self.y_min, img_w, img_h, margin, scale)
            x2, y2 = self.world_to_pixel(self.x_max, self.y_max, img_w, img_h, margin, scale)
            cv2.rectangle(image, (x1, y2), (x2, y1), (80, 80, 80), 3, cv2.LINE_AA)

            # Draw AR Tags as distinct markers
            for i, (tag_x, tag_y) in enumerate(self.tag_map):
                px, py = self.world_to_pixel(tag_x, tag_y, img_w, img_h, margin, scale)
                size_px = 7
                
                # Bright cyan for tags so they stand out
                cv2.rectangle(image, (px - size_px, py - size_px), (px + size_px, py + size_px), (255, 255, 0), -1)
                cv2.rectangle(image, (px - size_px, py - size_px), (px + size_px, py + size_px), (255, 255, 255), 1)
                
                cv2.putText(image, f"T{i}", (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

            with self.lock:
                # Calculate max weight to normalize particle colors
                max_w = max(p[3] for p in self.particles) if self.particles else 1.0
                if max_w <= 0.0:
                    max_w = 1.0

                # Draw Particles (Color by weight, include orientation)
                for p in self.particles:
                    px, py = self.world_to_pixel(p[0], p[1], img_w, img_h, margin, scale)
                    if px < 0 or px >= img_w or py < 0 or py >= img_h:
                        continue

                    # Normalize score
                    score = max(0.0, min(1.0, p[3] / max_w))

                    # Color Gradient: Blue (Low Weight) -> Red (High Weight)
                    b = int(255 * (1.0 - score))
                    g = int(50 + 100 * score)
                    r = int(255 * score)
                    color = (b, g, r)

                    # Draw directional tail to represent 'theta'
                    tail_length = 6
                    end_x = int(px - tail_length * math.cos(p[2]))
                    end_y = int(py + tail_length * math.sin(p[2]))  # + because OpenCV y is inverted
                    
                    cv2.line(image, (px, py), (end_x, end_y), color, 1, cv2.LINE_AA)
                    cv2.circle(image, (px, py), 2, color, -1, cv2.LINE_AA)

            # Draw Trajectories
            self.draw_recent_path(image, self.odom_path, color=(150, 150, 150), thickness=2, 
                                img_w=img_w, img_h=img_h, margin=margin, scale=scale, max_points=250)
            self.draw_recent_path(image, self.pf_path, color=(0, 255, 0), thickness=2, 
                                img_w=img_w, img_h=img_h, margin=margin, scale=scale, max_points=250)

            # Draw Robot Estimates (Odometry vs PF)
            est_x, est_y, est_theta = self.compute_weighted_estimate()
            self.draw_robot_arrow(image, est_x, est_y, est_theta, color=(0, 255, 0), 
                                img_w=img_w, img_h=img_h, margin=margin, scale=scale) # PF Estimate is Green

            if self.last_pose is not None:
                odom_x, odom_y, odom_theta = self.last_pose
                map_odom_x, map_odom_y, map_odom_theta = self.odom_to_map(odom_x, odom_y, odom_theta)
                self.draw_robot_arrow(image, map_odom_x, map_odom_y, map_odom_theta, color=(150, 150, 150), 
                                    img_w=img_w, img_h=img_h, margin=margin, scale=scale) # Odom is Gray

            # UI Overlay / Legend
            cv2.putText(image, "Particle Filter Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Modern Legend Layout
            cv2.putText(image, "AR Tags", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "PF Estimate (Green)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Odometry (Gray)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(image, "Particles (Blue -> Red)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1, cv2.LINE_AA)

            cv2.putText(image, f"Particles: {len(self.particles)}", (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.publish_compressed_image(self.debug_map_pub, image)
            self.latest_map_debug = image
            self.publish_combined_debug_view()

            if self.save_debug_frames:
                if self.debug_frame_count % self.debug_every_n_frames == 0:
                    filename = os.path.join(self.debug_frame_dir, f"pf_map_{self.debug_frame_count:06d}.jpg")
                    cv2.imwrite(filename, image)
                self.debug_frame_count += 1

    def publish_combined_debug_view(self):
        if self.latest_camera_debug is None or self.latest_map_debug is None:
            return

        camera_image = self.latest_camera_debug.copy()
        map_image = self.latest_map_debug.copy()

        target_h = 420

        cam_scale = target_h / camera_image.shape[0]
        cam_w = int(camera_image.shape[1] * cam_scale)

        camera_resized = cv2.resize(camera_image, (cam_w, target_h))
        map_resized = cv2.resize(map_image, (520, target_h))

        combined = np.hstack((camera_resized, map_resized))

        cv2.putText(
            combined,
            "Camera AprilTag View",
            (20, target_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        self.publish_raw_image(self.debug_view_pub, combined)

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