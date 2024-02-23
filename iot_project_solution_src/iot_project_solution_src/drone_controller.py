import time
import random
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle

from geometry_msgs.msg import Point, Vector3, Twist
from nav_msgs.msg import Odometry
from iot_project_solution_interfaces.action import PatrollingAction
from iot_project_solution_interfaces.msg import DronePosition
from iot_project_solution_src.math_utils import *
import threading



DRONE_MIN_ALTITUDE_TO_PERFORM_MOVEMENT = 1


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")

        self.drone_id = -1

        self.position = Point(x=0.0, y=0.0, z=0.0)
        self.yaw = 0
        self.other_drones_positions = {}
        self.last_visits_lock = threading.Lock()
        
        self.cmd_vel_topic = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.odometry_topic = self.create_subscription(
            Odometry,
            'odometry',
            self.store_position_callback,
            10
        )

        self.patrol_action = ActionServer(
            self,
            PatrollingAction,
            'patrol_targets',
            self.patrol_action_callback
        )

        self.other_drones_positions_subscriber = self.create_subscription(
            DronePosition,
            '/drone_positions',
            self.other_drones_position_callback,
            10
        )
        
        self.position_publisher = self.create_publisher(DronePosition, '/drone_positions', 10)


    def store_position_callback(self, msg : Odometry):
        
        self.position = msg.pose.pose.position

        if(self.drone_id != -1):
            drone_position_msg = DronePosition()
            drone_position_msg.drone_id = self.drone_id
            drone_position_msg.position = self.position

            # Pubblica il messaggio DronePosition
            self.position_publisher.publish(drone_position_msg)

        self.yaw = get_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

    def get_lock_for_drone(self, drone_id):
        with self.global_lock:
            if drone_id not in self.locks:
                self.locks[drone_id] = threading.Lock()
            return self.locks[drone_id]
        

    def other_drones_position_callback(self, msg: DronePosition):
        if msg.drone_id != self.drone_id:
            self.other_drones_positions[msg.drone_id] = (msg.position.x, msg.position.y, msg.position.z)


    def patrol_action_callback(self, msg: ServerGoalHandle):


        command_goal: PatrollingAction.Goal = msg.request
        self.drone_id = command_goal.drone_id
        
        targets = command_goal.targets

        self.fly_to_altitude()

        count = 0
        for target in targets:
            count += 1
            # rotate to target
            self.rotate_to_target(target)
            # move to target
            self.move_to_target(msg, count, target)
            # send feedback for the target reached
            self.report_target_reached(msg, count, "reached", target)

        msg.succeed()

        result = PatrollingAction.Result()
        result.success = "Patrolling completed!"

        return result


    def fly_to_altitude(self, altitude = DRONE_MIN_ALTITUDE_TO_PERFORM_MOVEMENT):

        # Skip movement if desiderd altitude is already reached
        if (self.position.z >= altitude):
            return

        # Instantiate the move_up message
        move_up = Twist()
        move_up.linear = Vector3(x=0.0, y=0.0, z=1.0)
        move_up.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self.cmd_vel_topic.publish(move_up)

        # Loop until for the drone reaches the desired altitude
        # Note that in order for the drone to be perfectly aligned with the
        # requested height (not required for the exercise), you should keep on
        # listening to the current position and reduce the linear speed when 
        # you get close to the desired altitude

        while(self.position.z < altitude):
            self.cmd_vel_topic.publish(move_up)
            time.sleep(0.1)

        # Stop movement after the altitue has been reached
        stop_mov = Twist()
        stop_mov.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_mov.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_mov)


    def rotate_to_target(self, target : Point, eps = 0.1):

        target = (target.x, target.y, target.z)

        # We compute the angle between the current target position and the target
        # position here

        start_position = (self.position.x, self.position.y)
        target_angle = angle_between_points(start_position, target)
        angle_to_rotate = target_angle - self.yaw

        # We verify the optimal direction of the rotation here
        rotation_dir = -1
        if angle_to_rotate < 0 or angle_to_rotate > math.pi:
            rotation_dir = 1
        
        # Prepare the cmd_vel message
        move_msg = Twist()
        move_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        move_msg.angular = Vector3(x=0.0, y=0.0, z = 0.5 * rotation_dir)


        # Publish the message until the correct rotation is reached (accounting for some eps error)
        # Note that here the eps also helps us stop the drone and not overshoot the target, as
        # the drone will keep moving for a while after it receives a stop message
        # Also note that rotating the drone too fast will make it loose altitude.
        # You can account for that by also giving some z linear speed to the rotation movement.
        while abs(angle_to_rotate) > eps:
            angle_to_rotate = target_angle - self.yaw
            self.cmd_vel_topic.publish(move_msg)
            # No sleep here. We don't want to miss the angle by sleeping too much. Even 0.1 seconds
            # could make us miss the given epsilon interval

        # When done, send a stop message to be sure that the drone doesn't
        # overshoot its target
        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_msg)


    


    def move_to_target(self, msg, count, target, eps=0.5, angle_eps=0.05, speed_increment=8, progress_threshold=0.1, check_interval=2):
        
        current_position = (self.position.x, self.position.y, self.position.z)
        objective_point = (target.x, target.y, target.z)
        last_check_time = time.time()
        last_position = current_position
        speed = 1.0  # Velocità iniziale

        while point_distance(current_position, objective_point) > eps:
            collision_detected, approaching_direction, close_to_target = self.check_for_collisions(target)

            if collision_detected and close_to_target:
                return

            if collision_detected:
                if approaching_direction == 'lateral':
                    self.fly_to_altitude(self.position.z + 1)
                else:
                    self.move_laterally()

            current_position = (self.position.x, self.position.y, self.position.z)
            direction_vector = move_vector(current_position, objective_point)
            #distance_to_target = point_distance(current_position, objective_point)
            
            # Aggiorna la velocità basandosi sulla distanza
            #speed = max(0.1, min(1.0, distance_to_target / 5))

            # Se non c'è stato progresso, aumenta la velocità per compensare il vento
            if time.time() - last_check_time > check_interval:
                if point_distance(current_position, last_position) < progress_threshold:
                    speed = speed + speed_increment
                    
                last_check_time = time.time()
                last_position = current_position

            mov = Twist()
            mov.linear = Vector3(x=direction_vector[0] * speed, y=0.0, z=direction_vector[1] * speed)
            mov.angular = Vector3(x=0.0, y=0.0, z=0.0)

            angle = angle_between_points(current_position, objective_point)
            current_angle = self.yaw

            if not (angle - angle_eps < current_angle < angle + angle_eps):
                angle_diff = (current_angle - angle)
                mov.angular = Vector3(x=0.0, y=0.0, z=math.sin(angle_diff) * speed)

            self.cmd_vel_topic.publish(mov)

        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_topic.publish(stop_msg)


    def check_for_collisions(self, target: Point):
        safety_distance = 2.0  # Distanza di sicurezza
        approaching_direction = None  # 'above', 'below', 'lateral', o None
        close_to_target = False  # Indica se l'altro drone è vicino al target

        current_drone_distance_to_target = math.sqrt(
            (target.x - self.position.x)**2 + 
            (target.y - self.position.y)**2 + 
            (target.z - self.position.z)**2)

        for drone_id, position in self.other_drones_positions.items():
            if drone_id == self.drone_id:
                continue  # Ignora il drone stesso

            # Calcola la distanza tra il drone corrente e gli altri droni
            drone_distance = math.sqrt(
                (self.position.x - position[0])**2 + 
                (self.position.y - position[1])**2 + 
                (self.position.z - position[2])**2)

            # Calcola la distanza tra l'altro drone e il target
            distance_other_drone_to_target = math.sqrt(
                (target.x - position[0])**2 + 
                (target.y - position[1])**2 + 
                (target.z - position[2])**2)

            if drone_distance < safety_distance:
                if self.position.z < position[2]:
                    approaching_direction = 'below'
                elif self.position.z > position[2]:
                    approaching_direction = 'above'
                else:
                    approaching_direction = 'lateral'

                if current_drone_distance_to_target > distance_other_drone_to_target and distance_other_drone_to_target <= 2:
                    return (True, approaching_direction, True)
                elif self.drone_id < drone_id:
                    return (True, approaching_direction, False)

        return (False, None, close_to_target)


    

    def move_laterally(self, distance=1.0, duration=2.0):
        """
        Muove il drone lateralmente per una distanza o durata specificata e poi lo ferma.
        
        Args:
        distance (float): La distanza laterale da percorrere dal drone (non usata se si basa su durata).
        duration (float): La durata dello spostamento laterale in secondi.
        """
        # Calcola il vettore di spostamento laterale con direzione randomizzata
        direction = random.choice([-1, 1])  # -1 per sinistra, 1 per destra

        lateral_move = Twist()
        lateral_move.linear.x = 0.0  # Nessun movimento in avanti/indietro
        lateral_move.linear.y = direction * 1.0  # Movimento laterale randomizzato
        lateral_move.linear.z = 0.0  # Mantiene l'altitudine corrente
        lateral_move.angular.z = 0.0  # Nessuna rotazione

        # Pubblica il comando di movimento laterale
        self.cmd_vel_topic.publish(lateral_move)

        # Aspetta per la durata dello spostamento prima di fermare il drone
        time.sleep(duration)

        # Fermare il drone
        self.stop_drone()

    def stop_drone(self):
        """
        Ferma il movimento del drone inviando un comando di velocità zero.
        """
        stop_move = Twist()
        stop_move.linear.x = 0.0
        stop_move.linear.y = 0.0
        stop_move.linear.z = 0.0
        stop_move.angular.z = 0.0
        self.cmd_vel_topic.publish(stop_move)


    
    def report_target_reached(self, goal_handle, target_count, progress, target):
        feedback = PatrollingAction.Feedback()

        # Imposta la posizione nel feedback
        #feedback.position = self.position questo è se vogliamo pubblicare la posizione per ogni movimento. Ovviamente dobbiamo aggiungere self.report_target_reached(msg, count, "reaching") alla fine di while a move_to_target

        feedback.position = target
        
        # Imposta il progresso nel feedback
        feedback.progress = f"Target %d {progress}" % target_count
        
        # Invia il feedback
        goal_handle.publish_feedback(feedback)

    

def main():
    rclpy.init()

    executor = MultiThreadedExecutor()
    drone_controller = DroneController()

    executor.add_node(drone_controller)
    executor.spin()

    executor.shutdown()
    drone_controller.destroy_node()

    rclpy.shutdown()