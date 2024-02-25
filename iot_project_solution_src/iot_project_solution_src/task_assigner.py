import time
import random

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from threading import Lock
from rosgraph_msgs.msg import Clock
from iot_project_interfaces.srv import TaskAssignment
from iot_project_solution_interfaces.action import PatrollingAction
import numpy as np
from sklearn.cluster import KMeans
from nav_msgs.msg import Odometry
from iot_project_solution_src.math_utils import *
from geometry_msgs.msg import Point
from iot_project_solution_interfaces.msg import DronePosition


class TaskAssigner(Node):

    def __init__(self):

        super().__init__('task_assigner')
            
        self.task = None
        self.no_drones = 0
        self.targets = []
        self.thresholds = []
        self.last_visits = []
        self.action_servers = []
        self.idle = []
        self.targets_visited_count = 0
        self.all_visited = True


        self.last_visits_lock = Lock()

        self.locks = []

        self.sim_time = 0

        self.drone_positions = {}
        self.cluster_assignments = {}
        self.target_status = []


        self.task_announcer = self.create_client(
            TaskAssignment,
            '/task_assigner/get_task'
        )

        self.sim_time_topic = self.create_subscription(
            Clock,
            '/world/iot_project_world/clock',
            self.store_sim_time_callback,
            10
        )

        self.drone_position_subscription = self.create_subscription(
            DronePosition,
            '/drone_positions',
            self.update_drone_position_callback,
            10
        )


    def update_drone_position_callback(self, msg: DronePosition):
        with self.last_visits_lock:
            self.drone_positions[msg.drone_id] = msg.position


        
    # Function used to wait for the current task. After receiving the task, it submits
    # to all the drone topics
    def get_task_and_subscribe_to_drones(self):

        self.get_logger().info("Task assigner has started. Waiting for task info")

        while not self.task_announcer.wait_for_service(timeout_sec = 1.0):
            time.sleep(0.5)

        self.get_logger().info("Task assigner is online. Requesting patrolling task")

        assignment_future = self.task_announcer.call_async(TaskAssignment.Request())
        assignment_future.add_done_callback(self.first_assignment_callback)



    def first_assignment_callback(self, assignment_future):

        task : TaskAssignment.Response = assignment_future.result()

        self.task = task
        self.no_drones = task.no_drones
        self.targets = task.target_positions
        self.thresholds = task.target_thresholds
        self.last_visits = task.last_visits

        self.current_tasks = [None]*self.no_drones
        self.idle = [True] * self.no_drones
        self.target_status = [False] * len(self.targets)
        self.locks = [Lock() for _ in range(self.no_drones)]

        # Now create a client for the action server of each drone
        for d in range(self.no_drones):
            self.action_servers.append(
                ActionClient(
                    self,
                    PatrollingAction,
                    'X3_%d/patrol_targets' % d,
                )
            )

        self.start_status_updater()


    # This method starts on a separate thread an ever-going patrolling task, it does that
    # by checking the idle state value of every drone and submitting a new goal as soon as
    # that value goes back to True
    def keep_patrolling(self):
        def keep_patrolling_inner():
            while True:
                if self.no_drones > 0:
                    self.calculate_clusters()
                    for d in range(self.no_drones):
                        with self.locks[d]:  
                            if self.idle[d]:
                                Thread(target=self.submit_task, args=(d,)).start()
                else:
                    print("Nessun drone disponibile per il pattugliamento.")
                time.sleep(0.1)
        Thread(target=keep_patrolling_inner).start()


    def calculate_clusters(self):
        with self.last_visits_lock:
            
            if(self.all_visited):
                points = np.array([[t.x, t.y, t.z] for t in self.targets])
                num_clusters = min(len(points), self.no_drones)

                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(points)

                self.cluster_assignments = {i: [] for i in range(self.no_drones)}

                for i, label in enumerate(kmeans.labels_):
                    self.cluster_assignments[label].append(self.targets[i])
                
                self.all_visited = False

        '''

        with self.last_visits_lock:
            unvisited_indices_and_targets = [i for i, _ in enumerate(self.targets) if not self.target_status[i]]
        
        if(unvisited_indices_and_targets):
            for d in range(self.no_drones):
                if(self.idle[d]):
                    oldest_visit = max([(i, self.last_visits[i]['last_visit']) for i in unvisited_indices_and_targets], key=lambda x: x[1])
                    oldest_visited_index = oldest_visit[0]
                    self.cluster_assignments[d] = [self.targets[oldest_visited_index]]
                    unvisited_indices_and_targets.remove(oldest_visited_index)

        '''
            
    # Callback used to verify if the action has been accepted.
    # If it did, prepares a callback for when the action gets completed
    def patrol_submitted_callback(self, future, drone_id):

        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info("Task has been refused by the action server")
            return
        
        result_future = goal_handle.get_result_async()

        # Lambda function as a callback, check the function before if you don't know what you are looking at
        result_future.add_done_callback(
            lambda future, d=drone_id: self.patrol_completed_callback(future, d)
        )


    # Callback used to update the idle state of the drone when the action ends
    def patrol_completed_callback(self, future, drone_id):
        #self.get_logger().info("Patrolling action for drone X3_%s has been completed. Drone is going idle" % drone_id)
        self.idle[drone_id] = True

    # Callback used to store simulation time
    def store_sim_time_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec / 1e9 

    
    def submit_task(self, drone_id):
        
        with self.locks[drone_id]:
            if not self.idle[drone_id]:
                return  

            self.idle[drone_id] = False

        if not self.action_servers[drone_id].wait_for_server(timeout_sec=1.0):
            with self.locks[drone_id]:  
                self.idle[drone_id] = True 
            return

        targets_to_patrol = self.cluster_assignments[drone_id]


        if self.drone_positions:
            # Calcola i pesi per ogni target basati sulla distanza e sul tempo trascorso dall'ultima visita
            weighted_targets = self.calculate_weighted_targets(targets_to_patrol, drone_id)
            # Ottimizza il percorso utilizzando TSP modificato per tenere conto dei pesi
            optimized_path = self.solve_tsp_with_weights(weighted_targets)
        else:
            optimized_path = targets_to_patrol

        patrol_task = PatrollingAction.Goal()
        patrol_task.targets = optimized_path  # Usa il percorso ottimizzato invece dell'assegnazione diretta
        patrol_task.drone_id = drone_id
        patrol_future = self.action_servers[drone_id].send_goal_async(patrol_task, feedback_callback=lambda feedback, d=drone_id: self.handle_feedback(feedback, d))
        patrol_future.add_done_callback(lambda future, d=drone_id: self.patrol_submitted_callback(future, d))


    def calculate_weighted_targets(self, targets, drone_id):
    # Calcola il tempo corrente della simulazione per determinare il tempo trascorso
        current_time = self.sim_time
        
        weighted_targets = []
        drone_position = self.drone_positions.get(drone_id, None)
        
        if drone_position is None:
            return targets
        
        for target in targets:
            # Calcola la distanza dal drone al target
            distance = point_distance((drone_position.x, drone_position.y, drone_position.z),
                                    (target.x, target.y, target.z))

            # Trova il tempo dell'ultima visita per il target corrente
            last_visit_time = next((item['last_visit'] for item in self.last_visits if item['position'] == target), None)
            if last_visit_time is None:
                continue  # Se non trovi il target, continua con il prossimo
            
            time_since_last_visit = current_time - last_visit_time / 1e9
            
            # Calcola il peso (qui uso una semplice somma)
            weight = distance + time_since_last_visit
            
            weighted_targets.append((target, weight))

        return weighted_targets


    def solve_tsp_with_weights(self, weighted_targets):
        
        path = []
        

        if not weighted_targets:
            return path

        # Ordina i target basandosi sui pesi pre-calcolati (dal più basso al più alto)
        sorted_targets = sorted(weighted_targets, key=lambda x: x[1], reverse=True)

        for target, _ in sorted_targets:
            path.append(target)

        return path


    def handle_feedback(self, feedback_msg, drone_id):
        current_position = feedback_msg.feedback.position
        
        for i, target in enumerate(self.targets):
            if (target.x == current_position.x and
                target.y == current_position.y and
                target.z == current_position.z):
                with self.last_visits_lock:  
                    if not self.target_status[i]:  
                        self.target_status[i] = True
                        self.targets_visited_count += 1 
                        if self.targets_visited_count >= len(self.target_status)/2+1:
                            self.target_status = [False] * len(self.target_status)
                            self.targets_visited_count = 0  
                            self.all_visited = True
                    break
        
        
    def get_updated_target_status(self):
        future = self.task_announcer.call_async(TaskAssignment.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        with self.last_visits_lock:
            self.last_visits = [{'position': pos, 'last_visit': lv} for pos, lv in zip(response.target_positions, response.last_visits)]


    def start_status_updater(self):
        def updater():
            while True:
                self.get_updated_target_status()
                time.sleep(0.5) 
        Thread(target=updater).start()

    

def main():

    time.sleep(3.0)
    
    rclpy.init()

    task_assigner = TaskAssigner()
    executor = MultiThreadedExecutor()
    executor.add_node(task_assigner)

    task_assigner.get_task_and_subscribe_to_drones()
    task_assigner.keep_patrolling()

    executor.spin()

    executor.shutdown()
    task_assigner.destroy_node()

    rclpy.shutdown()
