from asyncio import futures
from cmath import nan
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sycabot_utils.utilities import quat2eul

import math

import time
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial import ConvexHull
import numpy.matlib as matlib
from numpy.linalg import matrix_rank
from shapely.geometry import box, MultiPolygon, Polygon, LineString
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *

# import numpy as np
# from numpy.linalg import norm
# from scipy.optimize import linear_sum_assignment, minimize
# import math as m
# import time
# import sys

from std_srvs.srv import Trigger
from sycabot_interfaces.srv import BeaconSrv, Task
from geometry_msgs.msg import PoseStamped

from sycabot_interfaces.msg import Pose2D

class MIP(Node):
    # MAX_LIN_VEL = 0.3
    def __init__(self):
        super().__init__("MIP_pathplanner")

        self.initialised = False
        self.jb_positions = None
        self.OptiTrack_sub = []

        self.destinations = None

        cb_group = ReentrantCallbackGroup()
        # Define get ids service client
        self.get_ids_cli = self.create_client(BeaconSrv, 'get_list_ids')
        while not self.get_ids_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get ids service not available, waiting again...\n')
        
        # Define get ids service client
        self.refresh_ids_cli = self.create_client(Trigger, 'refresh_list_ids')
        while not self.refresh_ids_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Refresh ids service not available, waiting again...\n')
        
        # Create service for task request
        self.task_srv = self.create_service(Task, 'PRM_task_srv', self.set_task_cb, callback_group=cb_group)

        self.get_ids()
        self.initialise_pose_acquisition()
        self.wait4pose()

        
        self.map_x_min = -1.6
        self.map_x_max = 1.6
        self.map_y_min = -3.5
        self.map_y_max = 3.5
        self.robot_radius = 0.2      

        self.plan_path() 

    def initialise_destinations(self):
        # self.N_tasks = 8
        # self.destinations_x = np.zeros(self.N_tasks)
        # self.destinations_y = np.zeros(self.N_tasks)
        # self.destinations_x[0] = 0.5
        # self.destinations_y[0] = 3.1
        # self.destinations_x[1] = 1.6
        # self.destinations_y[1] = 0.0
        # self.destinations_x[2] = -0.5
        # self.destinations_y[2] = -3.1
        # self.destinations_x[3] = -1.6
        # self.destinations_y[3] = -0.1
        # self.destinations_x[4] = -1.6
        # self.destinations_y[4] = 1.2
        
        # self.destinations_x[5] = -1.239
        # self.destinations_y[5] = 2.377
        # self.destinations_x[6] = 0.582
        # self.destinations_y[6] = -1.798
        # self.destinations_x[7] = -1.153
        # self.destinations_y[7] = -1.260
        N = len(self.ids)
        destinations = np.zeros((N,2))
        for rob in range(N):
            print('Next goal for robot',self.ids[rob],'?')
            gi = int(input())
            # gi = np.random.randint(5)
            if gi == 1:
                destinations[rob,0] = 1.3
                destinations[rob,1] = 0
            if gi == 2:
                destinations[rob,0] = 1.6
                destinations[rob,1] = 0.0
            if gi == 3:
                destinations[rob,0] = -0.5
                destinations[rob,1] = -3.1
            if gi == 4:
                destinations[rob,0] = -1.6
                destinations[rob,1] = -0.1
            if gi == 5:
                destinations[rob,0] = -1.6
                destinations[rob,1] = 1.2

            if gi == 11:
                destinations[rob,0] = -1.239
                destinations[rob,1] = 2.377
            if gi == 12:
                destinations[rob,0] = 0.582
                destinations[rob,1] = -1.798
            if gi == 13:
                destinations[rob,0] = -1.153
                destinations[rob,1] = -1.260

            if gi == 103:
                destinations[rob,0] = -1.1
                destinations[rob,1] = 3.25
            if gi == 104:
                destinations[rob,0] = -0.65
                destinations[rob,1] = 3.25
            if gi == 105:
                destinations[rob,0] = -0.2
                destinations[rob,1] = 3.25

            

        #     if gi == 0:
        #         destinations[rob,0] = nan
        #         destinations[rob,1] = nan
        #     # found_destination = False
            # while not found_destination:
            #     gx = np.random.rand() * 3.0 - 1.5
            #     gy = np.random.rand() * 7.0 - 3.5

            #     dist, index = self.obstacle_kd_tree.query([gx, gy], k=1)
            #     if dist >= self.robot_radius:
            #         destinations[rob,0] = gx
            #         destinations[rob,1] = gy
            #         found_destination = True
        print('The destinationts are', destinations)
        self.destinations = destinations

        return 
        


    def set_task_cb(self, request, response):
        '''
        Callback for the tasks service. Sends the goal.

        arguments :
            request (interfaces.srv/Start.Response) =
                id (int64) = identifier of the jetbot [1,2,3,...]
        ------------------------------------------------
        return :
            response (interfaces.srv/Task.Response) = 
                task (geometry_msgs.msg/Pose2D[]) = pose of the assigned task
        '''
        idx = np.where(self.ids==request.id)[0][0]
        
        # tasks = []
        # tfs = [] # Problem here should send angles and times
        # for p in range(int(self.path_length[idx])):
        #     task_p = Pose2D()
        #     task_p.x = self.x_path[idx,p]
        #     task_p.y = self.y_path[idx,p]
        #     task_p.theta = 0.
        #     tasks.append(task_p)

        # tasks_augmented, times = self.add_time_to_wayposes([],[],tasks,0.3,3,'ignore_corners')
        trajectory = self.trajectory_list[idx]
        times = self.time_list[idx]

        tasks = []
        for i in range(self.trajectory_list[idx].shape[1]):
            pose = Pose2D()
            pose.x = trajectory[0][i]
            pose.y = trajectory[1][i]
            pose.theta = trajectory[2][i]
            tasks.append(pose)
        response.tasks = tasks
        response.tfs = times.tolist()
        return response

    def get_ids(self):
        '''
        Get the IDs from the ebacon node. Call and refresh until the beacon give a success.

        arguments :
        ------------------------------------------------
        return :
        '''
        get_ids_req = BeaconSrv.Request()
        self.future = self.get_ids_cli.call_async(get_ids_req)
        rclpy.spin_until_future_complete(self, self.future)
        while not self.future.result().success :
            self.future = self.refresh_ids()
            get_ids_req = BeaconSrv.Request()
            self.future = self.get_ids_cli.call_async(get_ids_req)
            rclpy.spin_until_future_complete(self, self.future)
        self.ids = np.array(self.future.result().ids)
        return

    def refresh_ids(self):
        '''
        Refresh the IDs of the beacon node.

        arguments :
        ------------------------------------------------
        return :
        '''
        refresh_ids_req = Trigger.Request()
        self.future = self.get_ids_cli.call_async(refresh_ids_req)
        rclpy.spin_until_future_complete(self, self.future)
        return

    def plan_path(self):
        N = len(self.ids)
        rounds = int(input('How many rounds?'))
        # rounds = 1
        self.x_path = np.empty((N,rounds * 60))
        self.x_path[:] = np.nan
        self.y_path = np.empty((N,rounds * 60))
        self.y_path[:] = np.nan
        self.path_cost = np.zeros(N)
        self.path_length = np.zeros(N,dtype=int)

        self.trajectory_list = [ [] for _ in range(N) ]
        self.time_list = [ [] for _ in range(N) ]
        for it in range(rounds):
            if it > 0:
                for rob in range(N):
                    self.jb_positions[rob,0] = self.destinations[rob,0]
                    self.jb_positions[rob,1] = self.destinations[rob,1]
            self.initialise_destinations()
     

            for rob in range(N):
                if it == 0:
                    start_x = self.jb_positions[rob,0]
                    start_y = self.jb_positions[rob,1]
                else:
                    start_x = self.x_path[rob,self.path_length[rob] - 1]
                    start_y = self.y_path[rob,self.path_length[rob] - 1]
                goal_x = self.destinations[rob,0]
                goal_y = self.destinations[rob,1]
                rx = []
                ry = []
                print('start_x',start_x,'start_y',start_y)

                # TO REPLACE
                obstacles, A_list, b_list, ax = self.map_def()
                x_traj, y_traj = self.solve_mip(ax, A_list, b_list, start_x, start_y, goal_x, goal_y)
                print(len(x_traj))
                print(y_traj)
                rx = x_traj[::-1]
                ry = y_traj[::-1]
                #rx, ry, dist = self.dijkstra_planning(start_x, start_y, goal_x, goal_y, rob, N, rob, N) 
                self.x_path[rob,self.path_length[rob]:self.path_length[rob] + len(x_traj)] = rx[-1::-1]
                self.y_path[rob,self.path_length[rob]:self.path_length[rob] + len(x_traj)] = ry[-1::-1]
                self.path_length[rob] = int(self.path_length[rob] + len(rx))
                # self.path_cost[rob] = self.path_cost[rob] + dist

                path = []
                for p in range(int(len(rx))):
                    pose_p = Pose2D()
                    pose_p.x = rx[-1 - p]
                    pose_p.y = ry[-1 - p]
                    pose_p.theta = 0.
                    path.append(pose_p)
                
                speed = 0.5
                spins = 3
                if it == rounds - 1:
                    spins = 0
                if it == 0:
                    self.trajectory_list[rob], self.time_list[rob] = self.add_time_to_wayposes([],[],path,speed,spins,'ignore_corners')
                else:
                    self.trajectory_list[rob], self.time_list[rob] = self.add_time_to_wayposes(self.trajectory_list[rob],self.time_list[rob],path,speed,spins,'ignore_corners')

                print('SycaBot_W',self.ids[rob])
                print('x path',self.x_path[rob,:self.path_length[rob]])
                print('y path',self.y_path[rob,:self.path_length[rob]])
                print('Path cost',self.path_cost[rob])
        return


    def add_time_to_wayposes(self, current_poses: list, current_waypose_times: list, poses: list, desired_speed: float,end_rounds, mode = 'ignore_corners'):
        W_poses = len(poses)
        W_current = len(current_waypose_times)
        if W_current == 0:
            current_poses = np.zeros((3,1))
            current_poses[0,0] = poses[0].x
            current_poses[1,0] = poses[0].y
            if W_poses > 1:
                current_poses[2,0] = np.arctan2(poses[1].y - poses[0].y, poses[1].x - poses[0].x)
            else:
                current_poses[2,0] = 0
            current_waypose_times = np.zeros(1)
            W_current = 1 

        if mode == 'ignore_corners':
            new_poses = np.zeros((3,W_poses))
            new_times = np.zeros(W_poses)
            for i in range(W_poses):
                new_poses[0,i] = poses[i].x
                new_poses[1,i] = poses[i].y
                if i > 0:
                    new_poses[2,i] = np.arctan2(poses[i].y - poses[i - 1].y, poses[i].x - poses[i - 1].x)
                    new_times[i] = new_times[i - 1] + 1 / desired_speed * np.sqrt((poses[i].y - poses[i - 1].y) ** 2 + (poses[i].x - poses[i - 1].x) ** 2)
                else:
                    if W_poses > 1:
                        new_poses[2,i] = np.arctan2(poses[i + 1].y - poses[i].y, poses[i + 1].x - poses[i].x)
                    else:
                        new_poses[2,i] = 0
                    new_times[i] = current_waypose_times[-1] + 2 * 0.12 / (2 * desired_speed) * math.pi
        if mode == 'stop_in_corners':
            new_poses = np.zeros((3,2 * W_poses))
            new_times = np.zeros(2 * W_poses)
            for i in range(W_poses):
                new_poses[0,i * 2] = poses[i].x
                new_poses[1,i * 2] = poses[i].y
                if i > 0:
                    new_poses[2,i  * 2] = np.arctan2(poses[i].y - poses[i - 1].y, poses[i].x - poses[i - 1].x)
                    new_times[i * 2] = new_times[i * 2 - 1] + 1 / desired_speed * np.sqrt((poses[i].y - poses[i - 1].y) ** 2 + (poses[i].x - poses[i - 1].x) ** 2)
                else:
                    new_poses[2,0] = poses[0].theta
                    new_times[0] = 1 + current_waypose_times[-1]
                new_poses[0,i  * 2 + 1] = poses[i].x
                new_poses[1,i * 2 + 1] = poses[i].y
                if i < W_poses - 1:
                    new_poses[2,i  * 2 + 1] = np.arctan2(poses[i + 1].y - poses[i].y, poses[i + 1].x - poses[i].x)
                    new_times[i  * 2 + 1] = new_times[i * 2] + 2 * 0.11 / (2 * desired_speed) * np.absolute(np.arctan2(np.sin(new_poses[2,i  * 2 + 1] -new_poses[2,i  * 2 ]),np.cos(new_poses[2,i  * 2 + 1] -new_poses[2,i  * 2 ])))
                else:
                    new_poses[2,i  * 2 + 1] = new_poses[2,i  * 2]
                    new_times[i  * 2 + 1] = new_times[i  * 2] + 2.     

        W_new = len(new_times)
        combined_poses = np.zeros((3,W_current + W_new + 4 * end_rounds))
        combined_times = np.zeros(W_current + W_new + 4 * end_rounds)
        combined_poses[:,:W_current] = current_poses
        combined_times[:W_current] = current_waypose_times

        combined_poses[:,W_current:W_current + W_new] = new_poses
        # if W_current > 0:
        #     combined_poses[2,W_current] = np.arctan2(new_poses[1,0] - current_poses[1,-1], new_poses[0,0] - current_poses[0,-1])
        combined_times[W_current:W_current + W_new] = new_times

        dir = np.sign(np.random.randn(1))
        for ts in range(end_rounds * 4):
            combined_poses[0,W_current + W_new + ts] = new_poses[0,-1]
            combined_poses[1,W_current + W_new + ts] = new_poses[1,-1]
            combined_poses[2,W_current + W_new + ts] = np.remainder(combined_poses[2,W_current + W_new + ts - 1] + dir * math.pi / 2 + math.pi,2 * math.pi) - math.pi
            combined_times[W_current + W_new + ts] = combined_times[W_current + W_new + ts - 1] + 2 * 0.12 * math.pi / (4 * desired_speed) 
        
        return combined_poses, combined_times

    def pol2ver(self, polygon):
        """
        pol2ver function gets the vertices from a polygon (shapely object).
        """
        xo, yo = polygon.exterior.xy
        vertices = np.array(list(set(list(zip(xo,yo)))))
        return vertices

    def ver2con(self, V):
        """ 
        ver2con function maps vertices of a convex polygon
        to linear constraints stored in matrices A and b. 
        """

        k = ConvexHull(V).vertices 
        u = np.roll(k,-1)

        k = np.vstack((k,u)).T
        c = np.mean(V[k[:,0]], axis = 0)

        V = V - matlib.repmat(c, V.shape[0],1)

        A = np.zeros((k.shape[0], V.shape[1]))
        A[:] = np.nan
        rc = 0

        for ix in range(k.shape[0]):
            F = V[k[ix]]
            if matrix_rank(F, 1e-5) == F.shape[0]:
                rc = rc+1
                A[rc-1,:] = np.linalg.solve(F,np.ones((F.shape[0])))

        A = A[0:rc,:]
        b = np.ones((A.shape[0], 1))
        b = b.T + A @ c

        return(A, b)

    # Defining the map and obstacles

    def map_def(self):

        map = Polygon(box(self.map_x_min, self.map_y_min, self.map_x_max, self.map_y_max))

        obstacles = []
        obstacles.append(LineString([(-1.498, 2.998), (0.001, 3.000)]))
        obstacles.append(LineString([(1.051, 3.001), (1.494, 3.000)]))
        obstacles.append(LineString([(1.494, 3.000), (1.493, 0.430)]))
        obstacles.append(LineString([(1.494, -0.374), (1.497, -2.998)]))
        obstacles.append(LineString([(0.002, -2.999), (1.497, -2.998)]))
        obstacles.append(LineString([(-1.498, -2.999), (-1.050, -2.999)]))
        obstacles.append(LineString([(-1.496,-0.500), (-1.498, -2.999)]))
        obstacles.append(LineString([(-1.496, 0.750), (-1.495, 0.299)]))
        obstacles.append(LineString([(-1.498, 2.998), (-1.498, 1.553)]))
        obstacles.append(LineString([(-0.481, 2.382), (0.879, 1.356)]))
        obstacles.append(LineString([(-1.498, 1.553), (-0.700, 1.551)]))
        obstacles.append(LineString([(1.018, 0.429), (1.493, 0.430)]))
        obstacles.append(LineString([(0.141, 1.040), (-0.269, 0.524)]))
        obstacles.append(LineString([(-1.496, 0.526), (-0.269, 0.524)]))
        obstacles.append(LineString([(-0.269, 0.524), (-0.261, -0.008)]))
        obstacles.append(LineString([(-0.261,-0.008), (0.480, -0.008)]))
        obstacles.append(LineString([(0.011, -0.008), (0.011, -0.486)]))
        obstacles.append(LineString([(-1.496, -0.859), (-0.492, -0.860)]))
        obstacles.append(LineString([(0.922, -0.613), (0.924, -2.093)]))
        obstacles.append(LineString([(0.260, -1.084), (0.260, -2.093)]))
        obstacles.append(LineString([(-0.665, -2.094), (0.924, -2.093)]))
        obstacles.append(LineString([(-0.685, -2.103), (-0.931, -2.414)]))

        fig, ax = plt.subplots()
        plt.xlim(self.map_x_min -0.1, self.map_x_max+ 0.1)
        plt.ylim(self.map_y_min -0.1, self.map_y_max + 0.1)

        dilated_obstacles = []
        area_obstacles = MultiPolygon()
        A_list = [] # List of matrices A for A * x <= b for each obstacle
        b_list = []

        for line in obstacles:
            dilated = line.buffer(0.025 + self.robot_radius, cap_style = 3,join_style = 2)
            
            dilated = dilated.minimum_rotated_rectangle

            dilated_obstacles.append(dilated)
            area_obstacles = area_obstacles.union(dilated)
            # ax.plot(line.xy[0], line.xy[1])
            x,y = dilated.exterior.xy
            ax.plot(x,y, 'b')

            vertices = self.pol2ver(dilated)
            A, b = self.ver2con(vertices)

            A_list.append(-A) #minus to get the constraint to be outside the polygon
            b_list.append(-b)


        xmap, ymap = map.exterior.xy
        ax.plot(xmap, ymap)

        return(obstacles, A_list, b_list, ax)

    def solve_mip(self, ax, A_list, b_list, start_x, start_y, goal_x, goal_y):
        # Create a new model
        m = gp.Model("mip1")    

        # Time step
        tau = 0.5 # [s]

        # Steps
        T = 30

        # Constraints on command
        u_max = 0.5  # [m/s]
        u_min = -0.5  # [m/s]

        # big M
        M = 100

        # Create variables
        x = m.addVars(T, lb=self.map_x_min, ub=self.map_x_max, vtype=GRB.CONTINUOUS, name="x")
        y = m.addVars(T, lb=self.map_y_min, ub=self.map_y_max, vtype=GRB.CONTINUOUS, name="y")

        ux = m.addVars(T, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="ux")
        uy = m.addVars(T, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="uy")

        expr = gp.QuadExpr()
        expr = 0
        for i in range(1,T-1):
            expr += (ux[i]**2 + uy[i]**2)

        # Set objective
        m.setObjective(expr, GRB.MINIMIZE)

        # Initial constraints
        m.addConstr(x[0] == start_x,  name="initx")
        m.addConstr(y[0] == start_y,  name="inity")
        m.addConstr(ux[0] == 0,name="initux")
        m.addConstr(uy[0] == 0,name="inituy")
        m.addConstr(x[T-1]-goal_x == 0, name = "end1" )
        m.addConstr(y[T-1]-goal_y == 0, name = 'end2')

        # System dynamics
        m.addConstrs((x[i] + tau*ux[i] == x[i+1] for i in range(T-1)),"c1")
        m.addConstrs((y[i] + tau*uy[i] == y[i+1] for i in range(T-1)),"c1")
        m.addConstrs((ux[i]**2 + uy[i]**2 <= 0.25 for i in range(T-1)),"c3")


        matrices = list(zip(A_list, b_list))

        bin_variables = []

        # Add all constraints for obstacles to gurobi solver
        for idx, (A, b) in enumerate(matrices): # use one set of binaries per obstacle
            bins = m.addMVar((A.shape[0],T), vtype=GRB.BINARY) # set of binaries for the obstacle
            bin_variables.append(bins)
            for i in range (A.shape[0]):
                namec = "matrix: " + str(idx) + " line: " + str(i)
                m.addConstrs((A[i,0]*x[t] + A[i,1]*y[t] <= b[0][i] + M*bins[i,t] for t in range(T-1)), name = namec )
            m.addConstrs((gp.quicksum(bins[i,t] for i in range(A.shape[0])) <= A.shape[0]-1 for t in range(T-1)), name = "matrix: " + str(idx)) 

        # Optimize model
        m.optimize()

        x_traj = []
        y_traj = []

        # m.computeIIS()

        # m.write("model.ilp")
        m.write("model.lp")


        if m.Status == GRB.OPTIMAL:

            for i in range(T-1):
                x_traj.append(x[i].X)
                y_traj.append(y[i].X)
                ax.plot([x[i].X, x[i+1].X],[y[i].X, y[i+1].X], 'ko-')
            x_traj.append(x[T-1].X)
            y_traj.append(y[T-1].X)

        plt.show()

        return (x_traj, y_traj)




    def initialise_pose_acquisition(self):
        '''
        Initialise the poses acuquistion by synchronizing all the topics and using the same callback.  

        arguments :
        ------------------------------------------------
        return :
        '''
        # Create sync callback group to get all the poses
        cb_group = ReentrantCallbackGroup()
        for id in self.ids:
            self.OptiTrack_sub.append(Subscriber(self, PoseStamped, f"/mocap_node/SycaBot_W{id}/pose"))
        self.ts = ApproximateTimeSynchronizer(self.OptiTrack_sub, queue_size=10, slop = 1.)
        self.ts.registerCallback(self.get_jb_pose_cb)

    def get_jb_pose_cb(self, *poses):
        '''
        Get and gather jetbot positions.
        arguments :
            *poses (PoseStamped) = array containing the position of the jetbots
        ------------------------------------------------
        return :
        '''
        quat = [poses[0].pose.orientation.x, poses[0].pose.orientation.y, poses[0].pose.orientation.z, poses[0].pose.orientation.w]
        theta = quat2eul(quat)
        self.jb_positions = np.array([[poses[0].pose.position.x, poses[0].pose.position.y, theta]])
        for p in poses[1:] :
            quat = [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
            theta = quat2eul(quat)
            self.jb_positions = np.append(self.jb_positions, np.array([[p.pose.position.x, p.pose.position.y, theta]]), axis=0)
    
        return       

    def wait4pose(self):
        while self.jb_positions is None :
            time.sleep(0.1)
            self.get_logger().info(f"Waiting for positions ...")
            rclpy.spin_once(self, timeout_sec=0.1)
 

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    node = MIP()

    executor.add_node(node)
    try :
        executor.spin()
    except Exception as e :
        print(e)
    finally:
        executor.shutdown()
        node.destroy_node()
    return


if __name__ == '__main__':
    main()
