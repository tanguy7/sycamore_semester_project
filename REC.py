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

import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt

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

import numpy as np
from shapely.geometry import box, MultiPolygon, Polygon, LineString, Point
from scipy.spatial import ConvexHull
import numpy.matlib as matlib
from numpy.linalg import matrix_rank
import torch
import gurobipy as gp
from gurobipy import GRB


def mapping_waypoint(class_waypoint, old_labels, labels):
  for idx, label in enumerate(labels):
      if class_waypoint == label:
        waypoint = old_labels[idx]
        continue
  return int(waypoint)

def mapping_strat(class_strat, strat_map_dict):
  c = [strat_map_dict[x].tolist() for x in class_strat]
  return c


def pol2ver(polygon):
    """
    pol2ver function gets the vertices from a polygon (shapely object).
    """
    xo, yo = polygon.exterior.xy
    vertices = np.array(list(set(list(zip(xo,yo)))))
    return vertices


def ver2con(V):
    """ 
    ver2con function maps vertices of a convex polygon
    to linear constraints stored in matrices A and b. And it works.
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

def map_def():

    map_x_min = -1.6
    map_x_max = 1.6
    map_y_min = -3.5
    map_y_max = 3.5


    # map_x_min = -2.5
    # map_x_max = 2.5
    # map_y_min = -1
    # map_y_max = 1.5


    map = [map_x_min, map_x_max, map_y_min, map_y_max]

    robot_radius = 0.2

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

    dilated_obstacles = []
    area_obstacles = MultiPolygon()


    A_list = [] # List of matrices A for A * x <= b for each obstacle
    b_list = []

    for line in obstacles:



        dilated = line.buffer(0.025 + robot_radius, cap_style = 3,join_style = 2)
        dilated = dilated.minimum_rotated_rectangle


        dilated_obstacles.append(dilated)
        area_obstacles = area_obstacles.union(dilated)


        vertices = pol2ver(dilated)
        A, b = ver2con(vertices)

        A_list.append(-A) #minus to get the constraint to be outside the polygon
        b_list.append(-b)

    area_obstacles = area_obstacles.simplify(tolerance=0.05)


    obstacles_boundary = []
    for geom in area_obstacles.geoms:    
        xo, yo = geom.exterior.xy
        obstacles_boundary.append(list(zip(xo,yo)))


    return(map, area_obstacles, obstacles_boundary, dilated_obstacles, robot_radius, A_list, b_list)

def get_nearest_obstacles(x, y, dilated_obstacles):
  idx_list = []
  pos = Point(x, y)
  for idx, obs in enumerate(dilated_obstacles):
    if pos.distance(obs) < 0.5: #0.75 is T*tau*umax 
      idx_list.append(idx)
  return idx_list

def get_features(starting_position, one_hot_obs):
  #starting position is a 1d matrix 1x2
  x0 = np.repeat(starting_position[0], repeats = 22, axis = 0)
  y0 = np.repeat(starting_position[1], repeats = 22, axis = 0)

  x0_torch = torch.from_numpy(x0).view(-1,1) # feature 1
  y0_torch = torch.from_numpy(y0).view(-1,1) # feature 2
  X = torch.hstack([x0_torch, y0_torch, one_hot_obs])
  return X        


def solve_mip_rec(A_list, b_list, start_x, start_y, vertices, waypoint, bin_seq, idx_list, T=3, tau=0.5):
    """
    solve_mip funtion uses Gurobi to solve a Mixed Integer Programming problem

    :param A_list:
    :param b_list:
    :param start_x:
    :param start_y:
    :param distance_array:
    :param vertices:
    :param ax:

    :return X_traj:
    :return Y_traj:
    :return U_traj:
    :return cost:
    :return X_traj[0,T-1]:
    :return X_traj[1, T-1]:
    :return ax:
    """

    # Create a new model
    m = gp.Model("mip1")    
    m.Params.LogToConsole = 0

    matrices = list(zip(A_list, b_list))

    vertices = [(1.3, 0), (0.22629999959955574, 2.775300400132888), (-1.273, 2.7732999998665777), (-1.273, 1.777436796878307), 
            (-0.4744367986471415, 1.7754353853460945), (-0.47556461465390576, 1.3254367986471416), (0.7925268154524867, 0.6535258182258848), 
            (1.2680873475579675, 0.6545269982934754), (1.2689126318094666, 2.7755075246459935), (0.825492673865953, 2.776508472632322),
            (0.793474181774115, 0.20352681545248683), (-0.22315045130358135, -2.774150551973562), (1.2717427843022142, -2.7731506233878727),
            (1.268742906365112, -0.1492573877361853), (-1.2708200000864573, -0.2751801440287423), (-1.2711073500192538, -0.6342238850579837), 
            (-0.2667760081307915, -0.635224215079785), (-0.267224215079785, -1.0852239918692086), (-1.2714674939684347, -1.0842237495595584), 
            (-1.272819855913565, -2.774), (-0.8250000000000001, -2.774), (0.705, -0.23300000000000012), (0.23600000000000002, -0.23300000000000007), 
            (0.23600000000000002, -0.711), (-0.214, -0.711), (-0.214, -0.233), (-0.48600000000000004, -0.23299999999999998), 
            (-0.49064740779348837, 0.29936098457022353), (-1.270003064041172, 0.3006313279381904), (-1.269501662966406, 0.07449944321876684), 
            (-1.2714994432187667, 0.9754983370335938), (-1.2710008516779385, 0.7506335521200315), (-0.37745968438862365, 0.7491770872996739), 
            (0.10481192590556132, 1.3561335529382093), (0.4571335529382091, 1.0761880740944387), (-0.04281640787757587, 0.4469827575555483), 
            (-0.03935802054591348, 0.21700000000000008), (0.705, 0.217), (-0.7961255394915645, 2.3378877073804762), 
            (-0.5251122926195231, 2.697125539491564), (1.1941255394915646, 1.4001122926195235), (0.9231122926195231, 1.0408744605084352),
            (1.1491415539057397, -2.317858356982588), (-0.5689280242231287, -2.3189395839040032), (-0.8941176571865752, -2.7300533068781165), 
            (-1.2470533068781164, -2.4508823428134248), (-0.8900594776107655, -1.9995608757315297), (-0.8901415539057398, -1.8691416430174121),
            (-0.7218823428134248, -1.7869466931218836), (-0.6181855865608905, -1.868970493887554), (0.035, -1.8685594268097938),
            (0.035, -0.8590000000000001), (0.485, -0.8590000000000001), (0.485, -1.8682762298305617), (0.6986959320518029, -1.8681417452918256),
            (0.6966961516652167, -0.38830425921807726), (1.1466957407819227, -0.3876961516652168)]

  
    matrices = [(np.array([[ 1.02616565e+00,  1.36913362e-03],
       [-5.92987390e-03,  4.44444049e+00],
       [-1.02616565e+00, -1.36913362e-03],
       [ 5.92987390e-03, -4.44444049e+00]]), np.array([[ -1.76397896,  12.33331554,  -0.23602104, -14.33331554]])), (np.array([[ 2.23963312, -0.00505561],
       [ 0.01003258,  4.44443312],
       [-2.23963312,  0.00505561],
       [-0.01003258, -4.44443312]]), np.array([[  1.8347638 ,  12.34828804,  -3.8347638 , -14.34828804]])), (np.array([[ 2.57685433e-04,  6.62251563e-01],
       [-4.44444411e+00,  1.72935568e-03],
       [-2.57685433e-04, -6.62251563e-01],
       [ 4.44444411e+00, -1.72935568e-03]]), np.array([[ 0.13614628, -7.63481143, -2.13614628,  5.63481143]])), (np.array([[ 4.44444154e+00,  5.08129749e-03],
       [-7.43845997e-04,  6.50617299e-01],
       [-4.44444154e+00, -5.08129749e-03],
       [ 7.43845997e-04, -6.50617299e-01]]), np.array([[ 5.63809526, -2.09805319, -7.63809526,  0.09805319]])), (np.array([[ 1.02827723e+00,  6.87810855e-04],
       [-2.97287187e-03,  4.44444345e+00],
       [-1.02827723e+00, -6.87810855e-04],
       [ 2.97287187e-03, -4.44444345e+00]]), np.array([[ -0.23136862, -14.32889185,  -1.76863138,  12.32889185]])), (np.array([[ 0.00000000e+00, -4.44444444e+00],
       [ 2.22717149e+00,  2.46716228e-16],
       [ 0.00000000e+00,  4.44444444e+00],
       [-2.22717149e+00, -0.00000000e+00]]), np.array([[ 12.32888889,  -3.83741648, -14.32888889,   1.83741648]])), (np.array([[ 4.44444302e+00, -3.55697721e-03],
       [ 5.42773587e-04,  6.78195597e-01],
       [-4.44444302e+00,  3.55697721e-03],
       [-5.42773587e-04, -6.78195597e-01]]), np.array([[-7.64710827, -2.18731573,  5.64710827,  0.18731573]])), (np.array([[ 4.44443352,  0.00985462],
       [-0.00492184,  2.21974764],
       [-4.44443352, -0.00985462],
       [ 0.00492184, -2.21974764]]), np.array([[-7.64148158,  0.17161824,  5.64148158, -2.17161824]])), (np.array([[ 0.        ,  1.05540897],
       [-4.44444444, -0.        ],
       [-0.        , -1.05540897],
       [ 4.44444444,  0.        ]]), np.array([[ 1.40158311,  5.65777778, -3.40158311, -7.65777778]])), (np.array([[ 2.67667404,  3.54802797],
       [-0.74136682,  0.55929585],
       [-2.67667404, -3.54802797],
       [ 0.74136682, -0.55929585]]), np.array([[ 6.16392241, -0.10220806, -8.16392241, -1.89779194]])), (np.array([[ 1.11389235e-02,  4.44443049e+00],
       [-1.60255585e+00,  4.01643070e-03],
       [-1.11389235e-02, -4.44443049e+00],
       [ 1.60255585e+00, -4.01643070e-03]]), np.array([[ 5.88551444,  0.76744238, -7.88551444, -2.76744238]])), (np.array([[ 0.0093567 , -4.4444346 ],
       [ 2.16215491,  0.00455191],
       [-0.0093567 ,  4.4444346 ],
       [-2.16215491, -0.00455191]]), np.array([[-2.89713732,  1.71654053,  0.89713732, -3.71654053]])), (np.array([[ 3.47971977, -2.76489362],
       [ 1.12185599,  1.41189681],
       [-3.47971977,  2.76489362],
       [-1.12185599, -1.41189681]]), np.array([[-3.38484887,  0.03230452,  1.38484887, -2.03230452]])), (np.array([[ 1.19260310e+00, -1.94393333e-03],
       [ 7.24439860e-03,  4.44443854e+00],
       [-1.19260310e+00,  1.94393333e-03],
       [-7.24439860e-03, -4.44443854e+00]]), np.array([[-2.0534928 ,  1.32693705,  0.0534928 , -3.32693705]])), (np.array([[ 4.44394202,  0.0668262 ],
       [-0.03062113,  2.03630492],
       [-4.44394202, -0.0668262 ],
       [ 0.03062113, -2.03630492]]), np.array([[-2.16040348, -0.46651873,  0.16040348, -1.53348127]])), (np.array([[ 5.59306310e-16,  4.44444444e+00],
       [-1.67926113e+00, -0.00000000e+00],
       [-3.72870873e-16, -4.44444444e+00],
       [ 1.67926113e+00, -2.46716228e-16]]), np.array([[-1.03555556, -1.18387909, -0.96444444, -0.81612091]])), (np.array([[ 0.        ,  2.15517241],
       [-4.44444444, -0.        ],
       [-0.        , -2.15517241],
       [ 4.44444444,  0.        ]]), np.array([[-1.53232759, -1.04888889, -0.46767241, -0.95111111]])), (np.array([[ 4.42673530e-03,  4.44444224e+00],
       [-1.37551467e+00,  1.37003453e-03],
       [-4.42673530e-03, -4.44444224e+00],
       [ 1.37551467e+00, -1.37003453e-03]]), np.array([[-4.82439828,  0.36608403,  2.82439828, -2.36608403]])), (np.array([[ 4.44444039e+00,  6.00600052e-03],
       [-1.40036184e-03,  1.03626776e+00],
       [-4.44444039e+00, -6.00600052e-03],
       [ 1.40036184e-03, -1.03626776e+00]]), np.array([[ 3.09409236, -2.40336281, -5.09409236,  0.40336281]])), (np.array([[-4.93432455e-16,  1.37080192e+00],
       [-4.44444444e+00, -0.00000000e+00],
       [-4.93432455e-16, -1.37080192e+00],
       [ 4.44444444e+00,  0.00000000e+00]]), np.array([[-3.17751885, -2.15555556,  1.17751885,  0.15555556]])), (np.array([[ 9.80872631e-01,  6.17289258e-04],
       [-2.79700665e-03,  4.44444356e+00],
       [-9.80872631e-01, -6.17289258e-04],
       [ 2.79700665e-03, -4.44444356e+00]]), np.array([[ -0.87426929, -10.30480481,  -1.12573071,   8.30480481]])), (np.array([[ 3.48578419, -2.75724409],
       [ 1.46569887,  1.85297703],
       [-3.48578419,  2.75724409],
       [-1.46569887, -1.85297703]]), np.array([[ 2.41072215, -6.36923332, -4.41072215,  4.36923332]]))]

    if idx_list == None:
      matrices = []
    else:
      matrices = [matrices[i] for i in idx_list]

    # Constraints on command
    u_max = 0.5  # [m/s]
    u_min = -0.5  # [m/s]

    # big M
    M = 1000

    # Create variables
    x = m.addVars(T ,lb=-1.6 ,ub =1.6 , vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(T ,lb=-3 ,ub=3 , vtype=GRB.CONTINUOUS, name="y")

    ux = m.addVars(T, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="ux")
    uy = m.addVars(T, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="uy")

    expr = gp.QuadExpr()
    expr += (vertices[waypoint][1] - y[T-1])**2  +(vertices[waypoint][0] - x[T-1])**2 
            

    # Set objective
    m.setObjective(expr, GRB.MINIMIZE)

    # Initial constraints
    m.addConstr(x[0] == start_x,  name="initx")
    m.addConstr(y[0] == start_y,  name="inity")

    # System dynamics
    m.addConstrs((x[i] + tau*ux[i] == x[i+1] for i in range(T-1)),"c1")
    m.addConstrs((y[i] + tau*uy[i] == y[i+1] for i in range(T-1)),"c2")
    m.addConstrs((ux[i]**2 + uy[i]**2 <= 0.25 for i in range(T-1)),"c3")

    bin_variables = []

    for idx, (A, b) in enumerate(matrices): # use one set of binaries per obstacle
        #print(bin_seq[idx])
        for i in range (A.shape[0]):
          namec = "matrix: " + str(idx) + " line: " + str(i)
          m.addConstr((A[i,0]*x[0] + A[i,1]*y[0] <= b[0][i] - 0.001 + M*bin_seq[idx][i*(T) + (0)] ), name = namec )
          m.addConstr((A[i,0]*x[1] + A[i,1]*y[1] <= b[0][i] - 0.001 + M*bin_seq[idx][i*(T) + (1)] ), name = namec )
          m.addConstr((A[i,0]*x[2] + A[i,1]*y[2] <= b[0][i] - 0.001 + M*bin_seq[idx][i*(T) + (2)] ), name = namec )
 
    # Optimize model
    m.optimize()
    m.write('model.mps')

    # m.computeIIS()
    # m.write('dsr.ilp')

    X_traj = np.zeros((2,T)) # to store positions
    Y_traj = np.zeros((4*len(matrices),(T))) # to store binaries
    U_traj = np.zeros((2,T))

    if m.Status == GRB.OPTIMAL:

      for i in range(T):
          X_traj[0,i] = x[i].X
          X_traj[1,i] = y[i].X
          U_traj[0,i] = ux[i].X
          U_traj[1,i] = uy[i].X

      obj = m.getObjective()
      cost = obj.getValue()
      return (X_traj, Y_traj, U_traj, cost)
  
    else:
        return(None, None, None, None)     
    
    

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=2, output_dim=22, n_hidden = 50 ):
    super(LinearClassifier, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, n_hidden)
    self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
    self.fc3 = torch.nn.Linear(n_hidden, output_dim)


  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x

#This neural network is used to predict the strategies

class LinearClassifier2(torch.nn.Module):
  def __init__(self, input_dim=24, output_dim= 90, n_hidden = 256 ):  
    super(LinearClassifier2, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, n_hidden)
    self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
    self.fc3 = torch.nn.Linear(n_hidden, output_dim)


  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x


class REC(Node):
    # MAX_LIN_VEL = 0.3
    def __init__(self):
        super().__init__("REC_pathplanner")

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
                destinations[rob,0] = 0.5
                destinations[rob,1] = 3.1
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
        N = 1
        rounds = 1
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
                trajectory = self.solve_mip(ax, A_list, b_list, start_x, start_y, goal_x, goal_y)
                # extract first components of tuples in trajetory and put them in a list
                x_traj = [x[0] for x in trajectory]
                y_traj = [x[1] for x in trajectory]
                # put multiple times the last value to reach 60 items in the list
                x_traj = x_traj + [x_traj[-1]]*(60-len(x_traj))
                y_traj = y_traj + [y_traj[-1]]*(60-len(y_traj))

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

        data = np.load("sycabot_pathplanner/sycabot_pathplanner/data_simple.npz")
        old_labels = np.unique(data['V'][:240000].round())
        len(np.unique(data['V'][:240000].round()))
        # We need to make these classes range from 0 to n
        labels = np.array((range(len(old_labels))))

        model_target = LinearClassifier()
        model_strat = LinearClassifier2()

        checkpoint_strat = torch.load('sycabot_pathplanner/sycabot_pathplanner/pred_strategies.ckpt')
        model_strat.load_state_dict(checkpoint_strat)

        checkpoint_target = torch.load('sycabot_pathplanner/sycabot_pathplanner/pred_candidates.ckpt')
        model_target.load_state_dict(checkpoint_target)


        with open('sycabot_pathplanner/sycabot_pathplanner/strategy_mapping1912.pkl', 'rb') as f:
            strat_map_dict = pickle.load(f)
        # Get features
        # Solve problems

        xend, yend = 1.3, 0      # This position is fixed
        T=3
        tau = 0.5
        umax = 0.5

        map, area_obstacles, obstacles_boundary, dilated_obstacles, robot_radius, A_list, b_list = map_def()

        vertices = [(1.3, 0), (0.22629999959955574, 2.775300400132888), (-1.273, 2.7732999998665777), (-1.273, 1.777436796878307), 
                    (-0.4744367986471415, 1.7754353853460945), (-0.47556461465390576, 1.3254367986471416), (0.7925268154524867, 0.6535258182258848), 
                    (1.2680873475579675, 0.6545269982934754), (1.2689126318094666, 2.7755075246459935), (0.825492673865953, 2.776508472632322),
                    (0.793474181774115, 0.20352681545248683), (-0.22315045130358135, -2.774150551973562), (1.2717427843022142, -2.7731506233878727),
                    (1.268742906365112, -0.1492573877361853), (-1.2708200000864573, -0.2751801440287423), (-1.2711073500192538, -0.6342238850579837), 
                    (-0.2667760081307915, -0.635224215079785), (-0.267224215079785, -1.0852239918692086), (-1.2714674939684347, -1.0842237495595584), 
                    (-1.272819855913565, -2.774), (-0.8250000000000001, -2.774), (0.705, -0.23300000000000012), (0.23600000000000002, -0.23300000000000007), 
                    (0.23600000000000002, -0.711), (-0.214, -0.711), (-0.214, -0.233), (-0.48600000000000004, -0.23299999999999998), 
                    (-0.49064740779348837, 0.29936098457022353), (-1.270003064041172, 0.3006313279381904), (-1.269501662966406, 0.07449944321876684), 
                    (-1.2714994432187667, 0.9754983370335938), (-1.2710008516779385, 0.7506335521200315), (-0.37745968438862365, 0.7491770872996739), 
                    (0.10481192590556132, 1.3561335529382093), (0.4571335529382091, 1.0761880740944387), (-0.04281640787757587, 0.4469827575555483), 
                    (-0.03935802054591348, 0.21700000000000008), (0.705, 0.217), (-0.7961255394915645, 2.3378877073804762), 
                    (-0.5251122926195231, 2.697125539491564), (1.1941255394915646, 1.4001122926195235), (0.9231122926195231, 1.0408744605084352),
                    (1.1491415539057397, -2.317858356982588), (-0.5689280242231287, -2.3189395839040032), (-0.8941176571865752, -2.7300533068781165), 
                    (-1.2470533068781164, -2.4508823428134248), (-0.8900594776107655, -1.9995608757315297), (-0.8901415539057398, -1.8691416430174121),
                    (-0.7218823428134248, -1.7869466931218836), (-0.6181855865608905, -1.868970493887554), (0.035, -1.8685594268097938),
                    (0.035, -0.8590000000000001), (0.485, -0.8590000000000001), (0.485, -1.8682762298305617), (0.6986959320518029, -1.8681417452918256),
                    (0.6966961516652167, -0.38830425921807726), (1.1466957407819227, -0.3876961516652168)]



        nums_obstacle = torch.arange(22, dtype=torch.long)
        one_hot_obs = F.one_hot(nums_obstacle, num_classes = 22)

        trajectory = []

        # starting point
        xlast = start_x
        ylast = start_y
        fail = 0
        solved = 0

        trajectory.append((xlast,ylast))

        time_to_solve = time.time()
        while (xlast-xend)**2 + (ylast-yend)**2 > 0.01:
            X_target = torch.Tensor([xlast, ylast])

            # ---------- prediction of the waypoint--------------
            scores_waypoint = F.softmax(model_target(X_target.float())).cpu().detach().numpy()[:]
            waypoint = mapping_waypoint(int(np.argmax(scores_waypoint)), old_labels, labels)
            #cvis = distance_array[waypoint]

            #------------ prediction of the strategy------------
            #_, predicted_binaries = torch.max(F.softmax(model_strat(X_strat.float())),-1)
            X = get_features([xlast, ylast], one_hot_obs)
            scores = F.softmax(model_strat(X.float())).cpu().detach().numpy()[:]

            # ----------getting only the scores of nearby obstacles-----------
            idx_list = get_nearest_obstacles(xlast, ylast, dilated_obstacles)
            scores = scores[idx_list, :]
            sorted_scores = np.argsort(scores, axis=1)[:, -2:][:,::-1]

            #--------------creating stategy tuples------------------
            predicted_binaries = sorted_scores[:,0]
            bin_seq = mapping_strat(predicted_binaries, strat_map_dict)

            #---- solving the MIP problem for the current state ----
            X_traj, Y_traj, U_traj, cost = solve_mip_rec(A_list, b_list, xlast, ylast, vertices, waypoint, bin_seq, idx_list)


            if X_traj is not None:
                print('OK')
                tmp = X_traj[:, :]
                xlast = X_traj[0, 1]
                ylast = X_traj[1, 1]
                trajectory.append((xlast, ylast))
                fail = 0
                solved += 1

            else:
                if solved == 0:
                    print("No feasible trajectory found")
                    break
                fail += 1
                if fail >= 2:
                    print("No feasible trajectory found")
                    break
                xlast = tmp[0, 1+fail]
                ylast = tmp[1, 1+fail]
                trajectory.append((xlast,ylast))

        time_to_solve = time.time()-time_to_solve
        print('Time to solve: ', time_to_solve/len(trajectory))

        print(trajectory)

        # plotting the trajectory
        #fig,ax = plt.subplots(figsize=(4,7))
        for k in range(len(trajectory)):
            plt.plot(trajectory[k][0], trajectory[k][1], color = 'green', marker = 'o')


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
        for line in obstacles:
                dilated = line.buffer(0.025 + robot_radius, cap_style = 3,join_style = 2)
                
                dilated = dilated.minimum_rotated_rectangle

                dilated_obstacles.append(dilated)
                area_obstacles = area_obstacles.union(dilated)
                x,y = dilated.exterior.xy
                plt.plot(x,y, 'b')
                
        plt.show()

        return trajectory


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

    node = REC()

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
