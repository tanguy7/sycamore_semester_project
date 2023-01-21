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