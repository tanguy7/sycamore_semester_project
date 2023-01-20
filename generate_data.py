"""
File name : generate_data.py
Description : This file generates data for the MIP model.
Author : Tanguy Lewko
Last modified : 27/12/2022
Python version : 3.9.7
"""

import pdb
import numpy as np
from scipy.spatial import ConvexHull
import numpy.matlib as matlib
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from shapely.geometry import box, MultiPolygon, Polygon, LineString, Point
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pl
from matplotlib import collections  as mc
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
from tqdm import trange
import time

def pol2ver(polygon):
    """
    pol2ver function gets the vertices from a polygon (shapely object).

    :param polygon: shapely polygon object

    :return vertices: vertices of the polygon
    """
    # Get vertices of the polygon
    xo, yo = polygon.exterior.xy
    # Remove duplicates
    vertices = np.array(list(set(list(zip(xo,yo)))))
    return vertices


def ver2con(V):
    """ 
    ver2con function maps vertices of a convex polygon
    to linear constraints stored in matrices A and b. 

    :param V: vertices of a convex polygon (numpy array)

    :return A: matrix of linear constraints
    :return b: vector of linear constraints
    """

    # Get the convex hull of the vertices
    k = ConvexHull(V).vertices 
    u = np.roll(k,-1)
    k = np.vstack((k,u)).T
    c = np.mean(V[k[:,0]], axis = 0)

    # Center the vertices
    V = V - matlib.repmat(c, V.shape[0],1)

    # Initialize matrices
    A = np.zeros((k.shape[0], V.shape[1]))
    A[:] = np.nan
    rc = 0

    for ix in range(k.shape[0]):
        # Get the linear constraints
        F = V[k[ix]]
        # Check if the linear constraints are independent
        if matrix_rank(F, 1e-5) == F.shape[0]:
            # Add the linear constraints to the matrix
            rc = rc+1
            A[rc-1,:] = np.linalg.solve(F,np.ones((F.shape[0])))

    # Remove the empty rows
    A = A[0:rc,:]

    # Add the constant term
    b = np.ones((A.shape[0], 1))
    b = b.T + A @ c

    return(A, b)



def map_def():
    """
    map_def function defines the map and the obstacles.
    """

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
        
        # Dilate the obstacles
        dilated = line.buffer(0.025 + robot_radius, cap_style = 3,join_style = 2)
        dilated = dilated.minimum_rotated_rectangle

        dilated_obstacles.append(dilated)
        # Create a union of polygons
        area_obstacles = area_obstacles.union(dilated)

        # Convert the polygon to linear constraints
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



def visible(a,b, polygon_obstacle, polygon_map):
    """
    visible function determines if the robot can travel
    between two points using a straigth line without leaving 
    the map or touch an obstacle

    :param a: origin point
    :param b: destination point
    :param polygon_obstacle: list of polygon representing the dilated obstacles
    :param polygon_map: polygon representing the dilated map

    :return visible: True if the line does not cross obstacles or leave map
                     False otherwise
    """
    visible = True
    line = LineString([a,b])
    for x in polygon_obstacle:
        within_obstacle = line.within(x)
        crosses_obstacle = line.crosses(x)
        if within_obstacle == True or crosses_obstacle == True:
            visible = False     
    in_map = polygon_map.contains(line)
    if in_map == False:
        visible = False
    return visible


def distance(a,b):
    """
    distance function compute the eucledean distance between two points

    :param a: first point
    :param b: second point

    :return dist: eucledean distance between points a and b
    """
    dist = np.linalg.norm(np.array(a)-np.array(b))

    return(dist)    


def vis_graph(end, area_obstacles, obstacles,polygon_obstacle,polygon_map):
    """
     vis_graph function computes the visibility graph based on the computer vision results

     :param end: point to reach
     :param obstacles: list of vertices of the obstacles
     :param polygon_obstacle: list of polygons of dilated obstacles
     :param polygon_map: polygon of the dilated map

     :return graph: dictionnary containing in the key the index of the vertex of interest 
                    and for the value, other vertices with distance that are valid paths
                    for the robot
     :return end_idx: index of the robot vertex in vertices list
     :return targets_idx_list: list of the indexes of the targets (based on vertices list)
     :return vertices: List of all vertices, robot position, vertices of obstacles, and targets
     """
    graph = {}
    end_idx = 0
    obstacles = [item for sublist in obstacles for item in sublist]

    vertices_with_rep = obstacles
    vertices = []
    for i in vertices_with_rep : 
        if i not in vertices and polygon_map.contains(Point(i)): 
           vertices.append(i)

    vertices= [end] +  vertices

    # Compute the visibility graph
    for vtx1_idx, vtx1 in enumerate(vertices):
        for vtx2_idx, vtx2 in enumerate(vertices):
            if vtx1_idx != vtx2_idx:
                if visible(vtx1,vtx2,area_obstacles,polygon_map)==True:
                    if vtx1_idx in graph :
                        graph[vtx1_idx].append([vtx2_idx,distance(vtx1,vtx2)])
                    else:
                        graph[vtx1_idx] = [(vtx2_idx,distance(vtx1,vtx2))]
                else:
                    if not(vtx1_idx in graph):
                        graph[vtx1_idx] = []

    return graph, end_idx, vertices


def display_map(visibility_graph, vertices, dilated_obstacles, end_idx):
    """
    display_map function displays the map with the visibility graph

    :param visibility_graph: dictionnary containing in the key the index of the vertex of interest
                                and for the value, other vertices with distance that are valid paths
                                for the robot
    :param vertices: List of all vertices, robot position, vertices of obstacles, and targets
    :param dilated_obstacles: list of polygons of dilated obstacles
    :param end_idx: index of the robot vertex in vertices list
    """

    fig, ax = plt.subplots()
    # for i in id:
    #     plt.scatter(vertices[i][1], vertices[i][0], marker = 'x', color = 'red')
    # for i in visibility_graph:

        #plt.scatter(vertices[i][1], vertices[i][0], marker = 'o', color = 'orange')
        # lines = []
        # for j in visibility_graph[i]:
        #     lines.append([tuple((vertices[i][1], vertices[i][0])), tuple((vertices[j[0]][1], vertices[j[0]][0]))])
        # if len(lines)>0:
        #     vis = mc.LineCollection(lines)
        #     ax.add_collection(vis)

    for dilated_obstacle in dilated_obstacles:
        x_obst, y_obst = dilated_obstacle.exterior.xy
        lst = list(zip(y_obst, x_obst))
        arr = np.zeros((len(lst),2))
        for i in range(len(lst)):
            arr[i,:] = lst[i]
        poly = pl(arr, closed = True, alpha = 1)
        collection = PatchCollection([poly], color = 'grey')
        ax.add_collection(collection)

        # poly = list((point[0],point[1]) for point in list(zip(x_obst, y_obst)))
        # polysides = []
        # for i in range(len(poly)-1):
        #     polysides.append([poly[i], poly[i+1]])
        # if len(polysides)>0:
        #     poly = mc.LineCollection(polysides, color = 'red')
        #     ax.add_collection(poly)
    ax.set_aspect('equal')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1.5)
    #plt.scatter(vertices[end_idx][1], vertices[end_idx][0], marker = 'x', linewidth= 3, s=100, color = 'red')

    return ax


def dijkstra_aglorithm(idx_end, visibility_graph):
    """
    This algorithm finds the shortest distance and the associated path from one vertex to another in a graph

    :param idx_start: index of the starting vertex
    :param idx_end: index of the ending vertex
    :param visibility_graph: a graph (python dictionnary where keys = vertices and value = connected vertices and the corresponding distance)

    :return distance_array[idx_end]: the shortest distance from the starting index to the ending in the visibility graph
    :return path: the path associated, which is a list of the indices of the vertices in the graph
    """

    #calulates the shortest distance from goal to all vertices
    nb_points = len(visibility_graph)
    distance_array = np.full(nb_points, np.Inf, dtype=np.double)  # creates an array to store distance to start
    distance_array[idx_end] = 0
    explored_array = np.full(nb_points, False, dtype=bool)  # keeps track of explored points
    explored_array[idx_end] = True
    exploring_idx = idx_end  # from where we are exploring


    while(not all(explored_array)):
        for vertices in visibility_graph[exploring_idx]:
            if distance_array[exploring_idx] + vertices[1] < distance_array[vertices[0]]:
                distance_array[vertices[0]] = distance_array[exploring_idx] + vertices[1]  # replace shortest distance
        temp_idx = np.argmin(distance_array[np.logical_not(explored_array)]) # 
        exploring_idx = np.arange(nb_points)[np.logical_not(explored_array)][temp_idx]
        explored_array[exploring_idx] = True

    return distance_array

def find_candidates(start, H, tau, umax, visibility_graph, polygon_map, area_obstacles):
    """
    This function gets the candidates for pvis. From the starting point, it looks at all the visible vertices.
    If they are within the horizon distance that can be covered, we search for their successors in the visibility graph 
    previously created. 

    :param start: coordinates of the starting vertex
    :param H: horizon
    :param tau: step time
    :param u_max: maximum speed of the robot
    :param visibility_graph: dictionnary containing in the key the index of the vertex of interest 
                            and for the value, other vertices with distance that are valid paths
                            for the robot
    :param distance_array: The shortest distance from all obstacles vertices to the goal point

    :return candidates: The set of candidates for searching x_vis in the MIP solving
    """

    candidates_idx = []
    candidates_dist = []
    seen = []

    # estimate the horizon distance
    d_horizon = (H-1) * tau * umax

    # look at visible points for the starting vertex
    for vtx2_idx, vtx2 in enumerate(vertices):
        if visible(start, vtx2, area_obstacles, polygon_map)==True:
            candidates_idx.append(vtx2_idx)
            candidates_dist.append(distance(start, vtx2))
            seen.append(False)

    #iterative deepeming to find the candidates
    while not all(seen):
        for idx, candidate in enumerate(candidates_idx):
            if seen[idx] == False:
                if candidates_dist[idx] <= d_horizon:
                    for vertex in visibility_graph[candidate]:
                        if vertex[0] not in candidates_idx:
                            candidates_idx.append(vertex[0])
                            candidates_dist.append(candidates_dist[idx] + vertex[1])
                            seen.append(False)
                seen[idx] = True
                    
    return candidates_idx


def solve_mip(A_list, b_list, start_x, start_y, distance_array, vertices, T, tau, candidates_idx):
    """
    solve_mip funtion uses Gurobi to solve a Mixed Integer Programming problem

    :param A_list: list of matrices A
    :param b_list: list of matrices b
    :param start_x: x coordinate of the starting point
    :param start_y: y coordinate of the starting point
    :param distance_array: array of the shortest distance from all vertices to the goal
    :param vertices: list of vertices
    :param ax: axis to plot the results

    :return X_traj: trajectory of the robot
    :return Y_traj: binaries associated to big-M constraints
    :return U_traj: commands of the robot
    :return cost: cost of the trajectory
    :return X_traj[0,T-1]: last x coordinate of the trajectory
    :return X_traj[1, T-1]: last y coordinate of the trajectory
    :return ax: axis to plot the results
    """
    # Create a new model
    m = gp.Model("mip1")    
    m.Params.LogToConsole = 0


    matrices = list(zip(A_list, b_list))

    # Pre-computed matrices (in the same way as above)
    matrices =[(np.array([[ 1.02616565e+00,  1.36913362e-03],
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

    xvis = m.addVars(2, lb = -3, ub = 3, vtype=GRB.CONTINUOUS, name="xvis")
    Cvis = m.addVar(vtype=GRB.CONTINUOUS, name="Cvis")

    # Get number of vertices
    bcost = m.addMVar(len(candidates_idx), vtype=GRB.BINARY)

    # Objective function
    expr = gp.QuadExpr()
    expr += (xvis[0] - x[T-1])**2 + (xvis[1] - y[T-1])**2  + Cvis**2


    # Set objective
    m.setObjective(expr, GRB.MINIMIZE)
    m.addConstr(xvis[0] == gp.quicksum(bcost[j]*vertices[candidates_idx[j]][0] for j in range(len(candidates_idx))), "xvis1")
    m.addConstr(xvis[1] == gp.quicksum(bcost[j]*vertices[candidates_idx[j]][1] for j in range(len(candidates_idx))), "xvis2")
    m.addConstr(Cvis == gp.quicksum(bcost[j]*distance_array[candidates_idx[j]] for j in range(len(candidates_idx))), "Cvis")
    m.addConstr(gp.quicksum(bcost[j] for j in range(len(candidates_idx))) == 1 , 'one_vertex')

    # Initial constraints
    m.addConstr(x[0] == start_x,  name="initx")
    m.addConstr(y[0] == start_y,  name="inity")

    # System dynamics
    m.addConstrs((x[i] + tau*ux[i] == x[i+1] for i in range(T-1)),"c1")
    m.addConstrs((y[i] + tau*uy[i] == y[i+1] for i in range(T-1)),"c2")
    m.addConstrs((ux[i]**2 + uy[i]**2 <= 0.25 for i in range(T-1)),"c3")

    bin_variables = []

    # Add all constraints for obstacles to gurobi solver
    for idx, (A, b) in enumerate(matrices): # use one set of binaries per obstacle
        bins = m.addMVar((A.shape[0],T), vtype=GRB.BINARY) # set of binaries for the obstacle
        bin_variables.append(bins)
        for i in range (A.shape[0]):
            namec = "matrix: " + str(idx) + " line: " + str(i)
            m.addConstrs((A[i,0]*x[t] + A[i,1]*y[t] <= b[0][i] -0.001 + M*bins[i,t] for t in range(T)), name = namec )
        m.addConstrs((gp.quicksum(bins[i,t] for i in range(A.shape[0])) <= A.shape[0]-1 for t in range(T)), name = "matrix: " + str(idx)) 


    # Optimize model
    m.optimize()

    #m.write("model.lp")
    # m.computeIIS()
    # m.write("model.ilp")

    X_traj = np.zeros((2,T)) # to store positions
    Y_traj = np.zeros((4*len(matrices),(T))) # to store binaries
    U_traj = np.zeros((2,T-1))

    if m.Status == GRB.OPTIMAL:

        for i in range(T-1):
            X_traj[0,i] = x[i].X
            X_traj[1,i] = y[i].X
            U_traj[0,i] = ux[i].X
            U_traj[1,i] = uy[i].X
            for idx1, bins in enumerate(bin_variables):
                for k in range (bins.shape[0]):
                    Y_traj[idx1*4 + k, i] = bins[k,i].X
        X_traj[0,T-1] = x[T-1].X
        X_traj[1,T-1] = y[T-1].X

        for idx1, bins in enumerate(bin_variables):
            for k in range (bins.shape[0]):
                Y_traj[idx1*4 + k, T-1] = bins[k,T-1].X

        obj = m.getObjective()
        cost = obj.getValue()

    else:
        return(None, None, None, None, None, None, None)

    return (bcost.X, X_traj, Y_traj, U_traj, cost, X_traj[0,T-1], X_traj[1, T-1])


def generate_params(map_x_min, map_x_max, map_y_min, map_y_max, area_obstacles):
    """
    Generate random initial positions for the agent
    that are not inside the obstacles

    :param map_x_min: minimum x coordinate of the map
    :param map_x_max: maximum x coordinate of the map
    :param map_y_min: minimum y coordinate of the map
    :param map_y_max: maximum y coordinate of the map
    :param area_obstacles: shapely polygon of the obstacles
    
    :return: numpy array of shape (2, num_params) containing the initial positions
    """

    num_params = 500000
    num_start = 0
    init_pos = np.zeros((2,num_params))
    while num_start < num_params:
        x0 = map_x_min + (map_x_max-map_x_min)*np.random.rand(1)
        y0 = map_y_min + (map_y_max-map_y_min)*np.random.rand(1)
        if not Point((x0,y0)).within(area_obstacles):
            init_pos[0,num_start] = x0
            init_pos[1,num_start] = y0
            num_start += 1
            if not (num_start%100000):
                print(num_start)
    return init_pos



if __name__ == '__main__':
    T = 5        # horizon  3
    tau = 0.05    # time step 0.5
    umax = 0.5   # maximum control input
    xend = 1.3   # end x position of the agent
    yend = 0     # end y position of the agent

    # Generate map
    map, area_obstacles, obstacles_boundary, dilated_obstacles, robot_radius, A_list, b_list = map_def()
    # Generate graph
    graph, end_idx, vertices = vis_graph((xend,yend), area_obstacles, obstacles_boundary, dilated_obstacles, Polygon(box(-1.6, -3, 1.6, 3)))
    # Compute shortest path
    distance_array = dijkstra_aglorithm(end_idx, graph)
    # ax = display_map(graph, vertices, dilated_obstacles, end_idx)

    # plt.scatter(ylast, xlast, marker = 'o', color = 'darkgreen')
    # plt.scatter(yend, xend, marker = 'x', color = 'red')

    # Generate initial positions
    init_pos = generate_params(-1.6, 1.6, -3, 3, area_obstacles)
    num_probs = int(init_pos.shape[1]/2)

    # Storing variables
    X = np.zeros((num_probs,2,T))                           # x and y positions
    Y = np.zeros((num_probs,4*len(A_list),T)).astype(int)   # binaries
    U = np.zeros((num_probs,2,T-1))                         # input velocities
    V = np.zeros(num_probs)                                 # selected vertex
    costs = np.zeros((num_probs))                           # cost of optimize objective function
    solve_times = np.zeros((num_probs))                     # solve times to solve subproblems
    param_x0y0 = np.zeros((num_probs,2))                    # Solving parameter : starting point

    i=0 
    for k in trange(num_probs*2):    # bar stops at 50% of the progress

        xlast ,ylast = init_pos[0,k], init_pos[1,k]
        x_prec = xlast
        y_prec = ylast    
        t = time.time()    
        # Find candidates for the next vertex to visit
        candidates_idx = find_candidates((xlast, ylast), T, tau, umax, graph,  Polygon(box(-1.6, -3, 1.6, 3)), area_obstacles)
        # Solve subproblem
        bcost, X_traj, Y_traj, U_traj, cost, xlast, ylast = solve_mip(A_list, b_list, xlast, ylast, distance_array, vertices, T, tau, candidates_idx)
        t = time.time()-t

        if bcost is not None:
            # Find the index of the selected vertex
            v = np.dot(bcost, candidates_idx)
            # Update the position of the agent
            param_x0y0[i,0] = x_prec
            param_x0y0[i,1] = y_prec
            X[i,:,:] = X_traj
            Y[i,:,:] = Y_traj
            U[i,:,:] = U_traj
            V[i] = v
            costs[i] = cost
            solve_times[i] = t
            i = i+1
        else:
            pass

        if not (i%10000):
            print(" iter {} on {}".format(i, num_probs))
            
        if i >= num_probs:
            break

    np.savez('data_simple.npz', X = X, Y = Y, U = U, solve_times=solve_times, V=V, param_x0y0=param_x0y0, costs=costs)