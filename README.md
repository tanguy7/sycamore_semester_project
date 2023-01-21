# Learning to make mixed-integer robot motion planning easy

I recommend to first go through the report which is available [here](Report_Semester_project_tanguy.pdf) before checking the code.

## Files:

- [generate_data.py](generate_data.py): Part of the offline algorithm. Computes the visibility graph, dijkstra single source shortest path, formulates the optimization problem for the receding horizon framework ,and storse the data for the receding horizon local MIP problem that is required to train both the classifier for waypoints and for individual strategies.

- [Waypoint_prediction.ipynb](Waypoint_prediction.ipynb) : Part of the offline algorithm. Training notebook for predicting waypoint.

- [Strategies_prediction.ipynb](Strategies_prediction.ipynb) : Part of the offline algorithm. Training notebook for predicting binary strategies.

- [receding_learning.ipynb](receding_learning.ipynb): Online algorithm. Notebook for solving the receding horizon optimization problem with predicted binaries. Evaluates local problem solutions, and example of one global problem solved.

- [implementations.py](implementations.py): 
