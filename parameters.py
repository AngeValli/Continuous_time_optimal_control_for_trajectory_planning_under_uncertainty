"Script with the parameters of the model"
import numpy as np

# Continuous-time parameters
T: int = 2 # Time horizon
N_DIM: int = 50 # Discretisation time step
N_DIM_CSV: int = 200 # Number of measurements in a CSV file generated
STD_X_TGTS: np.ndarray = np.array([1] * N_DIM) # Standards deviations for stochastic target vehicle coordinates x through dimension
STD_Y_TGTS: np.ndarray = np.array([1] * N_DIM) # Standards deviations for stochastic target vehicle coordinates y through dimension

# Input parameters for a scenario
ROAD_LENGTH_MAX: int = 10**3 # Frontier of the road on x-axis
ROAD_WIDTH_MAX: int = 5 # Frontier of the road on y-axis
V_RECOM: int = 14 # Recommended speed
A_MAX: int = 2 # Maximum acceleration
J_MAX: float = 0.6 # Maximum jerk
OMEGA_MAX: float = np.pi/6 # Threshold for angular speed
V_MAX: int = 40 # Maximum linear speed
DIS_MIN: int = 17 # Minimum distance between ego and target vehicles

ALPHA: float = 0.95 # Threshold of confidence for the chance constraint to be verified

# Weights of each component of the objective function
W_GX: int = 1 # Weight for the distance to the next waypoint on the x-axis
W_GY: int = 1 # Weight for the distance to the next waypoint on the y-axis
W_V: int = 1 # Weight for the error between current speed and recommended speed
W_A: int = 1 # Weight for controlling the linear acceleration
W_W: int = 1 # Weight for controlling the angular speed
W_J: int = 1 # Weight for controlling the jerk
W_H: int = 1 # Weight for controlling the angle between the head of the vehicle and the centre lane of the road
W_P: int = 1 # Weight for controlling the potential field function : headway distance between the ego and target vehicles