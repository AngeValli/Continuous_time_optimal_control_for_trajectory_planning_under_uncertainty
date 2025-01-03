"Script to implement deterministic and stochastic models for trajectory planification"
from typing import List, Tuple

import constants as c
import parameters as p

import pandas as pd
import numpy as np
from scipy.stats import norm
from gekko import GEKKO


class Vehicle:
    """
    Class for initializing a vehicle
    """

    ###############################################################################################################################
    # Initialize vehicle
    ###############################################################################################################################

    def __init__(self, csv_path: str, time_horizon: int = p.T, discretization_nb: int = p.N_DIM,
                 V_RECOM: int = p.V_RECOM, OMEGA_MAX: float = p.OMEGA_MAX, DIS_MIN: int = p.DIS_MIN,
                 V_MAX: int = p.V_MAX, W_GX: int = p.W_GX, W_GY: int = p.W_GY, W_V: int = p.W_V,
                 W_A: int = p.W_A, W_W: int = p.W_W, W_J: int = p.W_J, W_H: int = p.W_H, W_P: int = p.W_P):
        """
        Instantiate a vehicle object from an input scenario
        Parameters:
        - csv_path (str): Path to input scenario
        - time_horizon (int): Time horizon of the optimal control problem
        - discretization_nb (int): Number of discretization time steps
        - V_RECOM (int): Recommended linear speed
        - OMEGA_MAX (float): Maximum angular speed
        - DIS_MIN (int): Minimum distance between ego and target vehicles
        - V_MAX (int): Maximum linear speed
        - W_GX (int): Weight for the distance to the next waypoint on the x-axis in the objective function
        - W_GY (int): Weight for the distance to the next waypoint on the y-axis in the objective function
        - W_V (int): Weight for the error between current linear speed and recommended linear speed in the objective function
        - W_A (int): Weight for controlling the linear acceleration in the objective function
        - W_W (int): Weight for controlling the angular speed in the objective function
        - W_J (int): Weight for controlling the jerk in the objective function
        - W_H (int): Weight for controlling the angle between the head of the vehicle and the centre lane of the road in the objective function
        - W_P (int): Weight for controlling the potential field function : headway distance between the ego and target vehicles in the objective function
        """
        self.x0: float
        self.y0: float
        self.v0: float
        self.accx0: float
        self.accy0: float
        self.theta0: float
        self.cr_x: np.ndarray
        self.cr_y: np.ndarray
        self.x_tgt: np.ndarray
        self.y_tgt: np.ndarray
        self.waypoints_x: np.ndarray
        self.waypoints_y: np.ndarray
        self.waypoints_theta: np.ndarray
        self.time_horizon: int = time_horizon
        self.discretization_nb: int = discretization_nb
        self.time: np.ndarray = np.linspace(0, time_horizon, num=discretization_nb)
        self.V_RECOM: int = V_RECOM
        self.OMEGA_MAX: float = OMEGA_MAX
        self.DIS_MIN: int = DIS_MIN
        self.V_MAX: int = V_MAX
        self.W_GX: int = W_GX
        self.W_GY: int = W_GY
        self.W_V: int = W_V
        self.W_A: int = W_A
        self.W_W: int = W_W
        self.W_J: int = W_J
        self.W_H: int = W_H
        self.W_P: int = W_P
        self.csv = pd.read_csv(csv_path, sep=';')
        self.x0, self.y0, self.v0, self.accx0, self.accy0, self.theta0 = self.get_initial_state()
        self.cr_x, self.cr_y = self.get_center_path()
        self.n_dim_csv: int = p.N_DIM_CSV
        self.waypoints_x, self.waypoints_y, self.waypoints_theta = self.get_waypoints()
        self.x_tgt, self.y_tgt = self.get_target_vehicle_path()

    ###############################################################################################################################
    # Getters
    ###############################################################################################################################

    def get_initial_state(self, idx_start: int = 0) -> Tuple[float, float, float, float, float, float]:
        """
        Method to instantiate initial state of the vehicle
        Parameters:
        - idx_start (int): initial index for input scenario
        Output:
        - x0 (float): Initial coordinate x
        - y0 (float): Initial coordinate y
        - v0 (float): Initial linear speed v
        - accx0 (float): Initial acceleration along x-axis
        - accy0 (float): Initial acceleration along y-axis
        - theta0 (float): Initial angle theta
        """
        x0: float
        y0: float
        vx0: float
        vy0: float
        accx0: float
        accy0: float
        # Initial point
        x0, y0 = self.csv.loc[idx_start, c.COL_X0], self.csv.loc[idx_start, c.COL_Y0]

        # Initial speed
        vx0, vy0 = self.csv.loc[idx_start, c.COL_VX0], self.csv.loc[idx_start, c.COL_VY0]
        v0: float = np.sqrt(vx0 ** 2 + vy0 ** 2)

        # Initial acceleration
        accx0, accy0 = self.csv.loc[idx_start, c.COL_ACCX0], self.csv.loc[idx_start, c.COL_ACCY0]

        # Initial angle
        theta0: float = np.arctan(np.divide((self.csv.loc[idx_start + 1, c.COL_Y0] - self.csv.loc[idx_start, c.COL_Y0]),
                                            (self.csv.loc[idx_start + 1, c.COL_X0] - self.csv.loc[
                                                idx_start, c.COL_X0])))

        return x0, y0, v0, accx0, accy0, theta0

    def get_center_path(self, col_name: str = c.COL_MIDDLE_LANE_COORDS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute middle lane coordinates of the road
        Parameters:
        - col_name (str): Column name for middle lane coordinates in input CSV file
        Output:
        - cr_x (np.ndarray): Coordinates x for middle lane
        - cr_y (np.ndarray): Coordinates y for middle lane
        """
        str_arr: np.ndarray = np.array(self.csv[col_name])
        cr_str: List[List[str]] = [t[1:-1].split(", ") for t in str_arr]
        cr_x: np.ndarray = np.array([float(t[0]) for t in cr_str])
        cr_y: np.ndarray = np.array([float(t[1]) for t in cr_str])
        return cr_x, cr_y

    def get_waypoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to compute reference waypoints for the vehicle to follow
        Output:
        - waypoints_x (np.ndarray): Waypoints for x coordinates
        - waypoints_y (np.ndarray): Waypoints for y coordinates
        - waypoints_theta (np.ndarray): Waypoints for angle theta
        """
        # Find the initial index for the closest x position in the center line reference
        i: int = [i for i, x in enumerate(self.cr_x) if x >= self.x0][0]
        # Waypoints are sampled all along the road over the time horizon
        waypoints_x: np.ndarray = np.array(self.cr_x[i::int((self.n_dim_csv-i)/self.discretization_nb)])
        waypoints_y: np.ndarray = np.array(self.cr_y[i::int((self.n_dim_csv-i)/self.discretization_nb)])
        n_waypoints: int = len(waypoints_x)
        waypoints_theta: np.ndarray = np.array(
            [np.arctan(np.divide((waypoints_y[k + 1] - waypoints_y[k]),
                                 (waypoints_x[k + 1] - waypoints_x[k]))) for k in range(n_waypoints - 1)] + [0.]
        )
        return waypoints_x, waypoints_y, waypoints_theta

    def get_target_vehicle_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Polynomial interpolation to retrieve time series of x and y coordinates of the target vehicle
        Output:
        - x_tgt (np.poly1d): Polynomial fit for x target vehicle coordinate through time
        - y_tgt (np.poly1d): Polynomial fit for y target vehicle coordinate through time
        """
        x_tgt: np.ndarray = np.array(self.csv.loc[:self.discretization_nb-1, c.COL_TARGET_X])
        y_tgt: np.ndarray = np.array(self.csv.loc[:self.discretization_nb-1, c.COL_TARGET_Y])
        return x_tgt, y_tgt
    
    ###############################################################################################################################
    # Initialize model
    ###############################################################################################################################

    def initialize_model(self) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Array]:
        """
        Initialize GEKKO Vehicle model within the unicycle kinematic model framework
        Output:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation
        """
        m: GEKKO = GEKKO(remote=False)
        m.time = self.time

        # Initialize control variables
        z: GEKKO.Array = m.Array(m.Var, (4))  # z = [x, y, theta, v]
        u: GEKKO.Array = m.Array(m.Var, (2))  # u = [a, w]

        return m, z, u
    
    ###############################################################################################################################
    # Initialize parameters
    ###############################################################################################################################
    
    def initialize_parameters(self, m: GEKKO) -> Tuple[GEKKO.Param, GEKKO.Param, GEKKO.Param, GEKKO.Param, GEKKO.Param]:
        """
        Instantiate parameters in  GEKKO format
        Output:
        - x_tgt_param (GEKKO.Param): time-series of coordinates x of target vehicle as GEKKO Parameter
        - y_tgt_param (GEKKO.Param): time-series of coordinates y of target vehicle as GEKKO Parameter
        - waypoints_x_param (GEKKO.Param): time-series of coordinates x of waypoints as GEKKO Parameter
        - waypoints_y_param (GEKKO.Param): time-series of coordinates y of waypoints as GEKKO Parameter
        - waypoints_theta_param (GEKKO.Param): time-series of coordinates theta of waypoints as GEKKO Parameter
        """
        # Initialize parameters
        x_tgt_param: GEKKO.Param = m.Param(value=self.x_tgt)
        y_tgt_param: GEKKO.Param = m.Param(value=self.y_tgt)
        waypoints_x_param: GEKKO.Param = m.Param(value=self.waypoints_x)
        waypoints_y_param: GEKKO.Param = m.Param(value=self.waypoints_y)
        waypoints_theta_param: GEKKO.Param = m.Param(value=self.waypoints_theta)
        
        return x_tgt_param, y_tgt_param, waypoints_x_param, waypoints_y_param, waypoints_theta_param

    ###############################################################################################################################
    # Define constraints
    ###############################################################################################################################
    
    def L_Constraint(self, current_param: str) -> Tuple[float, float, float]:
        """
        Function to maintain the vehicle on the road
        Parameters:
        - current_param (str): current bound to retrieve (x, y or theta)
        Output:
        - init (float): initial parameter (x0, y0 or theta0)
        - min (float): lower bound of the input parameter
        - max (float): upper bound of the input parameter
        """
        init: float
        if current_param == "x":
            init = self.x0
            return init, self.cr_x[0] - 100, self.cr_x[-1] + 100
        elif current_param == "y":
            init = self.y0
            min_y_road: float = min(self.cr_y) - p.ROAD_WIDTH_MAX
            max_y_road: float = max(self.cr_y) + p.ROAD_WIDTH_MAX
            return init, min_y_road, max_y_road
        elif current_param == "theta":
            init = self.theta0
            return init, -np.pi/2, np.pi/2
    
    def define_constraints(
            self, m: GEKKO, z: GEKKO.Array, u: GEKKO.Array) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Array]:
        """
        Instantiate GEKKO constraints
        Parameters:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation
        - x_tgt_param (GEKKO.Param): position x of target vehicle
        Output:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation with constraints defined
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation with constraints defined
        """
        # Initial coordinate x
        z[0].value = self.x0

        # Initial coordinate y
        z[1].value = self.y0

        # Initial angle theta
        z[2].value = self.theta0

        # State variable constraints (a)
        m.Equation(z[0].dt() == z[3] * m.cos(z[2]))
        m.Equation(z[1].dt() == z[3] * m.sin(z[2]))
        m.Equation(z[2].dt() == u[1])
        m.Equation(z[3].dt() == u[0])

        # Initialize road constraints (b)
        for i, el in enumerate(["x", "y"]):
            z[i].value, z[i].lower, z[i].upper = self.L_Constraint(el)

        # Initialize linear speed constraints (c)
        z[3].value = self.v0
        z[3].lower = 0  # Can't go backward
        z[3].upper = self.V_MAX

        # Angular speed constraints (d)
        u[1].value = 0
        u[1].lower = -self.OMEGA_MAX
        u[1].upper = self.OMEGA_MAX

        # Linear acceleration constraints (e)
        u[0].value = 0
        u[0].lower = -p.A_MAX
        u[0].upper = p.A_MAX

        # Jerk constraints (f)
        u[0].dt().lower = -p.J_MAX
        u[0].dt().upper = p.J_MAX

        return m, z, u

    ###############################################################################################################################
    # Initialize objective function
    ###############################################################################################################################
    
    def D_Distance(self, x: GEKKO.Var, y: GEKKO.Var, waypoints_x: GEKKO.Param, waypoints_y: GEKKO.Param) -> GEKKO.Var:
        """
        D-Distance for the computation of objective function on distance to waypoint
        Parameters:
        - x (GEKKO.Var): position x of ego vehicle
        - y (GEKKO.Var): position y of ego vehicle
        - waypoints_x (GEKKO.Param) : waypoints of position x
        - waypoints_y (GEKKO.Param) : waypoints of position y
        Output:
        Pondered L2-Norm Distance between current position of ego vehicle and waypoints
        """
        return self.W_GX*(x-waypoints_x)**2 + self.W_GY*(y-waypoints_y)**2

    def H_Distance(self, theta: GEKKO.Var, waypoints_theta: GEKKO.Param) -> GEKKO.Var:
        """
        H-Distance for the computation of objective function on angular theta
        Parameters:
        - theta (float): angular of the ego vehicle
        Output:
        Euclidean distance between angle theta of ego vehicle and angle of waypoint
        """
        return self.W_H*(theta-waypoints_theta)**2

    def P_Distance(self, m: GEKKO, x: GEKKO.Var, y: GEKKO.Var, x_tgt: GEKKO.Param, y_tgt: GEKKO.Param) -> GEKKO.Var:
        """
        P-Distance for potential field between the two vehicles
        Parameters:
        - x_tgt (GEKKO.Param): position x of target vehicle
        - y_tgt (GEKKO.Param): position y of target vehicle
        - x (GEKKO.Var): position x of ego vehicle
        - y (GEKKO.Var): position y of ego vehicle
        Output:
        Potential field distance between both vehicles
        """
        dis: GEKKO.Var = m.sqrt((x_tgt-x)**2+(y_tgt-y)**2) * m.sign2((x_tgt-x))
        return self.W_P * m.exp(-((x_tgt-x)**2+(y_tgt-y)**2)/2) * (1+m.erf(dis/m.sqrt(2)))
    
    def define_objective_function(self, m: GEKKO, z: GEKKO.Array, u: GEKKO.Array,
                                  waypoints_x_param: GEKKO.Param, waypoints_y_param: GEKKO.Param, waypoints_theta_param: GEKKO.Param,
                                  x_tgt_param: GEKKO.Param, y_tgt_param: GEKKO.Param) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Array, GEKKO.Var]:
        """
        Definition of objective function
        Parameters:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation
        - waypoints_x (GEKKO.Param) : waypoints of position x
        - waypoints_y (GEKKO.Param) : waypoints of position y
        - waypoints_theta (GEKKO.Param) : waypoints of angle theta
        - x_tgt_param (GEKKO.Param): position x of target vehicle
        - y_tgt_param (GEKKO.Param): position y of target vehicle
        Output:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation with objective function defined
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation with objective function defined
        - obj (GEKKO.Var): GEKKO objective function value
        """
        # Objective function Dt
        obj1: GEKKO.Var = self.D_Distance(x=z[0], y=z[1], waypoints_x=waypoints_x_param, waypoints_y=waypoints_y_param)
        # Objective function linear speed
        obj2: GEKKO.Var = self.W_V*(self.V_RECOM-z[3])**2
        # Objective function for linear acceleration
        obj3: GEKKO.Var = self.W_A*u[0]**2
        # Objective function for angular speed
        obj4: GEKKO.Var = self.W_W*u[1]**2
        # Objective function for jerk
        obj5: GEKKO.Var = self.W_J*(u[1].dt())**2
        # Objective function for angle
        obj6: GEKKO.Var = self.H_Distance(theta=z[2], waypoints_theta=waypoints_theta_param)
        # Objective function for potential field P
        obj7: GEKKO.Var = self.P_Distance(m=m, x=z[0], y=z[1], x_tgt=x_tgt_param, y_tgt=y_tgt_param)

        # Objective function
        obj: GEKKO.Var = m.Var(0)
        m.Equation(obj.dt() == obj1+obj2+obj3+obj4+obj5+obj6+obj7)

        return m, z, u, obj

    ###############################################################################################################################
    # Compute constraint violations
    ###############################################################################################################################
    
    def constraint_violations(self, m: GEKKO, z: GEKKO.Array) -> Tuple[GEKKO, GEKKO.Intermediate]:
        """
        Compute the violatd constraints. If positive, the constraint is violated
        Parameters:
        - m (GEKKO): Current GEKKO model
        - z (GEKKO.Array): GEKKO control state variable
        Output:
        - m (GEKKO): Current GEKKO model
        - violated_constraints_param (GEKKO.Intermediate): GEKKO Intermediate variable of violated constraints
        """
        # Compute the violated constraint
        # if positive, the constraint is violated
        x_target_real: np.ndarray = np.array(self.csv.loc[:self.discretization_nb-1, c.COL_REAL_X])
        y_target_real: np.ndarray = np.array(self.csv.loc[:self.discretization_nb-1, c.COL_REAL_Y])
        constraint_dis_min: np.ndarray = np.array([self.DIS_MIN] * self.discretization_nb)

        # Conversion to GEKKO Param format
        x_real_param: GEKKO.Param = m.Param(value=x_target_real)
        y_real_param: GEKKO.Param = m.Param(value=y_target_real)
        constraint_dis_min_param: GEKKO.Param = m.Param(value=constraint_dis_min)

        # Define violated constraints
        violated_constraints_param: GEKKO.Intermediate = m.Intermediate(constraint_dis_min_param - m.abs(z[0] - x_real_param + z[1] - y_real_param))
        return m, violated_constraints_param

class DeterministicVehicle(Vehicle):
    """
    Initialize a GEKKO Vehicle model from an input scenario
    """

    def __init__(self, csv_path: str, time_horizon: int = p.T, discretization_nb: int = p.N_DIM,
                 V_RECOM: int = p.V_RECOM, OMEGA_MAX: float = p.OMEGA_MAX, DIS_MIN: int = p.DIS_MIN,
                 V_MAX: int = p.V_MAX, W_GX: int = p.W_GX, W_GY: int = p.W_GY, W_V: int = p.W_V,
                 W_A: int = p.W_A, W_W: int = p.W_W, W_J: int = p.W_J, W_H: int = p.W_H, W_P: int = p.W_P):
        super().__init__(csv_path=csv_path, time_horizon=time_horizon, discretization_nb=discretization_nb,
                         V_RECOM=V_RECOM, OMEGA_MAX=OMEGA_MAX, DIS_MIN=DIS_MIN, V_MAX=V_MAX,
                         W_GX=W_GX, W_GY=W_GY, W_V=W_V, W_A=W_A, W_W=W_W, W_J=W_J, W_H=W_H, W_P=W_P)
    
    ###############################################################################################################################
    # Define deterministic constraint
    ###############################################################################################################################

    def K_Distance(self, m: GEKKO, x: GEKKO.Var, y: GEKKO.Var, x_tgt: GEKKO.Param, y_tgt: GEKKO.Param) -> GEKKO.Var:
        """
        K-Distance for the computation of constraint (2g) on the minimum distance between ego and target vehicles
        Parameters:
        - x (GEKKO.Var): position x of ego vehicle
        - x_tgt (GEKKO.Param): position x of target vehicle
        Output:
        Euclidean distance between position x of target and ego vehicles 
        """
        return m.abs(x_tgt - x + y_tgt - y)
    
    def define_constraints(
            self, m: GEKKO, z: GEKKO.Array, u: GEKKO.Array, x_tgt_param: GEKKO.Param, y_tgt_param: GEKKO.Param) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Array]:
        """
        Overload define_constraints method from Vehicle class to add deterministic GEKKO constraints
        Parameters:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation
        - x_tgt_param (GEKKO.Param): position x of target vehicle
        - x_tgt_param (GEKKO.Param): position y of target vehicle
        Output:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation with constraints defined
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation with constraints defined
        """
        m, z, u = super().define_constraints(m, z, u)

        # Constraint K distance between ego and target vehicles (g)
        m.Equation(self.K_Distance(m=m, x=z[0], y=z[1], x_tgt=x_tgt_param, y_tgt=y_tgt_param) >= self.DIS_MIN)

        return m, z, u
        

    ###############################################################################################################################
    # Solve deterministic model
    ###############################################################################################################################

    def solve_deterministic_model(self, display: bool = False) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Intermediate, GEKKO.Array, GEKKO.Var]:
        """
        Deterministic model for continuous-time vehicle
        Parameters:
        - display (bool): GEKKO parameter for verbose solving
        Output:
        - m (GEKKO): Current GEKKO Model
        - z (GEKKO.Array): GEKKO state variable
        - violated_constraints (GEKKO.Intermediate): Array of violated constraints
        - u (GEKKO.Array): GEKKO control variable
        - obj (GEKKO.Var): GEKKO objective function value
        """
        m: GEKKO
        z: GEKKO.Array
        u: GEKKO.Array
        m, z, u = self.initialize_model()
        x_tgt_param, y_tgt_param, waypoints_x_param, waypoints_y_param, waypoints_theta_param = self.initialize_parameters(m=m)
        m, z, u = self.define_constraints(m=m, z=z, u=u, x_tgt_param=x_tgt_param, y_tgt_param=y_tgt_param)

        m, z, u, obj = self.define_objective_function(m=m, z=z, u=u,
            waypoints_x_param=waypoints_x_param, waypoints_y_param=waypoints_y_param, waypoints_theta_param=waypoints_theta_param,
            x_tgt_param=x_tgt_param, y_tgt_param=y_tgt_param)

        # Final time
        time_param: np.ndarray = np.zeros(self.discretization_nb)
        time_param[-1] = 1.0
        final: GEKKO.Param = m.Param(value=time_param)

        # Violated constraints GEKKO Intermediate Variable
        violated_constraints: GEKKO.Intermediate
        m, violated_constraints = self.constraint_violations(m=m, z=z)

        # params Gekko
        m.Obj(obj*final)
        # Default parameters for acceleration and tolerance of the solver
        m.options.IMODE = 6  # Type of solver used
        m.options.SOLVER = 3
        m.options.NODES = 3
        m.options.MAX_ITER = 2000
        m.options.OTOL = 1.0e-9
        # Solve
        m.solve(disp=display)

        # Convert GEKKO object format to np.ndarray
        violated_constraints_flatten: np.ndarray = np.array(violated_constraints.value.value)

        return m, z, u, obj, violated_constraints_flatten

class StochasticVehicle(Vehicle):
    """
    Class for implementing a stochastic model within the vehicle framework
    """

    ###############################################################################################################################
    # Initialize model
    ###############################################################################################################################

    def __init__(self, csv_path: str, time_horizon: int = p.T, discretization_nb: int = p.N_DIM,
                 V_RECOM: int = p.V_RECOM, OMEGA_MAX: float = p.OMEGA_MAX, DIS_MIN: int = p.DIS_MIN,
                 V_MAX: int = p.V_MAX, W_GX: int = p.W_GX, W_GY: int = p.W_GY, W_V: int = p.W_V,
                 W_A: int = p.W_A, W_W: int = p.W_W, W_J: int = p.W_J, W_H: int = p.W_H, W_P: int = p.W_P):
        super().__init__(csv_path=csv_path, time_horizon=time_horizon, discretization_nb=discretization_nb,
                         V_RECOM=V_RECOM, OMEGA_MAX=OMEGA_MAX, DIS_MIN=DIS_MIN, V_MAX=V_MAX,
                         W_GX=W_GX, W_GY=W_GY, W_V=W_V, W_A=W_A, W_W=W_W, W_J=W_J, W_H=W_H, W_P=W_P)
        self.mu_x_tgt: np.ndarray
        self.std_x_tgt: np.ndarray
        self.mu_y_tgt: np.ndarray
        self.std_y_tgt: np.ndarray
        self.mu_x_tgt, self.std_x_tgt, self.mu_y_tgt, self.std_y_tgt = self.mean_std_targets()

    def mean_std_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract mean and standard deviation from x and y time-series of target vehicle
        Output:
        - mu_x_tgt (GEKKO.Array): mean of x coordinates of target vehicle
        - std_x_tgt (GEKKO.Array): standard deviation of x coordinates of target vehicle
        - mu_y_tgt (GEKKO.Array): mean of y coordinates of target vehicle
        - std_y_tgt (GEKKO.Array): standard deviation of y coordinates of target vehicle
        """
        std_x_tgt: np.ndarray # Standard deviations for coordinate x as parameter
        std_y_tgt: np.ndarray # Standard deviations for coordinate y as parameter
        mu_x_tgt: np.ndarray = self.x_tgt
        mu_y_tgt: np.ndarray = self.y_tgt
        std_x_tgt = p.STD_X_TGTS
        std_y_tgt = p.STD_Y_TGTS
        return mu_x_tgt, std_x_tgt, mu_y_tgt, std_y_tgt
    
    ###############################################################################################################################
    # Initialize constraints
    ###############################################################################################################################

    def define_constraints(
            self, m: GEKKO, z: GEKKO.Array, u: GEKKO.Array) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Array]:
        """
        Instantiate GEKKO constraints
        Parameters:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation
        Output:
        - m (GEKKO): GEKKO Object instance
        - z (GEKKO.Array): GEKKO Array of GEKKO Variables for state variables representation with constraints defined
        - u (GEKKO.Array): GEKKO Array of GEKKO Variables for control variables representation with constraints defined
        """
        m, z, u = super().define_constraints(m, z, u)
        
        ### Define stochastic constraint
        stds: np.ndarray = np.sqrt(self.std_x_tgt**2 + self.std_y_tgt**2)
        sum_means: np.ndarray = self.mu_x_tgt + self.mu_y_tgt
        stochastic_bnd: np.ndarray = sum_means - self.DIS_MIN + norm.ppf(1-p.ALPHA) * stds
        stochastic_bnd_param: GEKKO.Param = m.Param(value=stochastic_bnd)
        m.Equation(z[0] + z[1] <= stochastic_bnd_param)

        return m, z, u  

    ###############################################################################################################################
    # Solve stochastic model
    ###############################################################################################################################
    
    def solve_stochastic_model(self, display: bool = False) -> Tuple[GEKKO, GEKKO.Array, GEKKO.Intermediate, GEKKO.Array, GEKKO.Var]:
        """
        Stochastic model for continuous-time vehicle
        Parameters:
        - display (bool): GEKKO parameter for displaying solver information
        Output:
        - m (GEKKO): Current GEKKO Model
        - z (GEKKO.Array): GEKKO state variable
        - violated_constraints (GEKKO.Intermediate): GEKKO Intermediate variable to spot violated constraints
        - u (GEKKO.Array): GEKKO control variable
        - obj (GEKKO.Var): GEKKO objective function value
        """
        m: GEKKO
        z: GEKKO.Array
        u: GEKKO.Array
        m, z, u = self.initialize_model()
        x_tgt_param, y_tgt_param, waypoints_x_param, waypoints_y_param, waypoints_theta_param = self.initialize_parameters(m=m)
        m, z, u = self.define_constraints(m=m, z=z, u=u, x_tgt_param=x_tgt_param, y_tgt_param=y_tgt_param)
        m, z, u, obj = self.define_objective_function(m=m, z=z, u=u,
            waypoints_x_param=waypoints_x_param, waypoints_y_param=waypoints_y_param, waypoints_theta_param=waypoints_theta_param,
            x_tgt_param=x_tgt_param, y_tgt_param=y_tgt_param)

        # Violated constraints Intermediate GEKKO Variable
        violated_constraints: GEKKO.Intermediate
        m, violated_constraints = self.constraint_violations(m=m, z=z)
        
        # Final time
        time_param: np.ndarray = np.zeros(self.discretization_nb)
        time_param[-1] = 1.0
        final: GEKKO.Param = m.Param(value=time_param)

        # params Gekko
        m.Obj(obj*final)
        # Default parameters acceleration, tolerance of the solver
        m.options.IMODE = 6  # Type of solver used
        m.options.SOLVER = 3
        m.options.NODES = 3
        m.options.MAX_ITER = 2000
        m.options.OTOL = 1.0e-9
        # Solve
        m.solve(disp=display)

        # Convert constraints format for further analysis
        violated_constraints_flatten: np.ndarray = np.array(violated_constraints.value.value)

        return m, z, u, obj, violated_constraints_flatten