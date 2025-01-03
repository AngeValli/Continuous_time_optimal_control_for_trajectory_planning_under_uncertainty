from typing import List
import os

import time
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parameters as p
import constants as c
from vehicle_models import DeterministicVehicle, StochasticVehicle

np.random.seed(0)

def generate_scenarios(n_csv, folder_name = "self_generated_data"):
    """
    Generate toy scenarios
    Parameters:
    n_csv (int): Number of csv file to generate
    folder_name (str): Folder name to save the generated files
    Output:
    Generation of n_csv files
    """
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    else:
      nb_extra_scenarios: int = n_csv
      name: str = folder_name + "/" + str(nb_extra_scenarios) + ".csv"
      while os.path.isfile(name):
        os.remove(name)
        nb_extra_scenarios += 1
        name = folder_name + "/" + str(nb_extra_scenarios) + ".csv"

    
    for i in range(n_csv):
        name = folder_name + "/" + str(i) + ".csv"
        v_recom_tgt: float = p.V_RECOM*99/100 + np.random.normal(scale = 1) # Recommended speed a bit under the real recommended speed
        v_recom: float = p.V_RECOM*99/100 + np.random.normal(scale = 1)
        initial_distance: int = p.DIS_MIN+3 # Initial distance between vehicles, a bit higher than the minimum distance

        # For those variables, we only need the first element
        X0: List[float] = [0] + [1] * (p.N_DIM_CSV - 1)  # for first angle theta0
        Y0: List[float] = [0] * p.N_DIM_CSV

        VX0: List[float] = [v_recom] * p.N_DIM_CSV
        VY0: List[float] = [0] * p.N_DIM_CSV

        ACCX0: List[float] = [0] * p.N_DIM_CSV
        ACCY0: List[float] = [0] * p.N_DIM_CSV

        # Middle line reference
        MIDDLE_LANE: List[List[float]] = []
        x_tmp: int = 0
        for _ in range(p.N_DIM_CSV):
            MIDDLE_LANE.append([x_tmp, 0])
            x_tmp += v_recom * p.T/p.N_DIM

        # Generate trajectory of target vehicle x_tgt, y_tgt
        X1 = [initial_distance + p.T/p.N_DIM * v_recom_tgt * idx for idx in range(p.N_DIM_CSV)] # add stochastic part
        Y1 = [0] * p.N_DIM_CSV 

        d = {c.COL_X0: X0, c.COL_Y0: Y0, c.COL_VX0: VX0, c.COL_VY0: VY0, c.COL_ACCX0: ACCX0, c.COL_ACCY0: ACCY0,
            c.COL_MIDDLE_LANE_COORDS: MIDDLE_LANE, c.COL_TARGET_X: X1, c.COL_TARGET_Y: Y1}
        df = pd.DataFrame(data=d)
        df.to_csv("generated_0.csv", sep=";")

        # CSV for stochastic
        straight_csv = pd.read_csv("generated_0.csv", sep=";")

        # add uncertainty
        sigma_target: float = 1

        straight_csv[c.COL_REAL_X] = straight_csv[c.COL_TARGET_X]
        straight_csv[c.COL_REAL_X] = straight_csv[c.COL_REAL_X] + [np.random.normal(scale = sigma_target) 
                                         for _ in range(len(straight_csv[c.COL_REAL_X]))]
        straight_csv[c.COL_REAL_Y] = straight_csv[c.COL_TARGET_Y]
        straight_csv[c.COL_REAL_Y] = straight_csv[c.COL_REAL_Y] + [np.random.normal(scale = sigma_target) 
                                         for _ in range(len(straight_csv[c.COL_REAL_Y]))]
        print(name)
        straight_csv.to_csv(name, sep = ";")

if __name__ == "__main__":
    start_time = time.time()
    n_csv: int = 100 # Generate 100 scenarios
    violations: List[List[float]] = []
    count_failed: int = 0
    violations_sto: List[List[float]] = []
    count_failed_sto: int = 0
    generate_scenarios(n_csv=n_csv)
    for i in range(n_csv):
        current_scenario: str = "self_generated_data/" + str(i) + ".csv"
        continuous_deterministic_vehicle = DeterministicVehicle(csv_path=current_scenario, time_horizon=p.T, discretization_nb=p.N_DIM, V_RECOM=p.V_RECOM, OMEGA_MAX=p.OMEGA_MAX)
        continuous_stochastic_vehicle = StochasticVehicle(csv_path=current_scenario, time_horizon=p.T, discretization_nb=p.N_DIM, V_RECOM=p.V_RECOM, OMEGA_MAX=p.OMEGA_MAX)
        print("solving " + str(i) + " ...")
        try:
            current_GEKKO_model_sto, z_sto, u_sto, obj_sto, violated_constraints_sto = continuous_stochastic_vehicle.solve_stochastic_model()
            print("Objective function value stochastic: " + str(obj_sto[-1]))
            violations_sto.append(violated_constraints_sto.tolist())
        except:
            print("Stochastic " + str(i) + " no result")
            count_failed_sto += 1
        try:
            current_GEKKO_model, z, u, obj, violated_constraints = continuous_deterministic_vehicle.solve_deterministic_model()
            print("Objective function value deterministic: " + str(obj[-1]))
            violations.append(violated_constraints.tolist())
        except:
            print("Deterministic " + str(i) + " no result")
            count_failed += 1
    
    print("Number of failed stochastic scenarios: " + str(count_failed_sto) + " over " + str(n_csv) + " scenarios in total")
    print("Number of failed deterministic scenarios: " + str(count_failed) + " over " + str(n_csv) + " scenarios in total")
    print("Time elapsed: " + str(time.time() - start_time) + " seconds")
    
    rc("figure", facecolor="white")

    fig, ax = plt.subplots(2, 1, figsize=(16, 19), sharex=True, constrained_layout=True)

    ax[0].axhline(y=0, color="black", linestyle="-")
    ax[1].axhline(y=0, color="black", linestyle="-")
    ax[0].set_title("Continuous stochastic model scenarios")
    ax[1].set_title("Continuous deterministic model scenarios")
    ax[0].set_ylabel("Constrained distance to target vehicle")
    ax[1].set_ylabel("Constrained distance to target vehicle")
    ax[0].set_xlabel("Time horizon")
    ax[1].set_xlabel("Time horizon")
    ax[0].set_xticks(range(0, p.N_DIM, p.N_DIM-1))
    ax[1].set_xticks(range(0, p.N_DIM, p.N_DIM-1))

    for k in range(len(violations_sto)):
        c: float = np.random.rand(3)
        ax[0].plot(violations_sto[k], color=c, alpha = 0.7)
    for k in range(len(violations)):
        c = np.random.rand(3)
        ax[1].plot(violations[k], color=c, alpha = 0.7)
    
    plt.savefig("continuous_model.png")