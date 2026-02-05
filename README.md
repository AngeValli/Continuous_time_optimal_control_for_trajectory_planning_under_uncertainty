# Trajectory_planning_under_uncertainty

Python code for implementing a continuous-time reference trajectory generator for autonomous vehicles. This code follows the article _Continuous-time optimal control for trajectory planning under uncertainty_

This project is composed of the following Python scripts :

* _constants.py_ : Column names for the generation of csv files of the different toy urban driving scenarios
* _parameters.py_ : Parameters for generating urban driving scenarios and the reference trajectory generator
* _vehicle_models.py_ : Classes of vehicles implementing deterministic and stochastic models in continuous-time within [GEKKO](https://gekko.readthedocs.io/en/latest/) framework.
    * _Vehicle_ : Parent class for initialising a GEKKO object with an input scenario. It also contains the methods to define the constraints, the objective function and to analyse constraints violations after solving.
    * _DeterministicVehicle_ : Child class of _Vehicle_ with deterministic constraint and the method to solve the problem and compute the constraints violations.
    * _StochasticVehicle_ : Child class of _Vehicle_ with stochastic constraint and the method to solve the problem and compute the constraints violations.
* _main.py_ : Main script to execute to generate toy scenarios and solve the trajectory planning for both deterministic and stochastic vehicles.

To run the project, run the script _main.py_. It will generate the initial scenario CSV file _generated_0.csv_ and create the other scenarios with stochastic component in the folder _self_generated_data_ with a hundred of different toy scenarios. Then, the script solves the optimal control problem for all scenarios with both deterministic and stochastic models, and compare the constraints violations between the those two. A graphic is generated and saved as _continuous_model.png_

## Bibliography

Beal, L.D.R., Hill, D., Martin, R.A., and Hedengren, J. D., GEKKO Optimization Suite, Processes, Volume 6, Number 8, 2018, doi: 10.3390/pr6080106.
Valli, A., Zhang, S., & Lisser, A. (2025). Continuous-time optimal control for trajectory planning under uncertainty. International Journal of Vehicle Autonomous Systems, 18(3), 261-286.
