"""
Imports -
    Make sure yo got everything installed
    Legacy version of fenics gives us less issues during compilation
"""
from fenics import *
import numpy as np
import random
import matplotlib.pyplot as plt

# This line imports functions in the 'functions.py'
# if you do add functions in that file, make sure you add them here as well
from functions import generate_bot_list, generate_nu, plot_fig, calc_all, update_loc, generate_vel_list

"""
Paths
    This is the path of your folder in which you got the code
    Outputs are generated at the same location
    The path is constant, op_folder_name changes
"""
path = "/mnt/c/Files/modlab/code/"
op_folder_name = "/jan11/"

"""
Parameters
"""
num_microbots = 100  # number of microbots
mass = 10e-2 # mass of each microbot
x_dim = 2  # dim of each x (position) vector (2D representation hence 2)
x_range = (
    0,
    0.5,
)  # range values have to be in range of 0,1 (If microbot position is (x[0],x[1]) then 0<x[0]<0.5 )
y_range = (
    0,
    0.5,
)  # range values have to be in range of 0,1 (If microbot position is (x[0],x[1]) then 0<x[1]<0.5 )
permittivity = 8.854e-12  # freespace permittivity for calc of forces, might change this based on fluid permittivity

# Parameters for the Gaussian approximation of the delta function, search for it on wikipedia for a very clear explanation
sigma = 1e-3
s = sigma ** 2

# Parameters for mesh (no. of cells in x and y direction), higher this number, more the values fenics calculates for
nx, ny = 100, 100

# dict to store parameters, could use a class instead but I prefer using a dict
# different datatypes are stored in this dict, so be cautious while adding any new ones
param_dict = {
    "path": path,
    "op_folder_name": op_folder_name,
    "num_microbots": num_microbots,
    "x_dim": x_dim,
    "x_range": x_range,
    "y_range": y_range,
    "permittivity": permittivity,
    "sigma": sigma,
    "s": s,
    "nx": nx,
    "ny": ny,
}

"""
Main function
"""


def main():
    range_LOT = [x_range, y_range]  # List of tuples

    # Generating x and nu lists with x_i and nu_i, X contains the positions of the microbots (2 dim vec) 
    # and heading vectors of each in the same order (2 dim vec)
    # initial_vel is initial 2D velocities of microbots, 0 in this case, can change if needed
    X = generate_bot_list(
        num_microbots, x_dim, range_LOT
    )  # localize to 4 areas, then randomize
    nu = generate_nu(num_microbots, x_dim)
    vel_init = generate_vel_list(num_microbots, x_dim)

    param_dict["X"] = X  # storing the position list in the dict defined before
    param_dict["nu"] = nu  # storing the heading vector list in the dict defined before

    # Plotting initial microbot positions and saving it in respective filename
    plot_fig(X, path, op_folder_name, "robot_positions")

    # Calculating all (Phi, E, F, T) in that order for bots in list 'X'
    F_list = calc_all(X, nu, param_dict)  

    X_new, vel_new = update_loc(mass = mass, positions = X, velocities= vel_init, forces=F_list, dt = 0.5)
    
    print(X[0])
    print(X_new[0])
    print(vel_new[0])

    # Plotting updated microbot positions and saving it in respective filename
    plot_fig(X, path, op_folder_name, "robot_positions_updated")

if __name__ == "__main__":
    main()
