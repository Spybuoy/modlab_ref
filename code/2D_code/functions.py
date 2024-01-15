from fenics import *
import numpy as np
import random
import matplotlib.pyplot as plt

# Function to create position 'x' vectors with controlled ranges
# Takes in the parameters as input and generates a list x that contains 'num_of_bots' positions with 'n' dimensions each
# 'ranges' is the constraint for the positions of the microbots, the actual positions though random are contained within a certain range
def generate_bot_list(num_of_bots, n, ranges):
    x = []
    for _ in range(num_of_bots):
        sub_list = []
        for i in range(n):
            min_val, max_val = ranges[i]
            value = random.uniform(min_val, max_val)
            sub_list.append(value)
        x.append(sub_list)
    return x

# Function to generate velocity vector list
# This generates all 0's for initial list, could change this similar to the generate_bot_list fn to get values
def generate_vel_list(num_of_bots, n):
    x = [0]*n
    vel_list = []
    for _ in range(num_of_bots):
        vel_list.append(x)
    return vel_list

# Function to create heading vectors
# Takes in the parameters as input and generates a list nu that contains 'num_of_bots' heading vectors with 'n' dimensions each
# Heading vector gives direction with a mag of 1
def generate_nu(num_of_bots, n):
    nu = []
    for _ in range(num_of_bots):
        # Generate a random vector
        sub_list = [random.uniform(-1, 1) for _ in range(n)]
        # Normalize the vector to have a magnitude of 1
        magnitude = np.linalg.norm(sub_list)
        normalized_vector = [value / magnitude for value in sub_list]
        nu.append(normalized_vector)
    return nu


# Function to plot robot positions and save figure, simple function
def plot_fig(X, path, op_folder_name, title):
    # Extract the x and y coordinates of the robots
    robot_x = [pos[0] for pos in X]
    robot_y = [pos[1] for pos in X]
    # Plot the robot positions
    plt.scatter(robot_x, robot_y, c="blue", label="Robots")
    # Additional options
    plt.xlabel("mesh_x[0]")
    plt.ylabel("mesh_x[1]")
    plt.title("Bot Positions")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)  # Optional: add a grid for easier visualization
    # Save the figure
    output_path = (
        path + op_folder_name + title + ".png"
    )  # Specify the path where you want to save the image
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Clear the figure after saving to avoid resource issues if you're making lots of plots
    plt.close()
    return None


"""
First we get Phi (potential) values from which we can calc Elec field E and Forces F
Look at the equations pdf for 2D explanation
We approximate the delta function using Gaussian approximation
"""


def calc_and_save_phi(X, nu, mesh, V, V_vector, param_dict, filename):

    # Reinitializing parameters from dict
    path = param_dict["path"]
    op_folder_name = param_dict["op_folder_name"]

    # Define trial and test functions
    phi = TrialFunction(V)
    v = TestFunction(V)
    # Define the weak form of the equation
    a = -dot(grad(phi), grad(v)) * dx
    # Define the spatial coordinates
    x = SpatialCoordinate(mesh)
    # Initialize the right-hand side
    L = Constant(0) * v * dx

    # Add contributions from each microbot to the right-hand side
    for i in range(param_dict["num_microbots"]):
        # Position and heading vector for the i-th microbot
        X = param_dict["X"]
        nu = param_dict["nu"]
        x_i = X[i]
        nu_i = nu[i]

        # Define a Gaussian centered at the microbot's position
        gaussian_expr = Expression(
            "A*exp(-pow(x[0] - x0, 2)/s - pow(x[1] - y0, 2)/s)",
            degree=2,
            A=1.0 / (2 * pi * param_dict["sigma"] ** 2),
            x0=x_i[0],
            y0=x_i[1],
            s=param_dict["s"],
            domain=mesh,
        )

        # Compute the gradient of the Gaussian
        grad_gaussian = project(grad(gaussian_expr), V_vector)
        # Multiply the gradient by the heading vector to get the source term
        source_term = nu_i[0] * grad_gaussian[0] + nu_i[1] * grad_gaussian[1]
        # Add this source term to the weak form of the Poisson equation for Phi
        L += source_term * v * dx

    # Assemble the system
    A = assemble(a)
    b = assemble(L)

    # Solve the system
    phi_solution = Function(V)
    solve(A, phi_solution.vector(), b)

    pvd_output_file_ = File(path + op_folder_name + filename + "_mag.pvd")
    pvd_output_file_ << phi_solution
    return phi_solution


"""
We get Elec Field E from phi using $E = \_nabla \Phi$
"""


def calc_E_from_phi(param_dict, V, V_vector, phi_solution, filename):
    path = param_dict["path"]
    op_folder_name = param_dict["op_folder_name"]

    E = project(-grad(phi_solution), V_vector)
    pvd_output_file = File(path + op_folder_name + filename + "_vec.pvd")
    pvd_output_file << E

    # Calculate the magnitude of the electric field
    E_magnitude = sqrt(dot(E, E))
    # Project the magnitude onto the function space
    E_magnitude_func = project(E_magnitude, V)

    pvd_output_file__ = File(path + op_folder_name + filename + "_mag.pvd")
    pvd_output_file__ << E_magnitude_func

    return E, E_magnitude_func


"""
F calc
"""


def calc_F_from_E(param_dict, V, V_vector, E, filename):

    # Reinitializign parameters
    permittivity = param_dict["permittivity"]
    path = param_dict["path"]
    op_folder_name = param_dict["op_folder_name"]

    # Define the permittivity (material property)
    epsilon = Constant(permittivity)

    # Define trial and test functions for the force
    F_trial = TrialFunction(V_vector)
    w = TestFunction(V_vector)

    # Define the weak form for the force
    a_F = inner(F_trial, w) * dx

    # This is the right-hand side of the weak form equation
    L_F = epsilon * div(dot(E, E) * w) * dx

    # Create a Function object to represent the force
    F = Function(V_vector)

    # Solve the weak form equation for the force
    solve(a_F == L_F, F)

    # Calculate the magnitude of the force
    F_magnitude = sqrt(dot(F, F))

    # Project the magnitude of the force onto the function space V
    F_magnitude_func = project(F_magnitude, V)

    # Save the magnitude of the electric field
    pvd_output_file = File(path + op_folder_name + filename + "_mag.pvd")
    pvd_output_file << F_magnitude_func

    # Save the magnitude of the electric field
    pvd_output_file_ = File(path + op_folder_name + filename + "_vec.pvd")
    pvd_output_file_ << F
    return F, F_magnitude_func


def calc_all(X, nu, param_dict):

    nx = param_dict["nx"]
    ny = param_dict["ny"]  # getting no.of cells from param dict
    
    F_list = []

    mesh = UnitSquareMesh(nx, ny)  # creating the 2D mesh

    V = FunctionSpace(mesh, "P", 1)  # creating function space (non_vector)
    V_vector = VectorFunctionSpace(mesh, "P", 1)  # creating function space (vectored)
    # Search up the parameters int the documentation, especially when working with 3D meshses

    phi_solution = calc_and_save_phi(
        X, nu, mesh, V, V_vector, param_dict, "Phi"
    )  # Calc Pot Phi, function above

    E, E_mag = calc_E_from_phi(
        param_dict, V, V_vector, phi_solution, "E"
    )  # Calc Elec Field E, function above

    F, F_mag = calc_F_from_E(
        param_dict, V, V_vector, E, "F"
    )  # Calc Forces F, function above

    for each in X:
        point = (each)
        F_list.append([F(point)[0], F(point)[1]])

    return F_list

def update_loc(mass, positions, velocities, forces, dt):
    new_positions = []
    new_velocities = []

    for i in range(len(positions)):
        # Extracting current position, velocity, and force for each microbot
        position = positions[i]
        velocity = velocities[i]
        force = forces[i]

        # Calculate acceleration
        acceleration = [f / mass for f in force]

        # Update velocity
        new_velocity = [velocity[j] + acceleration[j] * dt for j in range(len(velocity))]

        # Update position
        new_position = [position[j] + new_velocity[j] * dt for j in range(len(position))]

        new_positions.append(new_position)
        new_velocities.append(new_velocity)

    return new_positions, new_velocities