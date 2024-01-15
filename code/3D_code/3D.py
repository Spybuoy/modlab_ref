from fenics import *

path = "/mnt/c/Files/modlab/code/"

"""
** Make changes to a copy, not the org**
** Only change values in BLOCK_1 **
** May change equations in BLOCK_2 **
"""

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---------------------BLOCK_1--------------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# mesh dimensions
mesh_side = 600e-6
cells_x, cells_y, cells_z = 20, 20, 20
# Define the dimensions of the bot and pads
bot_x, bot_y, bot_z = 300e-6, 200e-6, 10e-6
pad_x, pad_y, pad_z = 50e-6, 50e-6, 10e-6

# Define boundary conditions for the pads
V_pad1, V_pad2, V_pad3, V_pad4 = 4, -4, 4, -4
s = str(V_pad1) + "," + str(V_pad2) + "," + str(V_pad3) + "," + str(V_pad4)
filename = s + ".pvd"
foldername = "(" + s + ")/"
op_folder_name = "/nov30"

# Variables for neumann conditions
# neumann_bc_val = I/(A*sigma)
I = 1
A = bot_x * bot_y
sigma = 1e4


# for Force calc
# permittivity = 11.7 #Si at 300K
permittivity = 8.854e-12  # freespace


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---------------------BLOCK_1_END----------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---------------------BLOCK_2--------------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Define the center of the bot
center = Point(0, 0, 0)

# Define the center of pads
p1_center = Point(bot_x / 4.0, bot_y / 4.0, 0)
p2_center = Point(-bot_x / 4.0, bot_y / 4.0, 0)
p3_center = Point(-bot_x / 4.0, -bot_y / 4.0, 0)
p4_center = Point(bot_x / 4.0, -bot_y / 4.0, 0)
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---------------------BLOCK_2_END----------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---------------------BLOCK_3--------------------
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

outer_mesh = BoxMesh(
    Point(-mesh_side / 2.0, -mesh_side / 2.0, -mesh_side / 2.0),
    Point(mesh_side / 2.0, mesh_side / 2.0, mesh_side / 2.0),
    cells_x,
    cells_y,
    cells_z,
)

# Define separate subdomains for pads and cuboid
class Pad1(SubDomain):
    def inside(self, x, on_boundary):
        return (
            (abs(x[0] - p1_center[0]) <= pad_x / 2)
            and (abs(x[1] - p1_center[1]) <= pad_y / 2)
            and (abs(x[2] - p1_center[2]) <= pad_z / 2)
            and (x[0] > 0)
            and (x[1] > 0)
        )


class Pad2(SubDomain):
    def inside(self, x, on_boundary):
        return (
            (abs(x[0] - p2_center[0]) <= pad_x / 2)
            and (abs(x[1] - p2_center[1]) <= pad_y / 2)
            and (abs(x[2] - p2_center[2]) <= pad_z / 2)
            and (x[0] < 0)
            and (x[1] > 0)
        )


class Pad3(SubDomain):
    def inside(self, x, on_boundary):
        return (
            (abs(x[0] - p3_center[0]) <= pad_x / 2)
            and (abs(x[1] - p3_center[1]) <= pad_y / 2)
            and (abs(x[2] - p3_center[2]) <= pad_z / 2)
            and (x[0] < 0)
            and (x[1] < 0)
        )


class Pad4(SubDomain):
    def inside(self, x, on_boundary):
        return (
            (abs(x[0] - p4_center[0]) <= pad_x / 2)
            and (abs(x[1] - p4_center[1]) <= pad_y / 2)
            and (abs(x[2] - p4_center[2]) <= pad_z / 2)
            and (x[0] > 0)
            and (x[1] < 0)
        )


# Define the subdomain for the cuboid
class Cuboid(SubDomain):
    def inside(self, x, on_boundary):
        return not any(
            [
                Pad1().inside(x, on_boundary),
                Pad2().inside(x, on_boundary),
                Pad3().inside(x, on_boundary),
                Pad4().inside(x, on_boundary),
            ]
        )


# Define function space
V = FunctionSpace(outer_mesh, "P", 1)

# Define trial and test functions
phi = TrialFunction(V)
q = TestFunction(V)

# Initialize mesh function for the subdomains
markers = MeshFunction("size_t", outer_mesh, outer_mesh.topology().dim())
markers.set_all(0)


pad1 = Pad1()
pad1.mark(markers, 1)
pad2 = Pad2()
pad2.mark(markers, 2)
pad3 = Pad3()
pad3.mark(markers, 3)
pad4 = Pad4()
pad4.mark(markers, 4)

# Mark the subdomain for the cuboid
cuboid = Cuboid()
cuboid.mark(markers, 5)

# neumann boundary condition RHS value
neumann_bc_val = I / (A * sigma)
ds_bot = Measure("ds", domain=outer_mesh, subdomain_data=markers, subdomain_id=5)

# Define a new measure for integration over the bot
dx_pad1 = Measure("dx", domain=outer_mesh, subdomain_data=markers)
dx_pad2 = Measure("dx", domain=outer_mesh, subdomain_data=markers, subdomain_id=2)
dx_pad3 = Measure("dx", domain=outer_mesh, subdomain_data=markers, subdomain_id=3)
dx_pad4 = Measure("dx", domain=outer_mesh, subdomain_data=markers, subdomain_id=4)

DX = dx_pad1 + dx_pad2 + dx_pad3 + dx_pad4

# Defining boundary conditions for the pads
bc_p1 = [DirichletBC(V, Constant(V_pad1), pad1)]
bc_p2 = [DirichletBC(V, Constant(V_pad2), pad2)]
bc_p3 = [DirichletBC(V, Constant(V_pad3), pad3)]
bc_p4 = [DirichletBC(V, Constant(V_pad4), pad4)]


# Define weak forms
a_phi = inner(grad(phi), grad(q)) * DX
L_phi = (Constant(0) * q * DX) + (
    neumann_bc_val * q * ds_bot
)  # Assuming no source term in this case
# Equivalent to ∇Φ^2=0

# Solve for Phi
Phi = Function(V)
BCS = bc_p1 + bc_p2 + bc_p3 + bc_p4
solve(a_phi == L_phi, Phi, BCS)

# Define the electric field function space
V_vec = VectorFunctionSpace(outer_mesh, "P", 1)

# Define the electric field
E = Function(V_vec)

# Define trial and test functions for the EF
E_trial = TrialFunction(V_vec)
v = TestFunction(V_vec)

# Define the weak form for the electric field
a_E = inner(E_trial, v) * DX
L_E = -inner(grad(Phi), v) * DX

# Solve for the electric field
solve(a_E == L_E, E)

# Calculate the magnitude of the electric field
E_magnitude = sqrt(dot(E, E))

# Project the magnitude onto the function space
E_magnitude_func = project(E_magnitude, V)

"""
BLOCK-4
"""
# Define the permittivity (material property)
epsilon = Constant(permittivity)

# Define trial and test functions for the force
F_trial = TrialFunction(V_vec)
w = TestFunction(V_vec)

# Define the weak form for the force
a_F = inner(F_trial, w) * DX

# This is the right-hand side of the weak form equation
L_F = epsilon * div(dot(E, E) * w) * DX

# Create a Function object to represent the force
F = Function(V_vec)

# Solve the weak form equation for the force
solve(a_F == L_F, F)

# Calculate the magnitude of the force
F_magnitude = sqrt(dot(F, F))

# Project the magnitude of the force onto the function space V
F_magnitude_func = project(F_magnitude, V)

# Save the magnitude of the electric field
pvd_output_file = File(
    path + op_folder_name + "/Dim (300e-6, 200e-6, 10e-6)/F_mag/" + foldername + filename
)
pvd_output_file << F_magnitude_func

# Save the magnitude of the electric field
pvd_output_file_ = File(
    path + op_folder_name + "/Dim (300e-6, 200e-6, 10e-6)/F/" + foldername + filename
)
pvd_output_file_ << F


# Save the magnitude of the electric field
pvd_output_file = File(
    path + op_folder_name + "/Dim (300e-6, 200e-6, 10e-6)/E_mag/" + foldername + filename
)
pvd_output_file << E_magnitude_func

# Save the magnitude of the electric field
pvd_output_file_ = File(
    path + op_folder_name + "/Dim (300e-6, 200e-6, 10e-6)/E/" + foldername + filename
)
pvd_output_file_ << E


"""
BLOCK - 5
"""

# Define the position vector
r = Expression(("x[0]", "x[1]", "x[2]"), degree=1)

# Define the volume measure for integration
dx_cuboid = Measure("dx", domain=outer_mesh, subdomain_data=markers, subdomain_id=5)

# Compute the integrals for each component of the position vector
X_COM = assemble(r[0] * dx_cuboid) / assemble(1 * dx_cuboid)
Y_COM = assemble(r[1] * dx_cuboid) / assemble(1 * dx_cuboid)
Z_COM = assemble(r[2] * dx_cuboid) / assemble(1 * dx_cuboid)

# Print the computed center of mass
print(f"\nCenter of Mass (X, Y, Z): ({X_COM}, {Y_COM}, {Z_COM})\n")

# Assuming position vector from center of rotation to center of mass
position_vector = Expression(("x[0]", "x[1]", "x[2]"), degree=1)

# Define trial and test functions for the torque
T_trial = TrialFunction(V_vec)
T_test = TestFunction(V_vec)

# Define the weak form for the torque
a_T = inner(T_trial, T_test) * DX

# This is the right-hand side of the weak form equation
L_T = dot(cross(position_vector, F), T_test) * DX

# Create a Function object to represent the torque
Torque = Function(V_vec)

# Solve the weak form equation for the torque
solve(a_T == L_T, Torque)

pvd_output_file_T = File(
    path + op_folder_name + "/Dim (300e-6, 200e-6, 10e-6)/Torque/" + foldername + filename
)
pvd_output_file_T << Torque
