import numpy as np

# Grid & Simulation Space
GRID_SIZE_X = 200  # mm
GRID_SIZE_Y = 200  # mm
RESOLUTION = 1.0   # mm per pixel
GRID_SHAPE = (int(GRID_SIZE_Y / RESOLUTION), int(GRID_SIZE_X / RESOLUTION))

# Microwave Physics Parameters
FREQ_HZ = 10e9     # Operating Frequency: 10 GHz (X-band)
Z0 = 377.0         # Impedance of free space (ohms)
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854e-12

# Baseline Material (e.g., FR4 substrate or generic dielectric)
MATERIAL_EPS_R = 4.4    # Relative permittivity
MATERIAL_SIGMA = 0.01   # Conductivity (S/m)

# Defect Material (e.g., Air void/crack)
DEFECT_EPS_R = 1.0
DEFECT_SIGMA = 0.0

# Noise parameters
NOISE_STD_DEV = 0.015   # Standard deviation of Gaussian noise added to Gamma magnitude

# Image Processing Parameters
BLUR_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)
