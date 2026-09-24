import numpy as np
import config

def calculate_impedance(eps_r_grid, sigma_grid, freq=config.FREQ_HZ):
    """
    Calculates the complex intrinsic impedance of the material grid.
    
    Z = sqrt(j * w * mu / (sigma + j * w * eps))
    For non-magnetic materials, mu = mu_0.
    eps = eps_r * eps_0.
    
    Args:
        eps_r_grid: 2D numpy array of relative permittivity
        sigma_grid: 2D numpy array of conductivity
        freq: Frequency in Hz
        
    Returns:
        2D numpy array of complex impedance Z.
    """
    omega = 2 * np.pi * freq
    mu = config.MU_0
    eps = eps_r_grid * config.EPS_0
    
    # Intrinsic impedance formula
    numerator = 1j * omega * mu
    denominator = sigma_grid + 1j * omega * eps
    Z = np.sqrt(numerator / denominator)
    return Z

def calculate_reflection_coefficient(Z_material, Z0=config.Z0):
    """
    Calculates the reflection coefficient (Gamma) at the boundary 
    between free space (probe) and the material.
    
    Gamma = (Z_material - Z0) / (Z_material + Z0)
    
    Args:
        Z_material: 2D numpy array of complex material impedance.
        Z0: Impedance of the incident medium (free space, 377 ohms)
        
    Returns:
        2D numpy array of complex reflection coefficient Gamma.
    """
    Gamma = (Z_material - Z0) / (Z_material + Z0)
    return Gamma

def get_microwave_response(eps_r_grid, sigma_grid, freq=config.FREQ_HZ):
    """
    End-to-end function to get the reflection magnitude map 
    from material properties.
    """
    Z_mat = calculate_impedance(eps_r_grid, sigma_grid, freq)
    Gamma = calculate_reflection_coefficient(Z_mat)
    
    # In NDT, sensors usually measure the magnitude of the reflection coefficient
    # (or S11 magnitude), though phase is also useful. We use magnitude here.
    return np.abs(Gamma)
