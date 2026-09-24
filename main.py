import os
import cv2
import numpy as np
import config
from defect_generator import ComponentSurface
from microwave_model import get_microwave_response, calculate_impedance, calculate_reflection_coefficient
from signal_processing import add_awgn_noise, normalize_response, apply_gaussian_filter
from image_processing import process_heatmap, extract_contours, calculate_defect_properties
from evaluation import evaluate_detection
from visualization import plot_results, plot_frequency_sweep

def run_simulation(case_name, defect_type, defect_params):
    print(f"--- Running Simulation: {case_name} ---")
    
    # 1. Define component surface & baseline parameters
    comp = ComponentSurface()
    
    # 2. Introduce defect
    true_area = 0
    true_centroid = (0, 0)
    
    if defect_type == 'rectangular':
        x, y, w, h = defect_params['x'], defect_params['y'], defect_params['w'], defect_params['h']
        severity = defect_params.get('severity', 1.0)
        comp.add_rectangular_defect(x, y, w, h, severity)
        true_area = w * h / (config.RESOLUTION ** 2)
        true_centroid = (x / config.RESOLUTION + w / (2*config.RESOLUTION), 
                         y / config.RESOLUTION + h / (2*config.RESOLUTION))
                         
    elif defect_type == 'circular':
        cx, cy, r = defect_params['cx'], defect_params['cy'], defect_params['r']
        severity = defect_params.get('severity', 1.0)
        comp.add_circular_defect(cx, cy, r, severity)
        true_area = np.pi * (r / config.RESOLUTION)**2
        true_centroid = (cx / config.RESOLUTION, cy / config.RESOLUTION)

    # 3. Calculate baseline and simulated reflection response (Microwave physics)
    raw_response = get_microwave_response(comp.eps_r, comp.sigma, config.FREQ_HZ)
    
    # 4. Add Noise (Signal Processing)
    noisy_response = add_awgn_noise(raw_response)
    
    # 5. Filter & Normalize (Signal Processing)
    filtered_response = apply_gaussian_filter(noisy_response)
    normalized_heatmap = normalize_response(filtered_response)
    
    # 6. Image Processing
    binary_mask = process_heatmap(normalized_heatmap)
    contours = extract_contours(binary_mask)
    
    # 7. Calculate properties
    defects_info, _ = calculate_defect_properties(contours, comp.shape)
    
    # 8. Evaluate
    metrics = evaluate_detection(comp.ground_truth, binary_mask, defects_info, 
                                 true_centroid=true_centroid, true_area=true_area)
    
    print("Detected Defects:", len(defects_info))
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # 9. Visualize
    os.makedirs('results', exist_ok=True)
    filename = f"results/dashboard_{case_name}.png"
    plot_results(comp.eps_r, raw_response, noisy_response, filtered_response,
                 normalized_heatmap, binary_mask, contours, comp.ground_truth,
                 filename=filename)
    print(f"Saved results to {filename}\n")
    return metrics

def run_frequency_sweep():
    print("--- Running Frequency Sweep ---")
    comp = ComponentSurface()
    comp.add_circular_defect(100, 100, 20, severity=1.0)
    
    freqs = np.linspace(1e9, 20e9, 20) # 1 GHz to 20 GHz
    responses = []
    
    center_idx = int(100 / config.RESOLUTION)
    
    for f in freqs:
        Z_mat = calculate_impedance(comp.eps_r, comp.sigma, f)
        Gamma = calculate_reflection_coefficient(Z_mat)
        mag = np.abs(Gamma[center_idx, center_idx])
        responses.append(mag)
        
    os.makedirs('results', exist_ok=True)
    plot_frequency_sweep(freqs, responses, "results/freq_sweep.png")
    print("Saved frequency sweep to results/freq_sweep.png\n")

if __name__ == "__main__":
    print("Starting Microwave NDT Simulation Project...\n")
    
    # Case 1: Large clear rectangular crack
    run_simulation("test_case_1", "rectangular", {'x': 50, 'y': 50, 'w': 30, 'h': 10})
    
    # Case 2: Small circular void
    run_simulation("test_case_2", "circular", {'cx': 150, 'cy': 120, 'r': 15})
    
    # Case 3: Partial severity crack (e.g., subsurface or partial void)
    run_simulation("test_case_3", "rectangular", {'x': 80, 'y': 150, 'w': 40, 'h': 20, 'severity': 0.5})
    
    # Frequency Sweep Demonstration
    run_frequency_sweep()
    
    print("All simulations completed successfully.")
