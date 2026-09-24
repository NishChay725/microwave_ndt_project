import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_results(eps_r_map, raw_response, noisy_response, filtered_response, 
                 normalized_heatmap, binary_mask, contour_img, ground_truth,
                 filename="results/result_dashboard.png"):
    """
    Generates a comprehensive multi-panel dashboard for the project demo.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    # 1. Ground Truth Material Properties (Epsilon r)
    im0 = axes[0].imshow(eps_r_map, cmap='viridis')
    axes[0].set_title('1. Material Permittivity ($\epsilon_r$)')
    fig.colorbar(im0, ax=axes[0])
    
    # 2. Simulated Raw Microwave Response (Gamma magnitude)
    im1 = axes[1].imshow(raw_response, cmap='jet')
    axes[1].set_title('2. Raw RF Response ($|\Gamma|$)')
    fig.colorbar(im1, ax=axes[1])
    
    # 3. Noisy Response
    im2 = axes[2].imshow(noisy_response, cmap='jet')
    axes[2].set_title('3. Noisy RF Response')
    fig.colorbar(im2, ax=axes[2])
    
    # 4. Filtered Response (Signal Processing)
    im3 = axes[3].imshow(filtered_response, cmap='jet')
    axes[3].set_title('4. Gaussian Filtered RF Response')
    fig.colorbar(im3, ax=axes[3])
    
    # 5. Normalized Heatmap (8-bit)
    im4 = axes[4].imshow(normalized_heatmap, cmap='gray')
    axes[4].set_title('5. Normalized 8-bit Heatmap')
    fig.colorbar(im4, ax=axes[4])
    
    # 6. Binary Mask (Image Processing - Otsu + Morph)
    im5 = axes[5].imshow(binary_mask, cmap='gray')
    axes[5].set_title('6. Processed Binary Mask')
    fig.colorbar(im5, ax=axes[5])
    
    # 7. Detected Contours vs Ground Truth
    # Create an RGB image to show contours over ground truth
    overlay = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
    overlay[:,:,1] = ground_truth  # Green channel for Ground Truth
    
    # Add contours (Red channel)
    cv2.drawContours(overlay, contour_img, -1, (255, 0, 0), 2)
    axes[6].imshow(overlay)
    axes[6].set_title('7. GT (Green) vs Detected (Red)')
    
    # 8. Text Summary Panel
    axes[7].axis('off')
    axes[7].set_title('8. Evaluation Summary')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_frequency_sweep(freqs, responses, filename="results/freq_sweep.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, responses, marker='o')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Reflection Coefficient Magnitude ($|\Gamma|$)')
    plt.title('Microwave Response vs Frequency at Defect Center')
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    plt.close()
