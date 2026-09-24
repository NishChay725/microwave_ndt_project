import cv2
import numpy as np
import config

def process_heatmap(heatmap_8bit):
    """
    Takes the normalized 8-bit heatmap from signal processing 
    and applies image processing to isolate the defect.
    
    Pipeline:
    1. Gaussian Blur (for additional smoothing if needed)
    2. Otsu's Binarization (Thresholding)
    3. Morphological Operations (Opening to remove small noise dots, Closing to fill holes)
    """
    # 1. Blur
    blurred = cv2.GaussianBlur(heatmap_8bit, config.BLUR_KERNEL_SIZE, 0)
    
    # 2. Otsu's Thresholding
    # We assume the defect has a different response than the background.
    # Depending on the material, the defect reflection could be higher or lower.
    # Let's invert if the defect appears darker, but Otsu works on bimodal distributions.
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological Operations
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    # Opening (erosion followed by dilation) to remove isolated noise pixels
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Closing (dilation followed by erosion) to fill small gaps inside the defect
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

def extract_contours(binary_mask):
    """
    Finds the contours of the detected defects in the binary mask.
    """
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_defect_properties(contours, original_shape):
    """
    Calculates properties of the detected defects (area, centroid).
    """
    defects_info = []
    total_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:  # Filter out tiny spurious contours
            # Calculate centroid using image moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            defects_info.append({
                'area_pixels': area,
                'area_mm2': area * (config.RESOLUTION ** 2),
                'centroid_x': cX,
                'centroid_y': cY,
                'contour': cnt
            })
            total_area += area
            
    total_area_percentage = (total_area / (original_shape[0] * original_shape[1])) * 100
    
    return defects_info, total_area_percentage
