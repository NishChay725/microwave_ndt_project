import numpy as np

def calculate_iou(ground_truth_mask, detected_mask):
    """
    Calculates Intersection over Union (IoU) to measure the overlap
    between the ground truth defect mask and the detected binary mask.
    """
    # Ensure binary format (0 and 1)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    det = (detected_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(gt, det).sum()
    union = np.logical_or(gt, det).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    iou = intersection / union
    return iou

def evaluate_detection(ground_truth_mask, detected_mask, defects_info, true_centroid=None, true_area=None):
    """
    Computes performance metrics of the detection algorithm.
    """
    iou = calculate_iou(ground_truth_mask, detected_mask)
    
    detected_area = sum(d['area_pixels'] for d in defects_info)
    
    metrics = {
        'IoU': iou,
        'Detected_Area': detected_area
    }
    
    if true_area is not None and true_area > 0:
        area_error_percentage = abs(detected_area - true_area) / true_area * 100
        metrics['Area_Error_Percent'] = area_error_percentage
        
    if true_centroid is not None and len(defects_info) > 0:
        # Assuming single defect for simplicity of error metric
        main_defect = max(defects_info, key=lambda x: x['area_pixels'])
        dx = main_defect['centroid_x'] - true_centroid[0]
        dy = main_defect['centroid_y'] - true_centroid[1]
        loc_error = np.sqrt(dx**2 + dy**2)
        metrics['Localization_Error_Pixels'] = loc_error
        
    return metrics
