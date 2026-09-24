import numpy as np
import config

class ComponentSurface:
    def __init__(self):
        """
        Initializes the 2D component grid with baseline material properties.
        """
        self.shape = config.GRID_SHAPE
        self.eps_r = np.full(self.shape, config.MATERIAL_EPS_R)
        self.sigma = np.full(self.shape, config.MATERIAL_SIGMA)
        self.ground_truth = np.zeros(self.shape, dtype=np.uint8) # 1 where defect exists
        
    def add_rectangular_defect(self, x, y, width, height, severity=1.0):
        """
        Adds a rectangular defect (e.g., crack or void).
        severity (0.0 to 1.0): 1.0 means it completely replaces material with air.
        """
        # Convert physical coordinates (mm) to grid indices
        start_x = int(x / config.RESOLUTION)
        start_y = int(y / config.RESOLUTION)
        end_x = int((x + width) / config.RESOLUTION)
        end_y = int((y + height) / config.RESOLUTION)
        
        # Clip to grid boundaries
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(self.shape[1], end_x)
        end_y = min(self.shape[0], end_y)
        
        # Calculate new properties based on severity
        # severity=1.0 -> 100% DEFECT properties
        # severity=0.0 -> 100% MATERIAL properties
        new_eps_r = config.MATERIAL_EPS_R * (1 - severity) + config.DEFECT_EPS_R * severity
        new_sigma = config.MATERIAL_SIGMA * (1 - severity) + config.DEFECT_SIGMA * severity
        
        self.eps_r[start_y:end_y, start_x:end_x] = new_eps_r
        self.sigma[start_y:end_y, start_x:end_x] = new_sigma
        
        # Mark ground truth
        self.ground_truth[start_y:end_y, start_x:end_x] = 255
        
    def add_circular_defect(self, cx, cy, radius, severity=1.0):
        """
        Adds a circular defect (e.g., circular void/pit).
        """
        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]
        # Convert physical coords to indices
        cx_idx = cx / config.RESOLUTION
        cy_idx = cy / config.RESOLUTION
        r_idx = radius / config.RESOLUTION
        
        dist_from_center = np.sqrt((X - cx_idx)**2 + (Y - cy_idx)**2)
        mask = dist_from_center <= r_idx
        
        new_eps_r = config.MATERIAL_EPS_R * (1 - severity) + config.DEFECT_EPS_R * severity
        new_sigma = config.MATERIAL_SIGMA * (1 - severity) + config.DEFECT_SIGMA * severity
        
        self.eps_r[mask] = new_eps_r
        self.sigma[mask] = new_sigma
        
        self.ground_truth[mask] = 255
