# Viva Preparation Notes: Microwave-Based Defect Detection

### 1. What is Microwave NDT?
Microwave Non-Destructive Testing (NDT) uses electromagnetic waves in the microwave frequency range (typically hundreds of MHz to hundreds of GHz) to inspect materials for flaws without damaging them. It works best on dielectric (non-conducting) materials like composites, plastics, and ceramics.

### 2. Why can microwaves detect defects?
Microwaves interact with the internal structure of materials. When a microwave signal travels through a material and hits a boundary where the material properties change (like a crack, air void, or moisture), a portion of the wave is reflected back. By measuring this reflection, we can deduce the presence of a defect.

### 3. What is impedance ($Z$)?
In this context, it's the intrinsic wave impedance of the material. It dictates how an electromagnetic wave propagates through it. It depends on the material's permittivity ($\epsilon$), permeability ($\mu$), and conductivity ($\sigma$). 
Formula: $Z = \sqrt{\frac{j \omega \mu}{\sigma + j \omega \epsilon}}$

### 4. What is the reflection coefficient ($\Gamma$)?
The reflection coefficient ($\Gamma$) is the ratio of the reflected wave's electric field to the incident wave's electric field. It tells us how much of the signal bounced back.
Formula: $\Gamma = \frac{Z_{material} - Z_{probe}}{Z_{material} + Z_{probe}}$
A defect changes $Z_{material}$, which in turn changes $\Gamma$.

### 5. What does S11 represent?
S11 is a scattering parameter commonly measured by a Vector Network Analyzer (VNA). It represents the reflection coefficient ($\Gamma$) at Port 1. In our project, the simulated $\Gamma$ magnitude is essentially the simulated $|S11|$ response of a single-probe setup.

### 6. Why are frequency and material properties relevant?
The material's response to microwaves depends on frequency ($\omega = 2\pi f$). The effective impedance changes with frequency, especially in lossy materials (where conductivity > 0). By changing the frequency, we can alter the penetration depth and sensitivity. Material properties ($\epsilon_r$) directly determine the baseline reflection; a defect (like air, $\epsilon_r=1$) causes a mismatch against the baseline (e.g., FR4, $\epsilon_r=4.4$).

### 7. Why is signal processing required?
Real sensors capture thermal noise and environmental interference. Without signal processing, this noise can be falsely identified as a defect. We use a 2D Gaussian filter (a low-pass filter) to smooth out high-frequency noise spikes in our raw reflection map.

### 8. Why do we normalize the data?
Our simulated $|\Gamma|$ values are small decimals (e.g., 0.2 to 0.6). Image processing algorithms (like OpenCV) typically work on 8-bit images (0 to 255). Normalization scales our RF data into a standard grayscale image format.

### 9. Why use Otsu's thresholding?
Otsu's method automatically calculates the optimal threshold value to separate pixels into two classes (foreground/defect and background) by minimizing intra-class variance. We don't have to hardcode a threshold value, making the system robust to different noise levels and materials.

### 10. Why use Morphological operations (Opening/Closing)?
*   **Opening (Erosion then Dilation)**: Removes small, isolated noisy pixels that survived thresholding.
*   **Closing (Dilation then Erosion)**: Fills in small holes or gaps within the detected defect contour, creating a solid, single blob representing the true defect.

### 11. What are the limitations of this simulation?
*   It is a simplified 1D impedance model evaluated over a 2D grid.
*   It ignores fringing fields, scattering, diffraction, and multiple internal reflections.
*   It assumes a perfect near-field probe that maps 1-to-1 with our grid pixels.

### 12. Why is this not equivalent to real VNA measurement?
A real VNA would capture complex 3D scattering effects and would have antenna radiation patterns to consider. Our simulation is a theoretical mathematical approximation of the physics to demonstrate the *concept* of the signal processing pipeline.

### 13. How could real VNA data be integrated in the future?
Instead of generating the 2D matrix using mathematical formulas, we would attach an X-Y mechanical scanner to a VNA probe, record the $|S11|$ values at each $(x, y)$ coordinate into a CSV, and feed that CSV directly into our `signal_processing.py` script.
