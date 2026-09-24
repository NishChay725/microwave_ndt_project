import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import config
from defect_generator import ComponentSurface
from microwave_model import get_microwave_response
from signal_processing import add_awgn_noise, normalize_response, apply_gaussian_filter
from image_processing import process_heatmap, extract_contours, calculate_defect_properties
from evaluation import evaluate_detection

st.set_page_config(page_title="Microwave NDT Simulator", layout="wide")

st.title("📡 Microwave-Based Defect Detection")
st.markdown("""
**Simulation Dashboard**

This application simulates Microwave Non-Destructive Testing (NDT) on a material surface. 
It calculates the theoretical reflection response from the material's impedance, applies noise, 
and uses signal and image processing to automatically detect structural defects.
""")

st.sidebar.header("⚙️ Simulation Parameters")

# Sidebar Controls
st.sidebar.subheader("Microwave Physics")
freq_ghz = st.sidebar.slider("Operating Frequency (GHz)", 1.0, 20.0, config.FREQ_HZ/1e9, 0.5)
mat_eps = st.sidebar.slider("Material Permittivity (εr)", 1.0, 10.0, config.MATERIAL_EPS_R, 0.1)
mat_sig = st.sidebar.slider("Material Conductivity (σ)", 0.0, 0.1, config.MATERIAL_SIGMA, 0.005)

st.sidebar.subheader("Sensor Noise")
noise_level = st.sidebar.slider("AWGN Noise Level (Std Dev)", 0.0, 0.1, config.NOISE_STD_DEV, 0.005)

st.sidebar.subheader("Defect Configuration")
defect_type = st.sidebar.selectbox("Defect Type", ["Rectangular Crack", "Circular Void"])
defect_x = st.sidebar.slider("Position X (mm)", 10, config.GRID_SIZE_X-10, int(config.GRID_SIZE_X/2))
defect_y = st.sidebar.slider("Position Y (mm)", 10, config.GRID_SIZE_Y-10, int(config.GRID_SIZE_Y/2))
defect_severity = st.sidebar.slider("Severity (0.0=None, 1.0=Full Air)", 0.0, 1.0, 1.0, 0.1)

if defect_type == "Rectangular Crack":
    defect_w = st.sidebar.slider("Width (mm)", 5, 50, 30)
    defect_h = st.sidebar.slider("Height (mm)", 5, 50, 10)
else:
    defect_r = st.sidebar.slider("Radius (mm)", 5, 40, 15)

if st.sidebar.button("▶️ Run Single Simulation", type="primary"):
    
    # 1. Setup Simulation Environment
    comp = ComponentSurface()
    comp.eps_r.fill(mat_eps)
    comp.sigma.fill(mat_sig)
    
    # Introduce defect
    true_area = 0
    true_centroid = (0, 0)
    
    if defect_type == "Rectangular Crack":
        comp.add_rectangular_defect(defect_x, defect_y, defect_w, defect_h, defect_severity)
        true_area = defect_w * defect_h / (config.RESOLUTION ** 2)
        true_centroid = (defect_x / config.RESOLUTION + defect_w / (2*config.RESOLUTION), 
                         defect_y / config.RESOLUTION + defect_h / (2*config.RESOLUTION))
    else:
        comp.add_circular_defect(defect_x, defect_y, defect_r, defect_severity)
        true_area = np.pi * (defect_r / config.RESOLUTION)**2
        true_centroid = (defect_x / config.RESOLUTION, defect_y / config.RESOLUTION)
        
    # Run physics model
    with st.spinner("Calculating EM Wave Reflection..."):
        freq_hz = freq_ghz * 1e9
        raw_response = get_microwave_response(comp.eps_r, comp.sigma, freq_hz)
        noisy_response = add_awgn_noise(raw_response, std_dev=noise_level)
        
    # Run processing
    with st.spinner("Applying Signal & Image Processing..."):
        filtered_response = apply_gaussian_filter(noisy_response)
        normalized_heatmap = normalize_response(filtered_response)
        
        # Image Processing intermediate steps (we'll manually do them to show intermediate results)
        blurred = cv2.GaussianBlur(normalized_heatmap, config.BLUR_KERNEL_SIZE, 0)
        ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        contours = extract_contours(binary_mask)
        defects_info, area_percentage = calculate_defect_properties(contours, comp.shape)
        metrics = evaluate_detection(comp.ground_truth, binary_mask, defects_info, true_centroid, true_area)

    st.success("Simulation Complete!")
    
    # --- 3D INTERACTIVE VISUALIZATION ---
    st.header("🌐 3D Microwave Reflection Topology")
    st.markdown("Interact with the 3D surface plot below to observe how the **reflection magnitude ($|\Gamma|$)** changes structurally across the simulated component due to the defect and noise.")
    
    # Create the 3D Surface Plot using Plotly
    fig_3d = go.Figure(data=[go.Surface(z=noisy_response, colorscale='Jet')])
    fig_3d.update_layout(
        title='3D Noisy Microwave Response ($|\Gamma|$)',
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Reflection $|\Gamma|$'
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.divider()

    # Display Pipeline visually
    st.header("🔍 Processing Pipeline")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**1. Ground Truth Material (εr)**")
        fig, ax = plt.subplots()
        cax = ax.imshow(comp.eps_r, cmap='viridis')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.write("**2. Simulated RF Response (|Γ|)**")
        fig, ax = plt.subplots()
        cax = ax.imshow(raw_response, cmap='jet')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)
        
    with col3:
        st.write("**3. Noisy Sensor Data**")
        fig, ax = plt.subplots()
        cax = ax.imshow(noisy_response, cmap='jet')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)
        
    with col4:
        st.write("**4. Filtered & Normalized**")
        fig, ax = plt.subplots()
        cax = ax.imshow(normalized_heatmap, cmap='gray')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)
        
    st.divider()
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.write("**5. Otsu's Thresholding**")
        fig, ax = plt.subplots()
        ax.imshow(thresholded, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
    with col6:
        st.write("**6. Morphological Output**")
        fig, ax = plt.subplots()
        ax.imshow(binary_mask, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
    with col7:
        st.write("**7. Detected Contours**")
        overlay = np.zeros((comp.shape[0], comp.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        fig, ax = plt.subplots()
        ax.imshow(overlay)
        ax.axis('off')
        st.pyplot(fig)
        
    with col8:
        st.write("**8. GT vs Detected Overlay**")
        comparison = np.zeros((comp.shape[0], comp.shape[1], 3), dtype=np.uint8)
        comparison[:,:,1] = comp.ground_truth  # Green
        cv2.drawContours(comparison, contours, -1, (255, 0, 0), 2) # Red
        fig, ax = plt.subplots()
        ax.imshow(comparison)
        ax.axis('off')
        st.pyplot(fig)
        
    st.divider()
    
    # Results Section
    st.header("📊 Final Detection Results")
    
    if len(defects_info) > 0:
        main_defect = max(defects_info, key=lambda x: x['area_pixels'])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detected Area", f"{main_defect['area_mm2']:.2f} mm²")
        c2.metric("Centroid (X,Y)", f"({main_defect['centroid_x']}, {main_defect['centroid_y']})")
        c3.metric("Defect Extent", f"{area_percentage:.2f}%")
        
        confidence = min(100.0, max(0.0, 100 * defect_severity * (1.0 - noise_level * 5)))
        c4.metric("Detection Confidence", f"{confidence:.1f}%")
        
        st.subheader("Performance Metrics (vs Ground Truth)")
        m1, m2, m3 = st.columns(3)
        m1.metric("IoU (Intersection over Union)", f"{metrics.get('IoU', 0):.4f}")
        m2.metric("Localization Error", f"{metrics.get('Localization_Error_Pixels', 0):.2f} pixels")
        m3.metric("Area Error", f"{metrics.get('Area_Error_Percent', 0):.2f}%")
    else:
        st.error("No defects were detected! Try lowering noise or increasing defect severity.")

st.sidebar.divider()
st.sidebar.subheader("Bulk Evaluation Mode")
if st.sidebar.button("📊 Run Evaluation Suite"):
    st.header("📈 Evaluation Suite Results")
    st.write("Running 5 different test cases automatically...")
    
    test_cases = [
        {"name": "Large Crack", "type": "rect", "params": (50, 50, 30, 10, 1.0)},
        {"name": "Small Void", "type": "circ", "params": (150, 120, 15, 1.0)},
        {"name": "Subsurface Flaw (Low Severity)", "type": "rect", "params": (80, 150, 40, 20, 0.4)},
        {"name": "Tiny Pit", "type": "circ", "params": (100, 30, 8, 1.0)},
        {"name": "Edge Crack", "type": "rect", "params": (10, 180, 40, 5, 0.9)}
    ]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, tc in enumerate(test_cases):
        comp = ComponentSurface()
        if tc["type"] == "rect":
            x, y, w, h, sev = tc["params"]
            comp.add_rectangular_defect(x, y, w, h, sev)
            ta = w * h / (config.RESOLUTION ** 2)
            tc_centroid = (x / config.RESOLUTION + w / (2*config.RESOLUTION), 
                           y / config.RESOLUTION + h / (2*config.RESOLUTION))
        else:
            cx, cy, r, sev = tc["params"]
            comp.add_circular_defect(cx, cy, r, sev)
            ta = np.pi * (r / config.RESOLUTION)**2
            tc_centroid = (cx / config.RESOLUTION, cy / config.RESOLUTION)
            
        raw = get_microwave_response(comp.eps_r, comp.sigma, config.FREQ_HZ)
        noisy = add_awgn_noise(raw)
        filt = apply_gaussian_filter(noisy)
        norm = normalize_response(filt)
        mask = process_heatmap(norm)
        conts = extract_contours(mask)
        d_info, _ = calculate_defect_properties(conts, comp.shape)
        met = evaluate_detection(comp.ground_truth, mask, d_info, tc_centroid, ta)
        
        results.append({
            "Test Case": tc["name"],
            "IoU": met.get('IoU', 0),
            "Loc Error (px)": met.get('Localization_Error_Pixels', 0),
            "Area Error (%)": met.get('Area_Error_Percent', 0)
        })
        progress_bar.progress((i + 1) / len(test_cases))
        
    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df.style.format({
        "IoU": "{:.4f}", 
        "Loc Error (px)": "{:.2f}", 
        "Area Error (%)": "{:.2f}%"
    }), use_container_width=True)
    
    st.success(f"Average IoU across test cases: {df['IoU'].mean():.4f}")
