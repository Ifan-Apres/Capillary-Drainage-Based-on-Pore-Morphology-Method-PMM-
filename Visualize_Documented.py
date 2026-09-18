import numpy as np
import pyvista as pv
import os
import re
import sys
import glob
import math

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

RAW_DIR = "Hasil_CUDA_ES32_HR_cropped_700_Segmented_001_Fix1_112"  # Directory containing .raw files
FILE_PATTERN = "sol_k*_sw*.raw" # File pattern
OUTPUT_IMG_DIR = "Visualisasi_Drainase_ES32_HR_cropped_700_Segmented_001_Fix1_112" # Output directory for images

# Use off-screen plotting for automation (prevents windows from popping up)
pv.set_plot_theme("document")

INPUT_TXT = "input.txt"

def read_params_from_input(path="input.txt"):
    params = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip().lower()
                    val = val.split("#", 1)[0].strip()
                    
                    if key == "filesize_x": params['x'] = int(val)
                    elif key == "filesize_y": params['y'] = int(val)
                    elif key == "filesize_z": params['z'] = int(val)
                    elif key == "resolution": params['res'] = float(val)
                    elif key == "sigma": params['sigma'] = float(val)
                    elif key == "theta": params['theta'] = float(val)
    except Exception as e:
        print(f"Error reading input: {e}")
        return None
    return params

def main():
    print("=== Drainage Visualization Generator (Static Images) ===")
    
    # 1. Get Parameters
    params = read_params_from_input(INPUT_TXT)
    if not params:
        print(f"Error: Could not read dimensions from {INPUT_TXT}")
        return
        
    width = params.get('x', 600)
    height = params.get('y', 600)
    depth = params.get('z', 600)
    res = params.get('res', 0.8)
    sigma = params.get('sigma', 485.0)
    theta = params.get('theta', 0.0)
    
    print(f"Volume: {width}x{height}x{depth}")
    print(f"Physics: Res={res} um, Sigma={sigma} mN/m, Theta={theta} deg")
    
    # 2. Find Files
    search_path = os.path.join(RAW_DIR, FILE_PATTERN)
    files = glob.glob(search_path)
    if not files:
        print(f"No files found in {search_path}")
        return
        
    # 3. Sort Files (High Sw -> Low Sw)
    # Parse (sw, k, fpath)
    file_list = []
    
    # Pre-calculate Constants for Pc
    # Pc = 2 * sigma * cos(theta) / r
    # sigma (mN/m) -> (N/m) * 1e-3
    # r (um) -> (m) * 1e-6
    sigma_nm = sigma * 1e-3
    theta_rad = math.radians(theta)
    
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse k and sw
        # pattern: sol_k*_sw*.raw
        match = re.search(r"sol_k(\d+)_sw([\d\.]+)\.raw", fname)
        if match:
            k = int(match.group(1))
            sw = float(match.group(2))
            
            # Calculate Pc
            # r = k * res (microns)
            if k > 0:
                r_microns = k * res
                r_meters = r_microns * 1e-6
                pc_pascal = (2 * sigma_nm * math.cos(theta_rad)) / r_meters
            else:
                pc_pascal = 0.0
                
            file_list.append({
                'sw': sw,
                'k': k,
                'pc': pc_pascal,
                'path': fpath
            })
            
    # Sort by Sw descending (Drainage start -> end)
    file_list.sort(key=lambda x: x['sw'], reverse=True)
    
    # Extract arrays for plotting the curve
    all_sw = [item['sw'] for item in file_list]
    all_pc = [item['pc'] for item in file_list]
    
    print(f"Found {len(file_list)} frames. Generating images...")
    
    # Create Output Directory
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
        
    # 4. Generate Images
    for i, item in enumerate(file_list):
        sw = item['sw']
        pc = item['pc']
        fpath = item['path']
        k_val = item['k']
        
        print(f"[{i+1}/{len(file_list)}] Processing Sw={sw:.4f}, Pc={pc:.2f} Pa (k={k_val})...", flush=True)
        
        
        # Load Data
        data = np.fromfile(fpath, dtype=np.uint8).reshape((depth, height, width))
        
        # Downsample for visualization if volume is large
        max_dim = max(width, height, depth)
        stride = 1
        if max_dim > 800:
            stride = 4
        elif max_dim > 400:
            stride = 4 # Aggressive downsample for testing/speed (600 -> 150)
            
        if stride > 1:
            print(f"  Downsampling data by stride {stride}...", flush=True)
            # Manual numpy downsampling to avoid PyVista version issues
            # data is (depth, height, width)
            data_down = data[::stride, ::stride, ::stride]
            
            # Update dimensions for the new grid
            d_depth, d_height, d_width = data_down.shape
            
            grid = pv.ImageData(
                dimensions=np.array([d_width, d_height, d_depth]) + 1,
                spacing=(stride, stride, stride),
                origin=(0, 0, 0)
            )
            grid.cell_data["Phase"] = data_down.flatten(order='F')
        else:
            grid = pv.ImageData(
                dimensions=np.array([width, height, depth]) + 1,
                spacing=(1, 1, 1),
                origin=(0, 0, 0)
            )
            grid.cell_data["Phase"] = data.flatten(order='F')
            
        # Thresholds on the (potentially) downsampled grid
        mesh_invaded = grid.threshold([0, 0.1], scalars="Phase")
        
        mesh_uninvaded = grid.threshold([0.9, 1.1], scalars="Phase")
        
        # Setup Plotter
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600))
        
        title_str = f"Sw = {sw:.4f} | Pc = {pc:.2f} Pa"
        
        # --- VIEW 1: Isometric ---
        plotter.subplot(0, 0)
        plotter.add_text(f"{title_str}\nIsometric View", font_size=10)
        if mesh_invaded.n_points > 0:
            plotter.add_mesh(mesh_invaded, color="red", show_edges=False, opacity=1.0)
        if mesh_uninvaded.n_points > 0:
            plotter.add_mesh(mesh_uninvaded, color="blue", show_edges=False, opacity=0.1)
        plotter.view_isometric()
        
        # --- VIEW 2: Top (XY) ---
        plotter.subplot(0, 1)
        plotter.add_text("Top View (XY)", font_size=10)
        if mesh_invaded.n_points > 0:
            plotter.add_mesh(mesh_invaded, color="red", show_edges=False, opacity=1.0)
        if mesh_uninvaded.n_points > 0:
            plotter.add_mesh(mesh_uninvaded, color="blue", show_edges=False, opacity=0.1) # Lower opacity for water to see through
        plotter.view_xy()
        
        # --- VIEW 3: Drainage Curve (2D Chart) ---
        plotter.subplot(0, 2)
        plotter.add_text("Capillary Pressure Curve", font_size=10)
        
        try:
            # Create Chart2D
            chart = pv.Chart2D()
            line_plot = chart.plot(all_sw, all_pc, "b-")
            
            # Handle return type safely
            if isinstance(line_plot, (list, tuple)):
                line_plot = line_plot[0]
            
            if line_plot is not None:
                line_plot.label = "Drainage Curve"
            
            # Highlight current point
            curr_point = chart.plot([sw], [pc], "r o")
            
            if isinstance(curr_point, (list, tuple)):
                curr_point = curr_point[0]
                
            if curr_point is not None:
                curr_point.label = "Current State"
                if hasattr(curr_point, "size"):
                     curr_point.size = 10
            
            chart.x_label = "Water Saturation (Sw)"
            chart.y_label = "Capillary Pressure (Pc) [Pa]"
            chart.grid = True
            
            plotter.add_chart(chart)
        except Exception as e:
            print(f"Warning: Could not add chart: {e}")

        
        # Add Legend to the first plot
        plotter.subplot(0, 0)
        plotter.add_legend(labels=[("Mercury (Invaded)", "red"), ("Water", "blue")])
        
        # Save
        out_name = f"drainage_step_{i:03d}_sw{sw:.4f}.png"
        out_path = os.path.join(OUTPUT_IMG_DIR, out_name)
        plotter.screenshot(out_path)
        
        # Close
        plotter.close()
        
    print(f"Done! Images saved to {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()
