# Skrip ini hasil revisi pada masalah terdapat nilai minus pada DSw
#untuk mengatasi hal tersebut, dilakukan penerapan memori trapping pada fasa merkuri juga
import cupy as cp
import cupyx.scipy.ndimage as ndi
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time

# --------------------------------------------------------------------------------------
# Input / Config Parsing
# --------------------------------------------------------------------------------------

def read_input_params(path="input.txt"):
    """Parse a simple key = value text file and return a lowercase-key dictionary."""
    params = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.split("#", 1)[0].strip()
                params[key.strip().lower()] = value
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    return params

def _validate_and_read_input():
    """Read params and raw volume, validate sizes; return a dict of runtime inputs."""
    params = read_input_params()

    required_keys = [
        "filename", "filesize_x", "filesize_y", "filesize_z",
        "resolution", "sigma", "theta",
    ]
    missing = [k for k in required_keys if k not in params]
    if missing:
        raise ValueError(f"Missing required keys in input.txt: {', '.join(missing)}")

    file_path = params["filename"]
    width = int(params["filesize_x"])
    height = int(params["filesize_y"])
    depth = int(params["filesize_z"])

    resolution = float(params["resolution"])
    sigma = float(params["sigma"])
    theta = float(params["theta"])

    total_elements = depth * height * width
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
        
    file_size_bytes = os.path.getsize(file_path)
    expected_bytes = total_elements
    if file_size_bytes != expected_bytes:
        # Try checking if header exists or mismatch
        print(f"Warning: File size {file_size_bytes} != Expected {expected_bytes}. Proceeding with reshape if possible.")

    print(f"Loading raw data from {file_path}...")
    with open(file_path, "rb") as f:
        # Load initially to CPU RAM as uint8
        # Assuming PMM standard: 0=Pore, 1=Solid
        raw_cpu = np.fromfile(f, dtype=np.uint8, count=total_elements)

    raw_cpu = raw_cpu.reshape((depth, height, width))
    
    # Transfer to GPU
    print("Transferring data to GPU...")
    domain_gpu = cp.array(raw_cpu)
    
    # Free CPU memory
    del raw_cpu
    
    # Calculate porosity on GPU
    # 0 = pore, 1 = solid (input standard)
    pore_voxels = cp.sum(domain_gpu == 0)
    solid_voxels = cp.sum(domain_gpu == 1)
    porosity = float(pore_voxels / (pore_voxels + solid_voxels))

    return {
        "params": params,
        "width": width, "height": height, "depth": depth,
        "resolution": resolution, "sigma": sigma, "theta": theta,
        "porosity": porosity,
        "domain_gpu": domain_gpu,
        "file_path": file_path
    }

# --------------------------------------------------------------------------------------
# GPU Morphology Helpers
# --------------------------------------------------------------------------------------

_GPU_SE_CACHE = {}

def _get_ball_gpu(radius: int):
    """Return a cached spherical structuring element on GPU."""
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if radius in _GPU_SE_CACHE:
        return _GPU_SE_CACHE[radius]
    
    # Generate on CPU using skimage (optimized C implementation)
    se_cpu = ball(radius, dtype=np.uint8)
    # Transfer to GPU
    se_gpu = cp.array(se_cpu)
    _GPU_SE_CACHE[radius] = se_gpu
    return se_gpu

# --------------------------------------------------------------------------------------
# Core Simulation Logic (GPU)
# --------------------------------------------------------------------------------------

def compute_saturation_gpu(domain, k, theta, mask_prev=None, nwp_prev=None):
    """
    Run PMM simulation step on GPU.
    domain: cupy array (0=pore, 1=solid)
    k: kernel radius
    mask_prev: previous trapping mask (0=trapped)
    nwp_prev: previous NWP mask (0=NWP)
    """
    theta_rad = math.radians(theta)
    r_solid = max(round(k * math.cos(theta_rad)), 1)
    
    # Kernel for NWP depends on basic PMM logic = k
    r_nwp = round(k)
    
    # 1. Grain Dilation (Solid Expansion)
    # domain: 1=Solid. dilation acts on 1s.
    se_solid = _get_ball_gpu(r_solid)
    grained = ndi.binary_dilation(domain, structure=se_solid)
    
    # 2. Connectivity to Inlet (z=0)
    # Accessible Pore Spaces are regions where grained == 0 (Pore) and connected to z=0
    pore_space = (grained == 0)
    labeled_pores, num = ndi.label(pore_space)
    
    if num > 0:
        inlet_labels = cp.unique(labeled_pores[0, :, :])
        inlet_labels = inlet_labels[inlet_labels != 0] # Remove background 0
        connected_pore_mask = cp.isin(labeled_pores, inlet_labels)
    else:
        connected_pore_mask = cp.zeros_like(domain, dtype=bool)

    # 3. NWP Expansion (Erosion of inversed space)
    # We want to place NWP spheres centered at valid locations (connected_pore_mask).
    # Physically: Place Sphere(R) at every pixel in connected_pore_mask.
    # Mathematical Morphology: Dilation of connected_pore_mask by Se(R).
    
    se_nwp = _get_ball_gpu(r_nwp)
    nwp_filled = ndi.binary_dilation(connected_pore_mask, structure=se_nwp)
    
    # 4. Construct Phase Field
    # 0 = NWP
    # 1 = Wetting
    # 2 = Solid
    comb = cp.full_like(domain, 1, dtype=cp.uint8) # Default WP
    if nwp_prev is not None:
        comb[nwp_prev] = 0
    comb[nwp_filled ==1] = 0
    comb[domain == 1] = 2


    # 5. Apply Trapping (Memory)
    # If WP was trapped previously, it cannot become NWP now.
    if mask_prev is not None:
        # mask_prev == 0 means TRAPPED.
        # Force those voxels to be WP (1).
        comb[mask_prev == 0] = 1
        
    # 6. Detect NEW Trapping
    # WP (1) must be connected to Outlet (z=max).
    wp_voxels = (comb == 1)
    labeled_wp, num_wp = ndi.label(wp_voxels)
    
    if num_wp > 0:
        outlet_labels = cp.unique(labeled_wp[-1, :, :])
        outlet_labels = outlet_labels[outlet_labels != 0]
        connected_wp = cp.isin(labeled_wp, outlet_labels)
        
        # Trapped = Is WP but NOT connected
        is_trapped = wp_voxels & (~connected_wp)
    else:
        is_trapped = cp.zeros_like(domain, dtype=bool)
        
    # Create new mask for next step (0=Trapped, 1=Active)
    new_mask = cp.ones_like(domain, dtype=cp.uint8)
    new_mask[is_trapped] = 0
    
    # Enforce trapping on current result too
    comb[is_trapped] = 1 # Redundant but safe
    new_nwp = (comb == 0)
    # Calc Saturation
    count_wp = cp.sum(comb == 1)
    count_nwp = cp.sum(comb == 0)
    total_pore = count_wp + count_nwp
    
    if total_pore == 0:
        sw = 1.0
    else:
        sw = float(count_wp / total_pore)
        
    return sw, comb, new_mask, new_nwp

# --------------------------------------------------------------------------------------
# Kernel Search (GPU)
# --------------------------------------------------------------------------------------

def find_best_kernel_size_gpu(domain, target_sat, theta):
    """Find starting kernel size."""
    k = 20
    step = 1
    max_k = 200
    tested = set()
    best_k = k
    best_diff = 100.0
    
    print(f"Searching for kernel size matching S_air={target_sat}...")
    
    while k <= max_k:
        sw, _, _ = compute_saturation_gpu(domain, k, theta, mask_prev=None)
        diff = abs(sw - target_sat)
        print(f"  Kernel {k} -> S_air {sw:.4f} (diff {diff:.4f})")
        
        if diff < best_diff:
            best_diff = diff
            best_k = k
            
        if diff < 0.01: # 1% tolerance
            break
            
        # Heuristic step
        if sw > target_sat:
            pass

        break # Placeholder, implementing Linear Scan below
        
    # Simple Linear/Smart Scan
    k_scan = 100
    while k_scan > 1:
        sw, _, _ = compute_saturation_gpu(domain, k_scan, theta, mask_prev=None)
        print(f"  Scan K={k_scan} -> S_air={sw:.4f}")
        if sw <= target_sat:
            return k_scan
        k_scan -= 2 # Step 2
        
    return 1

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("=== PMM CUDA ACCELERATED ===")
    
    # Load Input
    rt = _validate_and_read_input()
    params = rt["params"]
    domain = rt["domain_gpu"]
    theta = rt["theta"]
    sigma = rt["sigma"]
    res = rt["resolution"]
    
    kernel_search = params.get("kernel_search", "true").lower() == "true"
    vis = params.get("visualization", "true").lower() == "true"
    
    start_k = int(params.get("starting_kernel", 20))
    start_sat = float(params.get("starting_sat", 0.95))
    
    current_k = start_k
    
    if kernel_search:
        # Custom search logic
        print(f"Auto-search start kernel for Sat ~ {start_sat}...")
        for k_try in range(200, 1, -2):
            sw, _, _ = compute_saturation_gpu(domain, k_try, theta)
            print(f"  Check K={k_try} -> S_air={sw:.4f}")
            if sw <= start_sat:
                current_k = k_try
                break
        print(f"Selected Start Kernel: {current_k}")
        
    # Main Loop variables
    mask_global = None # Initial mask
    nwp_global = None
    sat_history = []
    pc_history = []
    
    # New Arrays for PSD and Data Tracking
    psd_history = []
    r_micron_history = []
    k_history = []
    
    s_air_prev = None
    
    # Create Output Dir
    out_dir = "Hasil_CUDA_P01_700_TA2"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    print("Starting Simulation Loop...")
    t0 = time.time()
    
    while current_k > 0:
        s_air, comb, new_mask, new_nwp = compute_saturation_gpu(domain, current_k, theta, mask_global, nwp_global)
        mask_global = new_mask # Update mask for next step
        nwp_global = new_nwp
        # Calculate Pc (Pascals)
        r_micron = current_k * res
        r_meter = r_micron * 1e-6
        sigma_nm = sigma * 1e-3
        pc_pascal = (2 * sigma_nm * math.cos(math.radians(theta))) / r_meter
        
        # Mercury Saturation
        s_hg = 1.0 - s_air
        
        # PSD Calculation (Delta_Sw = newly invaded pore volume)
        if s_air_prev is not None:
            delta_s = s_air_prev - s_air
        else:
            delta_s = 1.0 - s_air  # First invasion compared to 100% Air
            
        sat_history.append(s_air)
        pc_history.append(pc_pascal)
        psd_history.append((r_micron, delta_s))
        r_micron_history.append(r_micron)
        k_history.append(current_k)
        
        s_air_prev = s_air
        
        print(f"K={current_k:3d} | Pc={pc_pascal:10.2f} Pa | S_air={s_air:.4f} | S_Hg={s_hg:.4f} | dS={delta_s:.4f}")
        
        if vis and s_air < 1.0:
            # Save RAW
            comb_cpu = cp.asnumpy(comb)
            fname = os.path.join(out_dir, f"sol_k{current_k}_sw{s_air:.3f}.raw")
            comb_cpu.tofile(fname)
            
        current_k -= 1
        
        if s_air < 0.001:
            print("S_air reached ~0. Stopping.")
            break
            
    dt = time.time() - t0
    print(f"Simulation finished in {dt:.2f} seconds.")
    
    # Save Results (Mercury-Air System format)
    with open(os.path.join(out_dir, "result_cuda.txt"), "w") as f:
        f.write("Kernel_Radius_um\tPc_Pa\tS_air\tS_Hg\n")
        for r_mic, pc, s_air in zip(r_micron_history, pc_history, sat_history):
            s_hg = 1.0 - s_air
            f.write(f"{r_mic:.4f}\t{pc:.4f}\t{s_air:.6f}\t{s_hg:.6f}\n")
            
    # Save PSD data
    with open(os.path.join(out_dir, "psd_cuda.txt"), "w") as f:
        f.write("Radius(micron)\tDelta_Sw\n")
        for r_mic, delta_s in psd_history:
            f.write(f"{r_mic:.4f}\t{delta_s:.6f}\n")
            
    # Plot Configuration (Mercury-Air)
    plt.figure()
    plt.plot(sat_history, pc_history, '-o')
    plt.xlabel("Air Saturation (S_air)")
    plt.ylabel("Capillary Pressure (Pa) [Hg-Air]")
    plt.title("MICP Drainage Curve (Mercury-Air)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "curve_cuda.png"))
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
