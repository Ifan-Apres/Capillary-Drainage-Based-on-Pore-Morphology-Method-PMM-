import os
import numpy as np
import tifffile as tiff
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import porespy as ps
from skimage.filters import threshold_otsu
from skimage.morphology import ball, remove_small_objects

# ============================================================
# 1. LOAD DATA (TIFF SUPPORT)
# ============================================================

def load_tiff_volume(filepath):
    """
    Memuat volume dari file TIFF.
    """
    print(f"Loading TIFF file: {filepath}...")
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Read volume
        vol = tiff.imread(filepath)
        print(f"  -> Dimensions: {vol.shape}")
        print(f"  -> Dtype: {vol.dtype}")
        return vol
    except Exception as e:
        print(f"Error loading TIFF: {e}")
        # Try memmap if memory error is suspected
        try:
             print("  -> Retrying with memmap...")
             vol = tiff.memmap(filepath)
             return vol
        except:
             raise e

# ============================================================
# 2. PREPROCESSING (PURE OTSU & CLEANUP)
# ============================================================

def preprocess_microct(volume, pore_is="dark", output_dir=None):
    """
    Melakukan preprocessing: filter median, kontras, Otsu threshold, dan morphological opening.
    """
    print("  -> Applying Median Filter (size=2)...")
    vol = ndi.median_filter(volume, size=3)
    
    if output_dir:
        z = vol.shape[0] // 2
        plt.figure()
        plt.imshow(vol[z], cmap='gray')
        plt.title("After Median Filter")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "after_median.jpg"), dpi=300, bbox_inches='tight')
        plt.close()

    print("  -> Enhancing contrast...")
    vmin, vmax = np.percentile(vol, [1, 99])
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin)

    print("  -> Calculating Otsu Threshold...")
    t = threshold_otsu(vol)
    print(f"     Threshold: {t:.4f}")

    # --- PURE OTSU SEGMENTATION ---
    if pore_is == "dark":
        binary = vol < t   # PORE = TRUE (White)
    else:
        binary = vol > t

    if output_dir:
        z = binary.shape[0] // 2
        plt.figure()
        plt.imshow(binary[z], cmap='gray')
        plt.title("After Segmentation (Otsu)")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "After_segmentation.jpg"), dpi=300, bbox_inches='tight')
        plt.close()

    # --- NOISE REMOVAL ---
    print("  -> Applying Morphological Opening (Noise Removal)...")
    #binary = ndi.binary_opening(binary, structure=ball(1))
    
    #Optional: Very minimal cleanup
    #binary = remove_small_objects(binary, min_size=400) 
    
    if output_dir:
        z = binary.shape[0] // 2
        plt.figure()
        plt.imshow(binary[z], cmap='gray')
        plt.title("After Noise Removal")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "after_noise.jpg"), dpi=300, bbox_inches='tight')
        plt.close()
        
    return binary, float(t)

#def keep_largest_component(binary_vol):
    """
    Hanya menyisakan cluster pori terbesar yang terhubung.
    """
    #print("  -> Keeping largest connected pore cluster...")
    #labels, n = ndi.label(binary_vol)
    
    #sizes = np.bincount(labels.flat)
    #sizes[0] = 0  # ignore background
    #largest = sizes.argmax()

    #cleaned_binary = labels == largest
    #return cleaned_binary

# ============================================================
# 3. VISUALIZATION & CHECKS
# ============================================================

def visualize_rock_structure(im, vol_raw, output_dir):
    """
    Visualisasi hasil segmentasi: Grayscale, Binary, 3D Sketch, SEM View.
    """
    print(f"Generating Rock Structure Visualization in {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    crop_size = 650
    # Ensure crop size doesn't exceed image dimensions
    d, h, w = im.shape
    cz = min(crop_size, d)
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    
    im_crop = im[:cz, :ch, :cw]
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    slice_idx = im.shape[0] // 2
    
    # 1. Raw Grayscale
    ax[0].imshow(vol_raw[slice_idx], cmap='gray')
    ax[0].set_title(f"Raw Gray Z={slice_idx}")
    
    # 2. Binary Slice
    ax[1].imshow(im[slice_idx], cmap='gray')
    ax[1].set_title(f"Binary Z={slice_idx}")
    
    # 3. 3D Sketch
    ax[2].imshow(ps.visualization.show_3D(im_crop))
    ax[2].set_title(f"3D Sketch (Limit {crop_size})")
    ax[2].axis('off')

    # 4. SEM View
    ax[3].imshow(ps.visualization.sem(im_crop))
    ax[3].set_title("SEM View")
    ax[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Rock_Visualization.png'), dpi=300)
    plt.close()

# ============================================================
# 4. EXPORT
# ============================================================

def export_as_raw(binary_vol, output_dir, filename="segmented.raw"):
    """
    Ekspor volume biner sebagai file .raw (uint8, 0/255).
    """
    output_path = os.path.join(output_dir, filename)
    print(f"Exporting raw file to: {output_path}...")
    
    # Convert to PMM format: Pore=0, Solid=1
    # binary_vol: True=Pore, False=Solid
    # ~binary_vol: False(0)=Pore, True(1)=Solid
    if binary_vol.dtype == bool:
        raw_data = (~binary_vol).astype(np.uint8)
    else:
        # If already 0/1 but 1=Pore, invert it
        # Assuming input is 1=Pore, 0=Solid
        raw_data = (1 - binary_vol).astype(np.uint8)
        
    raw_data.tofile(output_path)
    print(f"  -> Export Complete. Format: Pore=0, Solid=1. Dtype: {raw_data.dtype}, Shape: {raw_data.shape}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    INPUT_TIFF = "mtgambier.tif"
    OUTPUT_DIR = "mtgambier_Segmented_Laporan_khususTA2"
    
    VOXEL_SIZE_UM = 3
    PREVIEW_ONLY = False  # Set True untuk cek visual dulu, False untuk run full
    PORE_IS_DARK = False    # True = Pori Hitam (< Threshold), False = Pori Putih (> Threshold)
    # ---------------------
    
    print(f"--- FASE 1: LOAD & SEGMENTASI ---")
    if not os.path.exists(INPUT_TIFF):
        print(f"ERROR: File {INPUT_TIFF} tidak ditemukan!")
        exit()
        
    vol = load_tiff_volume(INPUT_TIFF)
    
    # Create output dir early
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # --- PREVIEW MODE (Pop-up Comparison) ---
    if PREVIEW_ONLY:
        print("\n[PREVIEW MODE] Generating Dark vs Bright comparison...")
        
        # Calculate both options
        bin_dark, t_val = preprocess_microct(vol, pore_is="dark") # Option A
        bin_bright, _   = preprocess_microct(vol, pore_is="bright") # Option B
        
        # Viz Middle Slice
        z = vol.shape[0] // 2
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        ax[0].imshow(vol[z], cmap='gray')
        ax[0].set_title(f"Original Raw (Slice {z})")
        ax[0].axis('off')
        
        # Option A
        porosity_A = np.sum(bin_dark) / bin_dark.size
        ax[1].imshow(bin_dark[z], cmap='gray')
        ax[1].set_title(f"Option A: PORE IS DARK\n(White=Pore)\nPorosity: {porosity_A:.4f}")
        ax[1].axis('off')
        
        # Option B
        porosity_B = np.sum(bin_bright) / bin_bright.size
        ax[2].imshow(bin_bright[z], cmap='gray')
        ax[2].set_title(f"Option B: PORE IS BRIGHT\n(White=Pore)\nPorosity: {porosity_B:.4f}")
        ax[2].axis('off')
        
        plt.tight_layout()
        out_file = os.path.join(OUTPUT_DIR, "Preview_Comparison.png")
        plt.savefig(out_file, dpi=150)
        plt.close()
        
        print(f"1. Gambar perbandingan tersimpan di: {out_file}")
        print(f"2. Cek mana yang benar (butiran vs pori).")
        print(f"3. Ubah 'PORE_IS_DARK' di script sesuai hasil (True/False).")
        print(f"4. Ubah 'PREVIEW_ONLY = False' untuk lanjut simulasi.")
        print("EXITING PREVIEW.")
        exit()

    # Normal Run
    target_pol = "dark" if PORE_IS_DARK else "bright"
    binary, t_val = preprocess_microct(vol, pore_is=target_pol, output_dir=OUTPUT_DIR)
    
    # --- CONNECTIVITY CLEANUP ---
    binary = binary #keep_largest_component(binary)

    # Stats
    porosity = np.sum(binary) / binary.size
    print(f"Voxel Porosity: {porosity:.4f} (Mode: {target_pol})")
    
    # Viz Check
    visualize_rock_structure(binary, vol, OUTPUT_DIR)
    print(f"Visualisasi disimpan di {OUTPUT_DIR}. Cek Rock_Visualization.png")

    # --- EXPORT RAW ---
    export_as_raw(binary, OUTPUT_DIR, filename="mtgambier_Segmented_Laporan_khususTA2.raw")
