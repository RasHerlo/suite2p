import os
import numpy as np
from suite2p import default_ops
from suite2p.registration import register
from pathlib import Path
from tifffile import imread, imwrite
import re
import matplotlib.pyplot as plt
import suite2p
import shutil

def combine_registered_tiffs(suite2p_dir):
    """
    Combine all registered TIFF files in the reg_tif folder into a single stack.
    Files are combined in numerical order based on their frame numbers.
    """
    reg_tif_dir = os.path.join(suite2p_dir, 'reg_tif')
    if not os.path.exists(reg_tif_dir):
        print(f"No reg_tif directory found in {suite2p_dir}")
        return
    
    # Get all tif files and sort them numerically
    tif_files = [f for f in os.listdir(reg_tif_dir) if f.endswith('.tif')]
    tif_files.sort(key=lambda x: int(re.search(r'file(\d+)', x).group(1)))
    
    if not tif_files:
        print("No TIFF files found in reg_tif directory")
        return
    
    # Read and combine all files
    print(f"Combining {len(tif_files)} TIFF files...")
    combined_stack = []
    for tif_file in tif_files:
        file_path = os.path.join(reg_tif_dir, tif_file)
        stack = imread(file_path)
        combined_stack.append(stack)
    
    # Concatenate all stacks
    final_stack = np.concatenate(combined_stack, axis=0)
    
    # Save combined stack
    output_path = os.path.join(suite2p_dir, 'combined_registered.tif')
    imwrite(output_path, final_stack)
    print(f"Saved combined stack to {output_path}")

def save_mean_images(suite2p_dir, ops):
    """
    Save MeanImg and MeanImgE as PNG files with coolwarm colormap
    """
    # Save MeanImg
    plt.figure(figsize=(8, 8))
    plt.imshow(ops['meanImg'], cmap='coolwarm')
    plt.axis('off')
    plt.savefig(os.path.join(suite2p_dir, 'MeanImg.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save MeanImgE if it exists
    if 'meanImgE' in ops:
        plt.figure(figsize=(8, 8))
        plt.imshow(ops['meanImgE'], cmap='coolwarm')
        plt.axis('off')
        plt.savefig(os.path.join(suite2p_dir, 'MeanImgE.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def process_folder(parent_folder):
    """
    Process all SUPPORT_ChanA/B folders within the parent folder.
    For each folder containing denoised_cut.tif, perform registration
    and save outputs in a 'suite2p files' subfolder.
    """
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_folder):
        # Check if this is a SUPPORT_Chan folder
        if 'SUPPORT_ChanA' in root or 'SUPPORT_ChanB' in root:
            # Look for denoised_cut.tif
            if 'denoised_cut.tif' in files:
                print(f"Processing {root}")
                
                # Create suite2p files directory
                suite2p_dir = os.path.join(root, 'suite2p files')
                
                # Skip if suite2p files already exist
                if os.path.exists(suite2p_dir):
                    print(f"Suite2p files already exist in {root}, skipping...")
                    continue
                
                os.makedirs(suite2p_dir, exist_ok=True)
                
                # Get the tif file path
                tif_path = os.path.join(root, 'denoised_cut.tif')
                
                # Check if denoised.tif exists in the same directory
                denoised_path = os.path.join(root, 'denoised.tif')
                if os.path.exists(denoised_path):
                    print(f"Found denoised.tif, deleting: {denoised_path}")
                    os.remove(denoised_path)
                
                # Set up default options
                ops = default_ops()
                ops['reg_tif'] = True  # Save registered tiffs
                ops['nonrigid'] = True  # Use nonrigid registration
                ops['save_path'] = suite2p_dir
                
                # Read the tif file using tifffile
                mov = imread(tif_path)
                if mov.ndim == 2:
                    mov = mov[np.newaxis, :, :]
                
                # Convert data to int16 following suite2p's approach
                if mov.dtype.type == np.uint16:
                    mov = (mov // 2).astype(np.int16)
                elif mov.dtype.type == np.int32:
                    mov = (mov // 2).astype(np.int16)
                elif mov.dtype.type != np.int16:
                    mov = mov.astype(np.int16)
                
                # Update ops with movie dimensions
                ops['nframes'] = mov.shape[0]
                ops['Ly'] = mov.shape[1]
                ops['Lx'] = mov.shape[2]
                
                # Perform registration
                refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest = register.compute_reference_and_register_frames(
                    f_align_in=mov,
                    ops=ops
                )
                
                # Save ops file with registration results
                ops['refImg'] = refImg
                ops['rmin'] = rmin
                ops['rmax'] = rmax
                ops['meanImg'] = meanImg
                ops['yoff'] = rigid_offsets[0]
                ops['xoff'] = rigid_offsets[1]
                ops['corrXY'] = rigid_offsets[2]
                if nonrigid_offsets:
                    ops['yoff1'] = nonrigid_offsets[0]
                    ops['xoff1'] = nonrigid_offsets[1]
                    ops['corrXY1'] = nonrigid_offsets[2]
                
                # Generate enhanced mean image
                ops['meanImgE'] = register.compute_enhanced_mean_image(ops['meanImg'].astype(np.float32), ops)
                
                # Save ops file
                np.save(os.path.join(suite2p_dir, 'ops.npy'), ops)
                print(f"Saved registration results to {suite2p_dir}")
                
                # Save mean images as PNGs
                save_mean_images(suite2p_dir, ops)
                
                # Combine registered TIFF files
                combine_registered_tiffs(suite2p_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process registration for SUPPORT_Chan folders')
    parser.add_argument('parent_folder', type=str, help='Parent folder containing SUPPORT_Chan folders')
    args = parser.parse_args()
    
    process_folder(args.parent_folder) 