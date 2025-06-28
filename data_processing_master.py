#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master Script for Data Processing Pipeline
-----------------------------------------
This script combines multiple processing steps:
1. FFT mask processing for noise removal
2. Suite2p pipeline processing
3. ROI selection and classification
4. Rasterplot generation
5. Data export

Usage:
    python data_processing_master.py path_to_root_directory

Author: [User]
Date: 2024-03-19
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from rastermap.rastermap import Rastermap
import pickle
from suite2p.run_s2p import pipeline, run_s2p
from suite2p.default_ops import default_ops
from roi_selection_new import ROISelector
import time
from datetime import timedelta
from suite2p.io import BinaryFile
import tifffile

def find_channel_folders(root_dir):
    """
    Find all SUPPORT_ChanA and SUPPORT_ChanB folders in the directory tree.
    
    Parameters:
    -----------
    root_dir : str
        Root directory to search in
        
    Returns:
    --------
    dict
        Dictionary with channel folders and their types
    """
    channel_folders = {}
    root_path = Path(root_dir)
    
    for folder in root_path.rglob('SUPPORT_Chan*'):
        if folder.name == 'SUPPORT_ChanA':
            channel_folders[str(folder)] = 'A'
        elif folder.name == 'SUPPORT_ChanB':
            channel_folders[str(folder)] = 'B'
    
    return channel_folders

def process_fft_masks(input_path, channel_type, output_dir):
    """
    Process FFT masks based on channel type.
    
    Parameters:
    -----------
    input_path : str
        Path to input .tif file
    channel_type : str
        'A' or 'B' indicating channel type
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    str
        Path to processed output file
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the .tif stack
    print(f"Loading {input_path}...")
    images = imread(input_path, is_ome=False)
    
    # Normalize if necessary
    if images.min() < 0:
        images -= images.min()
    
    # Compute FFT for all frames
    print("Computing FFT...")
    fft_images = np.fft.fft2(images)
    fft_images = np.fft.fftshift(fft_images)
    
    # Compute the mean amplitude for visualization
    fft_amplitude = np.abs(fft_images)
    fft_mean = np.mean(fft_amplitude, axis=0)
    
    # Create mask based on channel type
    if channel_type == 'A':
        # Circular/ring mask for ChanA
        mask = create_circular_mask(images.shape, [(15, 25)])
    else:
        # Rectangular masks for ChanB
        mask = create_rectangular_mask(images.shape, [[-52, -81, 43, 7], [52, 81, 43, 7]])
    
    # Apply the mask to all frames
    masked_fft = fft_images.copy()
    masked_fft[:, ~mask] = 0
    
    # Inverse FFT to get denoised images
    print("Computing inverse FFT...")
    reconstructed_images = np.fft.ifftshift(masked_fft)
    reconstructed_images = np.fft.ifft2(reconstructed_images)
    denoised_images = np.abs(reconstructed_images)
    
    # Save outputs
    output_tif = output_dir / "derippled_stack.tif"
    output_fft_img = output_dir / "fft_amplitude.png"
    output_fft_mean = output_dir / "fft_amplitude_mean.png"
    
    # Save denoised images
    print(f"Saving denoised images to {output_tif}...")
    imwrite(output_tif, denoised_images.astype(np.float32))
    
    # Save FFT visualizations
    save_fft_visualization(fft_amplitude[0], mask, output_fft_img, 
                          [(15, 25)] if channel_type == 'A' else [[-52, -81, 43, 7], [52, 81, 43, 7]],
                          'circle' if channel_type == 'A' else 'rect')
    
    save_fft_visualization(fft_mean, mask, output_fft_mean,
                          [(15, 25)] if channel_type == 'A' else [[-52, -81, 43, 7], [52, 81, 43, 7]],
                          'circle' if channel_type == 'A' else 'rect')
    
    return str(output_tif)

def create_rectangular_mask(shape, mask_coords_list):
    """Create rectangular mask for FFT filtering."""
    _, h, w = shape
    mask = np.ones((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    
    for mask_coords in mask_coords_list:
        x0, y0, dx, dy = mask_coords
        x_min = int(cx + x0 - dx)
        x_max = int(cx + x0 + dx)
        y_min = int(cy + y0 - dy)
        y_max = int(cy + y0 + dy)
        
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        mask[y_min:y_max, x_min:x_max] = False
    
    return mask

def create_circular_mask(shape, mask_coords_list):
    """Create circular mask for FFT filtering."""
    _, h, w = shape
    mask = np.ones((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    y = y - cy
    x = x - cx
    dist_squared = x*x + y*y
    
    for inner_radius, outer_radius in mask_coords_list:
        ring_mask = (dist_squared < inner_radius**2) | (dist_squared > outer_radius**2)
        mask = mask & ring_mask
    
    return mask

def save_fft_visualization(fft_data, mask, output_path, mask_coords_list, mask_type):
    """Save FFT visualization with masks."""
    plt.figure(figsize=(10, 8))
    
    fft_vis = np.log(fft_data + 1e-6)
    fft_min = np.percentile(fft_vis, 1)
    fft_max = np.percentile(fft_vis, 99)
    fft_vis = (fft_vis - fft_min) / (fft_max - fft_min)
    
    rgb_img = np.zeros((*fft_vis.shape, 3))
    rgb_img[:, :, 0] = fft_vis
    rgb_img[:, :, 1] = fft_vis
    rgb_img[:, :, 2] = fft_vis
    
    rgb_img[~mask, 0] = 0.8
    rgb_img[~mask, 1] = 0.0
    rgb_img[~mask, 2] = 0.8
    
    if mask_type == 'rect':
        mask_str = ' '.join([f'[{int(x0)},{int(y0)},{int(dx)},{int(dy)}]' 
                            for x0, y0, dx, dy in mask_coords_list])
    else:
        mask_str = ' '.join([f'({int(inner)},{int(outer)})' 
                            for inner, outer in mask_coords_list])
    
    plt.imshow(rgb_img)
    plt.title(f'FFT Amplitude with masks: {mask_str}')
    plt.colorbar(label='Log Amplitude')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_suite2p_pipeline(input_path, output_dir):
    """
    Run suite2p pipeline with default settings.
    
    Parameters:
    -----------
    input_path : str
        Path to input .tif file
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    Path
        Path to suite2p output directory
    """
    # Initialize ops with default settings
    ops = default_ops()
    
    # Set up file paths
    ops['save_path0'] = str(output_dir)
    ops['save_folder'] = 'suite2p'
    ops['fast_disk'] = str(output_dir)  # Use same directory for temporary files
    
    # Set up processing options
    ops['nplanes'] = 1
    ops['nchannels'] = 1
    ops['functional_chan'] = 1
    ops['tau'] = 1.0  # Timescale of the sensor
    ops['fs'] = 10.0  # Sampling rate per plane
    
    # Registration options
    ops['do_registration'] = True
    ops['nonrigid'] = True
    ops['block_size'] = [128, 128]
    ops['maxregshift'] = 0.1
    ops['align_by_chan'] = 1
    
    # ROI detection options
    ops['roidetect'] = True
    ops['spikedetect'] = True
    ops['spatial_scale'] = 0  # Multi-scale detection
    ops['connected'] = True
    ops['max_overlap'] = 0.75
    
    # Signal extraction options
    ops['neuropil_extract'] = True
    ops['inner_neuropil_radius'] = 2
    ops['min_neuropil_pixels'] = 350
    ops['neucoeff'] = 0.7
    
    # Set data path to the directory containing the TIFF file
    ops['data_path'] = [str(Path(input_path).parent)]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if suite2p output already exists and is complete
    suite2p_dir = Path(output_dir) / 'suite2p' / 'plane0'
    required_files = ['ops.npy', 'stat.npy', 'F.npy', 'Fneu.npy', 'iscell.npy', 'spks.npy']
    
    if suite2p_dir.exists():
        existing_files = [f.name for f in suite2p_dir.iterdir() if f.is_file()]
        print(f"[DEBUG] Found existing suite2p files: {existing_files}")
        if all(f in existing_files for f in required_files):
            print("[DEBUG] All required suite2p files exist, skipping processing")
            return suite2p_dir
    
    # Run the pipeline using run_s2p
    try:
        print("[DEBUG] Starting suite2p pipeline with ops:", ops)
        ops = run_s2p(ops=ops)
        return suite2p_dir
    except Exception as e:
        print(f"[DEBUG] Error in suite2p pipeline: {str(e)}")
        print(f"[DEBUG] Error type: {type(e)}")
        print(f"[DEBUG] Error args: {e.args}")
        raise

def process_roi_selection(suite2p_dir, output_dir):
    """
    Process ROI selection with fixed parameters.
    
    Parameters:
    -----------
    suite2p_dir : str
        Path to suite2p output directory
    output_dir : str
        Directory to save outputs
    """
    selector = ROISelector(suite2p_dir)
    
    # Apply selection with fixed parameters
    new_iscell = selector.apply_selection_function(
        'select_by_roi_ellipticity_and_components',
        ellipticity_threshold=0.78,
        components_threshold=3,
        show_plot=False
    )
    
    # Save updated files
    np.save(Path(suite2p_dir) / 'iscell.npy', new_iscell)
    
    # Update selector's iscell attribute with new selections
    selector.iscell = new_iscell
    
    # Save ROI visualizations
    save_roi_visualizations(selector, output_dir)

def save_roi_visualizations(selector, output_dir):
    """Save ROI visualization plots."""
    # Get mean image
    mean_img = selector.ops['meanImg']
    
    # Create figure for selected and non-selected ROIs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot selected ROIs
    ax1.imshow(mean_img, cmap='gray')
    for i, roi in enumerate(selector.stat):
        if selector.iscell[i, 0]:
            ypix = roi['ypix']
            xpix = roi['xpix']
            ax1.plot(xpix, ypix, '.', markersize=1, alpha=0.5)
    ax1.set_title('Selected ROIs')
    ax1.axis('off')
    
    # Plot non-selected ROIs
    ax2.imshow(mean_img, cmap='gray')
    for i, roi in enumerate(selector.stat):
        if not selector.iscell[i, 0]:
            ypix = roi['ypix']
            xpix = roi['xpix']
            ax2.plot(xpix, ypix, '.', markersize=1, alpha=0.5)
    ax2.set_title('Non-selected ROIs')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'roi_selection.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_rasterplots(suite2p_dir, output_dir):
    """
    Generate rasterplots using suite2p visualization tools.
    
    Parameters:
    -----------
    suite2p_dir : str
        Path to suite2p output directory
    output_dir : str
        Directory to save outputs
    """
    # Load data
    F = np.load(Path(suite2p_dir) / 'F.npy')
    iscell = np.load(Path(suite2p_dir) / 'iscell.npy')
    
    # Get selected cells
    selected_cells = F[iscell[:, 0] == 1]
    
    # Create default rasterplot
    plt.figure(figsize=(15, 10))
    plt.imshow(selected_cells, aspect='auto', cmap='viridis')
    plt.colorbar(label='Fluorescence')
    plt.title('Rasterplot (Default Order)')
    plt.xlabel('Time')
    plt.ylabel('Cell Index')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rasterplot_default.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create rastermap-sorted rasterplot
    model = Rastermap()
    model.fit(selected_cells)
    sorted_cells = selected_cells[model.isort]
    
    # Save rastermap model
    rastermap_data = {
        'isort': model.isort,  # Sorting indices
        'embedding': model.embedding  # Final embedding
    }
    np.save(Path(suite2p_dir) / 'rastermap_model.npy', rastermap_data)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(sorted_cells, aspect='auto', cmap='viridis')
    plt.colorbar(label='Fluorescence')
    plt.title('Rasterplot (Rastermap Sorted)')
    plt.xlabel('Time')
    plt.ylabel('Cell Index')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rasterplot_sorted.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_pickle_file(suite2p_dir, output_dir):
    """Generate pickle file with selected traces data."""
    # Load data
    F = np.load(Path(suite2p_dir) / 'F.npy')
    iscell = np.load(Path(suite2p_dir) / 'iscell.npy')
    
    # Get selected cells
    selected_cells = F[iscell[:, 0] == 1]
    
    # Create and save pickle file with data
    data_dict = {
        'roi_indices': np.where(iscell[:, 0] == 1)[0],
        'traces': selected_cells
    }
    
    with open(Path(output_dir) / 'selected_traces.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

def main():
    """Main function to run the processing pipeline."""
    parser = argparse.ArgumentParser(description='Master script for data processing pipeline.')
    parser.add_argument('root_dir', type=str, help='Root directory containing channel folders')
    parser.add_argument('--overwrite', action='store_true', help='Force overwrite of existing files')
    args = parser.parse_args()
    
    # Find channel folders
    channel_folders = find_channel_folders(args.root_dir)
    
    if not channel_folders:
        print("No channel folders found!")
        return 1
    
    # Process each channel folder
    for folder_path, channel_type in channel_folders.items():
        print(f"\nProcessing {folder_path} (Channel {channel_type})...")
        start_time = time.time()
        
        try:
            # Find raw stack
            raw_stack_path = Path(folder_path) / 'suite2p files' / 'combined_registered.tif'
            if not raw_stack_path.exists():
                print(f"ERROR: Raw stack not found in {raw_stack_path}")
                print("Skipping this folder and continuing with next...")
                continue
            
            # Create output directories
            output_dir = Path(folder_path) / 'derippled'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Checkpoint 1: FFT Processing
            fft_outputs = [
                output_dir / "derippled_stack.tif",
                output_dir / "fft_amplitude.png",
                output_dir / "fft_amplitude_mean.png"
            ]
            
            if not args.overwrite and all(f.exists() and f.stat().st_size > 0 for f in fft_outputs):
                print("Skipping FFT processing - output files already exist")
                derippled_stack = str(fft_outputs[0])
            else:
                print("Starting FFT mask processing...")
                fft_start = time.time()
                derippled_stack = process_fft_masks(str(raw_stack_path), channel_type, output_dir)
                print(f"FFT mask processing completed in {timedelta(seconds=int(time.time() - fft_start))}")
            
            # Checkpoint 2: Suite2p Pipeline
            suite2p_dir = output_dir / 'suite2p'
            suite2p_outputs = [
                suite2p_dir / 'F.npy',
                suite2p_dir / 'iscell.npy'
            ]
            
            if not args.overwrite and all(f.exists() and f.stat().st_size > 0 for f in suite2p_outputs):
                print("Skipping suite2p pipeline - output files already exist")
            else:
                print("Starting suite2p pipeline...")
                suite2p_start = time.time()
                try:
                    suite2p_dir = run_suite2p_pipeline(derippled_stack, output_dir)
                    print(f"Suite2p pipeline completed in {timedelta(seconds=int(time.time() - suite2p_start))}")
                except Exception as e:
                    print(f"ERROR: Suite2p pipeline failed: {str(e)}")
                    print("Skipping this folder and continuing with next...")
                    continue
            
            # Checkpoint 3: ROI Selection
            roi_outputs = [
                suite2p_dir / 'iscell.npy',
                output_dir / 'roi_selection.png'
            ]
            
            if not args.overwrite and all(f.exists() and f.stat().st_size > 0 for f in roi_outputs):
                print("Skipping ROI selection - output files already exist")
            else:
                print("Starting ROI selection...")
                roi_start = time.time()
                try:
                    process_roi_selection(suite2p_dir, output_dir)
                    print(f"ROI selection completed in {timedelta(seconds=int(time.time() - roi_start))}")
                except Exception as e:
                    print(f"ERROR: ROI selection failed: {str(e)}")
                    print("Skipping this folder and continuing with next...")
                    continue
            
            # Checkpoint 4: Rasterplot Generation
            raster_outputs = [
                output_dir / 'rasterplot_default.png',
                output_dir / 'rasterplot_sorted.png',
                suite2p_dir / 'rastermap_model.npy'
            ]
            
            if not args.overwrite and all(f.exists() and f.stat().st_size > 0 for f in raster_outputs):
                print("Skipping rasterplot generation - output files already exist")
            else:
                print("Generating rasterplots...")
                raster_start = time.time()
                try:
                    generate_rasterplots(suite2p_dir, output_dir)
                    print(f"Rasterplot generation completed in {timedelta(seconds=int(time.time() - raster_start))}")
                except Exception as e:
                    print(f"ERROR: Rasterplot generation failed: {str(e)}")
                    print("Skipping this folder and continuing with next...")
                    continue
            
            # Checkpoint 5: Pickle File
            pickle_file = output_dir / 'selected_traces.pkl'
            
            if not args.overwrite and pickle_file.exists() and pickle_file.stat().st_size > 0:
                print("Skipping pickle file generation - file already exists")
            else:
                print("Generating pickle file...")
                pickle_start = time.time()
                try:
                    generate_pickle_file(suite2p_dir, output_dir)
                    print(f"Pickle file generation completed in {timedelta(seconds=int(time.time() - pickle_start))}")
                except Exception as e:
                    print(f"ERROR: Pickle file generation failed: {str(e)}")
                    print("Skipping this folder and continuing with next...")
                    continue
            
            total_time = time.time() - start_time
            print(f"\nProcessing complete for {folder_path}")
            print(f"Total processing time: {timedelta(seconds=int(total_time))}")
            
        except Exception as e:
            print(f"\nERROR: An error occurred while processing {folder_path}:")
            print(f"Error message: {str(e)}")
            print("Skipping this folder and continuing with next...")
            continue
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 