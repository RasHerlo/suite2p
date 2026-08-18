#!/usr/bin/env python
# FFT_mask_conversion.py
# 
# This script takes a .tif stack and applies one or more rectangular masks to its FFT
# to remove noise patterns. It saves the denoised stack as a new .tif file.
#
# Input parameters:
#    - Full path to .tif file
#    - One or more mask coordinates in bracket notation: [x0 y0 dX dY] where:
#      * x0, y0 are center coordinates in FFT space (0,0 is the center of frequency domain)
#      * dX, dY are dimensions of the mask extending in each direction
#    - Example: [0 0 7 7] or [0 0 100 0] [0 0 0 100]
#    OR
#    - One or more circular/ring masks in parenthesis notation: (inner_radius,outer_radius)
#    - Example: (0,45) for a circle or (7,45) for a ring
#
# Outputs:
#    - Denoised .tif stack (saved as original_name_masked.tif)
#    - Mean FFT image with masked areas highlighted in purple

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread, imwrite
import colorcet as cc
import matplotlib.patches as patches
import re

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply FFT mask to a .tif stack for noise removal.')
    
    parser.add_argument('input_path', type=str, 
                        help='Full path to the .tif file')
    
    # Create a mutually exclusive group for mask type
    mask_group = parser.add_mutually_exclusive_group(required=True)
    
    mask_group.add_argument('--rect', dest='rect_masks', type=str, nargs='+',
                        help='One or more rectangular mask coordinates in bracket notation: [x0 y0 dX dY]. '
                             'Example: [0 0 7 7] or [0 0 100 0] [0 0 0 100]')
    
    mask_group.add_argument('--circle', dest='circle_masks', type=str, nargs='+',
                        help='One or more circular/ring mask coordinates in parenthesis notation: (inner_radius,outer_radius). '
                             'Example: (0,45) for a circle or (7,45) for a ring')
    
    return parser.parse_args()

def parse_mask_brackets(mask_brackets):
    """
    Parse mask coordinates from bracket notation.
    
    Parameters:
    - mask_brackets: List of strings with bracket notation like "[0 0 7 7]"
    
    Returns:
    - List of mask coordinates as [x0, y0, dX, dY]
    """
    masks = []
    
    for bracket in mask_brackets:
        # Extract values from bracket notation using regex
        match = re.match(r'\[([^]]+)\]', bracket)
        if match:
            # Extract the values and convert to float
            values = match.group(1).split()
            if len(values) == 4:
                masks.append([float(val) for val in values])
            else:
                print(f"Warning: Ignoring invalid mask format: {bracket}")
                continue
        else:
            print(f"Warning: Ignoring invalid mask format: {bracket}")
            continue
    
    return masks

def parse_circular_masks(mask_strings):
    """
    Parse circular mask coordinates from parenthesis notation.
    
    Parameters:
    - mask_strings: List of strings with parenthesis notation like "(0,45)"
    
    Returns:
    - List of mask coordinates as [inner_radius, outer_radius]
    """
    masks = []
    
    for mask_str in mask_strings:
        # Extract values using regex
        match = re.match(r'\(([^)]+)\)', mask_str)
        if match:
            # Extract the values and convert to float
            values = match.group(1).split(',')
            if len(values) == 2:
                inner = float(values[0])
                outer = float(values[1])
                if inner < outer:  # Validation
                    masks.append([inner, outer])
                else:
                    print(f"Warning: Inner radius must be less than outer radius: {mask_str}")
            else:
                print(f"Warning: Invalid circular mask format: {mask_str}")
        else:
            print(f"Warning: Invalid circular mask format: {mask_str}")
    
    return masks

def create_rectangular_mask(shape, mask_coords_list):
    """
    Create a binary mask for FFT filtering from multiple rectangular coordinate sets.
    
    Parameters:
    - shape: Shape of the FFT array (num_frames, height, width)
    - mask_coords_list: List of [x0, y0, dX, dY] coordinates for masks in FFT space
                        where (0,0) is the center of the frequency domain
    
    Returns:
    - Binary mask (1=keep, 0=mask out) with shape (height, width)
    """
    _, h, w = shape
    
    # Create a mask filled with ones (keep everything by default)
    mask = np.ones((h, w), dtype=bool)
    
    # Get the center of the FFT
    cy, cx = h // 2, w // 2
    
    # Apply each mask set
    for mask_coords in mask_coords_list:
        # Extract mask coordinates
        x0, y0, dx, dy = mask_coords
        
        # Convert from FFT coordinates (center is 0,0) to array indices
        x_min = int(cx + x0 - dx)
        x_max = int(cx + x0 + dx)
        y_min = int(cy + y0 - dy)
        y_max = int(cy + y0 + dy)
        
        # Ensure coordinates are within bounds
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        # Create the mask (0 in the rectangle to be removed)
        mask[y_min:y_max, x_min:x_max] = False
    
    return mask

def create_circular_mask(shape, mask_coords_list):
    """
    Create a binary mask for FFT filtering from multiple circular/ring coordinate sets.
    
    Parameters:
    - shape: Shape of the FFT array (num_frames, height, width)
    - mask_coords_list: List of [inner_radius, outer_radius] coordinates for circular masks
    
    Returns:
    - Binary mask (1=keep, 0=mask out) with shape (height, width)
    """
    _, h, w = shape
    
    # Create a mask filled with ones (keep everything by default)
    mask = np.ones((h, w), dtype=bool)
    
    # Get the center of the FFT
    cy, cx = h // 2, w // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Convert to coordinates relative to center
    y = y - cy
    x = x - cx
    
    # Calculate squared distances for efficiency
    dist_squared = x*x + y*y
    
    # Apply each mask set
    for mask_coords in mask_coords_list:
        # Extract mask coordinates
        inner_radius, outer_radius = mask_coords
        
        # Create the mask (True for areas to keep, False for areas to mask out)
        # We want to mask out areas where inner_radius^2 <= dist_squared <= outer_radius^2
        ring_mask = (dist_squared < inner_radius**2) | (dist_squared > outer_radius**2)
        
        # Apply this mask (keep only where both masks are True)
        mask = mask & ring_mask
    
    return mask

def save_fft_visualization(fft_mean, mask, output_path, mask_coords_list, mask_type):
    """
    Save a visualization of the mean FFT with the masked areas highlighted.
    
    Parameters:
    - fft_mean: Mean of the log-amplitude FFT
    - mask: Binary mask (1=keep, 0=mask out)
    - output_path: Path to save the visualization image
    - mask_coords_list: List of mask coordinates
    - mask_type: Type of mask ('rect' or 'circle')
    """
    # Create a new figure
    plt.figure(figsize=(10, 8))
    
    # Create a copy of the FFT mean for visualization
    fft_vis = np.log(fft_mean + 1e-6)
    
    # Scale between 0 and 1 for better visualization
    fft_min = np.percentile(fft_vis, 1)
    fft_max = np.percentile(fft_vis, 99)
    fft_vis = (fft_vis - fft_min) / (fft_max - fft_min)
    
    # Create an RGB image
    rgb_img = np.zeros((*fft_vis.shape, 3))
    
    # Set the FFT as the brightness in grayscale
    rgb_img[:, :, 0] = fft_vis
    rgb_img[:, :, 1] = fft_vis
    rgb_img[:, :, 2] = fft_vis
    
    # Highlight the masked area in purple
    # Use ~mask to get the masked-out areas
    rgb_img[~mask, 0] = 0.8  # Red component
    rgb_img[~mask, 1] = 0.0  # Green component
    rgb_img[~mask, 2] = 0.8  # Blue component
    
    # Format mask coordinates for title based on mask type
    if mask_type == 'rect':
        mask_str = ' '.join([f'[{int(x0)},{int(y0)},{int(dx)},{int(dy)}]' for x0, y0, dx, dy in mask_coords_list])
    else:  # circle
        mask_str = ' '.join([f'({int(inner)},{int(outer)})' for inner, outer in mask_coords_list])
    
    # Display the image with mask coordinates in title
    plt.imshow(rgb_img)
    plt.title(f'Mean FFT Amplitude with masks: {mask_str}')
    plt.colorbar(label='Log Amplitude')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_tif_stack(input_path, mask_coords_list, mask_type):
    """
    Process a .tif stack by applying FFT masks.
    
    Parameters:
    - input_path: Full path to the .tif file
    - mask_coords_list: List of mask coordinates
    - mask_type: Type of mask ('rect' or 'circle')
    """
    # Ensure the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return False
    
    # Get directory and filename
    dir_path = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    
    # Create a mask identifier string for filenames based on mask type
    num_masks = len(mask_coords_list)
    mask_id = ""
    
    if num_masks > 0:
        if mask_type == 'rect':
            # For rectangular masks - use existing format
            x0, y0, dx, dy = mask_coords_list[0]
            first_mask_str = f"{abs(int(x0))}{abs(int(y0))}{abs(int(dx))}{abs(int(dy))}"
            mask_id = f"{num_masks}_{first_mask_str}"
        else:  # circle
            # For circular masks - use i{inner}_o{outer} format for each mask
            mask_parts = []
            for inner, outer in mask_coords_list:
                mask_parts.append(f"_i{int(inner)}_o{int(outer)}")
            mask_id = ''.join(mask_parts)
    else:
        mask_id = "0_none"
    
    # Generate output file paths with mask coordinates
    base_name = os.path.splitext(file_name)[0]
    output_tif = os.path.join(dir_path, f"{base_name}_masked{mask_id}.tif")
    output_fft_img = os.path.join(dir_path, f"{base_name}_fft_mask{mask_id}.png")
    
    # Load the .tif stack
    print(f"Loading {input_path}...")
    images = imread(input_path, is_ome=False)
    
    # Normalize if necessary
    if images.min() < 0:
        images -= images.min()
    
    num_frames = images.shape[0]
    print(f"Loaded {num_frames} frames with shape {images.shape[1:]}.")
    print(f"Applying {len(mask_coords_list)} {mask_type} mask(s)")
    
    # Compute FFT for all frames
    print("Computing FFT...")
    fft_images = np.fft.fft2(images)
    fft_images = np.fft.fftshift(fft_images)
    
    # Compute the mean amplitude for visualization
    fft_amplitude = np.abs(fft_images)
    fft_mean = np.mean(fft_amplitude, axis=0)
    
    # Create the binary mask based on mask type
    print(f"Creating {mask_type} masks...")
    if mask_type == 'rect':
        mask = create_rectangular_mask(images.shape, mask_coords_list)
    else:  # circle
        mask = create_circular_mask(images.shape, mask_coords_list)
    
    # Apply the mask to all frames
    # We're using broadcasting to apply the 2D mask to all frames
    print("Applying masks to FFT...")
    masked_fft = fft_images.copy()
    masked_fft[:, ~mask] = 0
    
    # Inverse FFT to get denoised images
    print("Computing inverse FFT...")
    reconstructed_images = np.fft.ifftshift(masked_fft)
    reconstructed_images = np.fft.ifft2(reconstructed_images)
    denoised_images = np.abs(reconstructed_images)
    
    # Save the denoised images
    print(f"Saving denoised images to {output_tif}...")
    imwrite(output_tif, denoised_images.astype(np.float32))
    
    # Save FFT visualization with mask
    print(f"Saving FFT visualization to {output_fft_img}...")
    save_fft_visualization(fft_mean, mask, output_fft_img, mask_coords_list, mask_type)
    
    print("Processing complete!")
    return True

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Determine mask type and parse mask coordinates
    if args.rect_masks:
        mask_type = 'rect'
        mask_coords_list = parse_mask_brackets(args.rect_masks)
    else:
        mask_type = 'circle'
        mask_coords_list = parse_circular_masks(args.circle_masks)
    
    if not mask_coords_list:
        print(f"Error: No valid {mask_type} mask coordinates provided.")
        return
    
    # Process the .tif stack
    process_tif_stack(args.input_path, mask_coords_list, mask_type)

if __name__ == "__main__":
    main() 