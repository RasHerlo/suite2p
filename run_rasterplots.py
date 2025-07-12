#!/usr/bin/env python3
"""
Standalone Rasterplot Generation Script

Generate rasterplots from suite2p output directory.

Usage:
    python run_rasterplots.py /path/to/suite2p/plane0 /path/to/output/dir
    python run_rasterplots.py /path/to/suite2p/plane0 /path/to/output/dir --no-rastermap
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rastermap.rastermap import Rastermap

def generate_rasterplots(suite2p_dir, output_dir, use_rastermap=True):
    """
    Generate rasterplots using suite2p output data.
    
    Parameters:
    -----------
    suite2p_dir : str
        Path to suite2p output directory (plane0 folder)
    output_dir : str
        Directory to save outputs
    use_rastermap : bool
        Whether to use rastermap for sorting
    """
    suite2p_dir = Path(suite2p_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required files
    required_files = ['F.npy', 'iscell.npy']
    missing_files = []
    
    for file in required_files:
        if not (suite2p_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files in {suite2p_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Load data
    print(f"Loading data from: {suite2p_dir}")
    F = np.load(suite2p_dir / 'F.npy')
    iscell = np.load(suite2p_dir / 'iscell.npy')
    
    # Get selected cells
    selected_cells = F[iscell[:, 0] == 1]
    
    print(f"Total ROIs: {len(F)}")
    print(f"Selected cells: {len(selected_cells)}")
    
    if len(selected_cells) == 0:
        print("ERROR: No cells selected! Check iscell.npy file.")
        return False
    
    # Create default rasterplot
    print("Creating default rasterplot...")
    plt.figure(figsize=(15, 10))
    plt.imshow(selected_cells, aspect='auto', cmap='viridis')
    plt.colorbar(label='Fluorescence')
    plt.title('Rasterplot (Default Order)')
    plt.xlabel('Time')
    plt.ylabel('Cell Index')
    plt.tight_layout()
    
    default_path = output_dir / 'rasterplot_default.png'
    plt.savefig(default_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Default rasterplot saved to: {default_path}")
    
    # Create rastermap-sorted rasterplot if requested
    if use_rastermap:
        print("Creating rastermap-sorted rasterplot...")
        try:
            model = Rastermap()
            model.fit(selected_cells)
            sorted_cells = selected_cells[model.isort]
            
            # Save rastermap model
            rastermap_data = {
                'isort': model.isort,  # Sorting indices
                'embedding': model.embedding  # Final embedding
            }
            rastermap_path = suite2p_dir / 'rastermap_model.npy'
            np.save(rastermap_path, rastermap_data)
            print(f"Rastermap model saved to: {rastermap_path}")
            
            plt.figure(figsize=(15, 10))
            plt.imshow(sorted_cells, aspect='auto', cmap='viridis')
            plt.colorbar(label='Fluorescence')
            plt.title('Rasterplot (Rastermap Sorted)')
            plt.xlabel('Time')
            plt.ylabel('Cell Index')
            plt.tight_layout()
            
            sorted_path = output_dir / 'rasterplot_sorted.png'
            plt.savefig(sorted_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Sorted rasterplot saved to: {sorted_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to create rastermap-sorted plot: {e}")
            print("Continuing with default plot only...")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate rasterplots from suite2p output')
    parser.add_argument('suite2p_dir', help='Path to suite2p output directory (plane0 folder)')
    parser.add_argument('output_dir', help='Directory to save rasterplot outputs')
    parser.add_argument('--no-rastermap', action='store_true',
                       help='Skip rastermap sorting (only create default plot)')
    
    args = parser.parse_args()
    
    success = generate_rasterplots(
        args.suite2p_dir, 
        args.output_dir,
        not args.no_rastermap
    )
    
    if success:
        print("Rasterplot generation completed successfully!")
        return 0
    else:
        print("Rasterplot generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 