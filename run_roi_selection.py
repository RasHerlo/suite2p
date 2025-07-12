#!/usr/bin/env python3
"""
Standalone ROI Selection Script

Run ROI selection on suite2p output directory.

Usage:
    python run_roi_selection.py /path/to/suite2p/plane0 /path/to/output/dir
    python run_roi_selection.py /path/to/suite2p/plane0 /path/to/output/dir --ellipticity 0.8 --components 2
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from roi_selection_new import ROISelector

def process_roi_selection(suite2p_dir, output_dir, ellipticity_threshold=0.78, components_threshold=3, show_plot=False):
    """
    Process ROI selection with specified parameters.
    
    Parameters:
    -----------
    suite2p_dir : str
        Path to suite2p output directory (plane0 folder)
    output_dir : str
        Directory to save outputs
    ellipticity_threshold : float
        Threshold for ellipticity filtering
    components_threshold : int
        Threshold for components filtering
    show_plot : bool
        Whether to show plots interactively
    """
    suite2p_dir = Path(suite2p_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if required files exist
    required_files = ['ops.npy', 'stat.npy', 'F.npy', 'Fneu.npy', 'iscell.npy']
    missing_files = []
    
    for file in required_files:
        if not (suite2p_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files in {suite2p_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print(f"Loading suite2p data from: {suite2p_dir}")
    selector = ROISelector(suite2p_dir)
    
    print(f"Original cells: {np.sum(selector.iscell[:, 0])}")
    
    # Apply selection with specified parameters
    print(f"Applying selection with ellipticity_threshold={ellipticity_threshold}, components_threshold={components_threshold}")
    new_iscell = selector.apply_selection_function(
        'select_by_roi_ellipticity_and_components',
        ellipticity_threshold=ellipticity_threshold,
        components_threshold=components_threshold,
        show_plot=show_plot
    )
    
    print(f"Selected cells: {np.sum(new_iscell[:, 0])}")
    
    # Save updated files
    print(f"Saving updated iscell.npy to: {suite2p_dir}")
    np.save(suite2p_dir / 'iscell.npy', new_iscell)
    
    # Update selector's iscell attribute with new selections
    selector.iscell = new_iscell
    
    # Save ROI visualizations
    print(f"Saving ROI visualizations to: {output_dir}")
    save_roi_visualizations(selector, output_dir, ellipticity_threshold, components_threshold)
    
    return True

def save_roi_visualizations(selector, output_dir, ellipticity_threshold, components_threshold):
    """Save ROI visualization plots."""
    # Get mean image
    mean_img = selector.ops['meanImg']
    
    # Create figure for selected and non-selected ROIs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot selected ROIs
    ax1.imshow(mean_img, cmap='gray')
    selected_count = 0
    for i, roi in enumerate(selector.stat):
        if selector.iscell[i, 0]:
            ypix = roi['ypix']
            xpix = roi['xpix']
            ax1.plot(xpix, ypix, '.', markersize=1, alpha=0.5)
            selected_count += 1
    ax1.set_title(f'Selected ROIs ({selected_count})')
    ax1.axis('off')
    
    # Plot non-selected ROIs
    ax2.imshow(mean_img, cmap='gray')
    non_selected_count = 0
    for i, roi in enumerate(selector.stat):
        if not selector.iscell[i, 0]:
            ypix = roi['ypix']
            xpix = roi['xpix']
            ax2.plot(xpix, ypix, '.', markersize=1, alpha=0.5)
            non_selected_count += 1
    ax2.set_title(f'Non-selected ROIs ({non_selected_count})')
    ax2.axis('off')
    
    # Add main title with parameters
    fig.suptitle(f'ROI Selection Results\nEllipticity ≤ {ellipticity_threshold}, Components ≥ {components_threshold}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'roi_selection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROI visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run ROI selection on suite2p output')
    parser.add_argument('suite2p_dir', help='Path to suite2p output directory (plane0 folder)')
    parser.add_argument('output_dir', help='Directory to save visualization outputs')
    parser.add_argument('--ellipticity', type=float, default=0.78, 
                       help='Ellipticity threshold (default: 0.78)')
    parser.add_argument('--components', type=int, default=3,
                       help='Components threshold (default: 3)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    success = process_roi_selection(
        args.suite2p_dir, 
        args.output_dir,
        args.ellipticity,
        args.components,
        args.show_plot
    )
    
    if success:
        print("ROI selection completed successfully!")
        return 0
    else:
        print("ROI selection failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 