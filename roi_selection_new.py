#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Selection Module for Suite2p
--------------------------------
This script provides functionality to classify ROIs as cells or non-cells
based on customizable criteria. It loads data from a Suite2p output folder 
and allows application of different selection functions.

Usage:
    python roi_selection_new.py path_to_plane0_folder

Author: [User]
Date: 2025-05-18
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button
import pandas as pd


class ROISelector:
    """Class for handling ROI selection based on various criteria."""
    
    def __init__(self, plane_dir):
        """Initialize with path to Suite2p plane directory."""
        self.plane_dir = Path(plane_dir)
        
        # Check if the directory exists and contains required files
        if not self.plane_dir.exists():
            raise FileNotFoundError(f"Directory not found: {plane_dir}")
        
        # Load Suite2p output files
        self.load_data()
        
        # Initialize results dictionary to store selection results
        self.selection_results = {}
    
    def load_data(self):
        """Load all relevant files from the Suite2p output directory."""
        print(f"Loading data from {self.plane_dir}...")
        
        # Required files
        required_files = ['stat.npy', 'iscell.npy', 'F.npy', 'Fneu.npy', 'spks.npy', 'ops.npy']
        
        # Check for existence of required files
        for file in required_files:
            if not (self.plane_dir / file).exists():
                print(f"Warning: {file} not found in {self.plane_dir}")
        
        # Load files
        try:
            self.stat = np.load(self.plane_dir / 'stat.npy', allow_pickle=True)
            self.iscell = np.load(self.plane_dir / 'iscell.npy', allow_pickle=True)
            self.Fcell = np.load(self.plane_dir / 'F.npy', allow_pickle=True)
            self.Fneu = np.load(self.plane_dir / 'Fneu.npy', allow_pickle=True)
            self.spks = np.load(self.plane_dir / 'spks.npy', allow_pickle=True)
            self.ops = np.load(self.plane_dir / 'ops.npy', allow_pickle=True).item()
            
            # Optional: Try to load redcell.npy if it exists
            if (self.plane_dir / 'redcell.npy').exists():
                self.redcell = np.load(self.plane_dir / 'redcell.npy', allow_pickle=True)
                print("Red channel data loaded.")
            
            print(f"Loaded {len(self.stat)} ROIs.")
            print(f"Current cell count: {np.sum(self.iscell[:, 0])}")
        
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_roi_stats(self):
        """Extract relevant statistics for each ROI."""
        stats_df = pd.DataFrame()
        
        # Extract common metrics from stat array
        stats_df['roi_id'] = np.arange(len(self.stat))
        stats_df['is_cell'] = self.iscell[:, 0].astype(bool)
        stats_df['cell_prob'] = self.iscell[:, 1]
        
        # Extract metrics from stat dictionary
        common_keys = ['npix', 'npix_norm', 'med', 'footprint', 'compact', 'aspect_ratio', 'radius', 'skew']
        for key in common_keys:
            try:
                stats_df[key] = [s.get(key, np.nan) for s in self.stat]
            except:
                pass
        
        # Calculate additional metrics
        stats_df['mean_Fcell'] = np.mean(self.Fcell, axis=1)
        stats_df['std_Fcell'] = np.std(self.Fcell, axis=1)
        stats_df['mean_Fneu'] = np.mean(self.Fneu, axis=1)
        stats_df['mean_spks'] = np.mean(self.spks, axis=1)
        
        # SNR calculation (simplified)
        stats_df['snr'] = stats_df['mean_Fcell'] / (stats_df['mean_Fneu'] + 1e-6)
        
        return stats_df
    
    def apply_selection_function(self, func_name, **kwargs):
        """Apply a selection function to classify ROIs."""
        # Get selection function
        if not hasattr(self, func_name):
            raise ValueError(f"Selection function {func_name} not found")
        
        selection_func = getattr(self, func_name)
        
        # Apply selection function
        print(f"Applying selection function: {func_name}")
        new_iscell = selection_func(**kwargs)
        
        # Store results
        self.selection_results[func_name] = {
            'iscell': new_iscell.copy(),
            'params': kwargs,
            'cell_count': np.sum(new_iscell[:, 0]),
        }
        
        print(f"Selection complete. New cell count: {np.sum(new_iscell[:, 0])}")
        return new_iscell
    
    def save_results(self, func_name=None, backup=True):
        """Save the selected cells back to iscell.npy."""
        if func_name is None:
            # Use the latest selection if none specified
            if not self.selection_results:
                print("No selection results to save.")
                return
            func_name = list(self.selection_results.keys())[-1]
        
        if func_name not in self.selection_results:
            print(f"No results found for selection function: {func_name}")
            return
        
        # Get the iscell array to save
        new_iscell = self.selection_results[func_name]['iscell']
        
        # Create backup if requested
        if backup:
            backup_path = self.plane_dir / f"iscell_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.npy"
            np.save(backup_path, self.iscell)
            print(f"Backup saved to: {backup_path}")
        
        # Save new iscell array
        np.save(self.plane_dir / 'iscell.npy', new_iscell)
        print(f"New iscell classification saved. Cell count: {np.sum(new_iscell[:, 0])}")
    
    def compare_selections(self, func_names=None):
        """Compare results of different selection functions."""
        if not self.selection_results:
            print("No selection results to compare.")
            return
        
        if func_names is None:
            func_names = list(self.selection_results.keys())
        
        # Create comparison dataframe
        results = {
            'original': {'cell_count': np.sum(self.iscell[:, 0]), 'params': 'Original'}
        }
        
        for func in func_names:
            if func in self.selection_results:
                results[func] = {
                    'cell_count': self.selection_results[func]['cell_count'],
                    'params': self.selection_results[func]['params']
                }
        
        # Print comparison
        print("\nSelection Results Comparison:")
        print("=" * 50)
        for name, res in results.items():
            print(f"{name}: {res['cell_count']} cells  |  Parameters: {res['params']}")
    
    # ===========================================
    # Selection functions below
    # ===========================================
    
    def select_by_roi_ellipticity_and_components(self, ellipticity_threshold=0.5, components_threshold=1, show_plot=True):
        """
        Select cells based on ROI ellipticity and number of connected components.
        
        ROIs are classified as cells if they have:
        1. Ellipticity below the ellipticity threshold, AND
        2. Number of connected components less than or equal to the components threshold
        
        Parameters:
        -----------
        ellipticity_threshold : float
            Threshold value (0-1) for ellipticity. Lower values are more circular.
        components_threshold : int
            Maximum number of connected components allowed.
        show_plot : bool
            Whether to display the interactive visualization.
            
        Returns:
        --------
        np.ndarray
            Updated iscell array where cells have been selected based on criteria.
        """
        print("Calculating ROI ellipticity...")
        ellipticity_values = self.calculate_roi_ellipticity()
        
        print("Calculating connected components...")
        component_counts = self.calculate_roi_components()
        
        # Save metrics
        np.save(self.plane_dir / 'roi_ellipticity.npy', ellipticity_values)
        np.save(self.plane_dir / 'roi_components.npy', component_counts)
        print(f"Metrics saved to {self.plane_dir}")
        
        # Create a copy of the iscell array to modify
        new_iscell = self.iscell.copy()
        
        # Initial classification based on both thresholds
        new_iscell[:, 0] = ((ellipticity_values < ellipticity_threshold) & 
                            (component_counts <= components_threshold)).astype(int)
        
        # Create an interactive plot if requested
        if show_plot:
            # Use a global variable to store the updated iscell array after user interaction
            self.interactive_iscell = new_iscell.copy()
            
            # Create interactive plot for threshold selection
            self.visualize_roi_metrics(ellipticity_values, component_counts, 
                                      ellipticity_threshold, components_threshold)
            
            # Return the interactively updated iscell array
            return self.interactive_iscell
        
        return new_iscell
    
    def calculate_roi_ellipticity(self):
        """
        Calculate ellipticity for each ROI in the stat array.
        
        Ellipticity is calculated as 1 - (minor_axis / major_axis),
        where axes are derived from eigenvalues of the covariance matrix
        of pixel coordinates.
        
        Returns:
        --------
        np.ndarray
            Array of ellipticity values (0-1) for each ROI.
        """
        ellipticity = np.zeros(len(self.stat))
        
        for i, roi in enumerate(tqdm(self.stat, desc="Calculating ellipticity")):
            # Get pixel coordinates
            y_coords = roi['ypix']
            x_coords = roi['xpix']
            
            # Stack coordinates into a 2D array
            if len(y_coords) > 1:  # Ensure we have at least 2 pixels
                coords = np.vstack((y_coords, x_coords)).T
                
                # Calculate covariance matrix
                cov_matrix = np.cov(coords, rowvar=False)
                
                # Calculate eigenvalues
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues in descending order
                eigenvalues = sorted(eigenvalues, reverse=True)
                
                # Calculate ellipticity (ensure we don't divide by zero)
                if eigenvalues[1] > 0:
                    # Ellipticity: 0 = circle, 1 = line
                    ellipticity[i] = 1 - (np.sqrt(eigenvalues[1]) / np.sqrt(eigenvalues[0]))
                else:
                    # Handle degenerate case
                    ellipticity[i] = 1.0
            else:
                # For single-pixel ROIs (should be rare)
                ellipticity[i] = 0.0  # Assume circular
        
        return ellipticity
    
    def calculate_roi_components(self):
        """
        Calculate number of connected components for each ROI.
        
        Returns:
        --------
        np.ndarray
            Array of component counts for each ROI.
        """
        print("Calculating connected components...")
        component_counts = np.zeros(len(self.stat), dtype=int)
        
        for i, roi in enumerate(self.stat):
            # Get ROI pixels
            ypix = roi['ypix']
            xpix = roi['xpix']
            
            # Create binary mask
            mask = np.zeros((self.ops['Ly'], self.ops['Lx']), dtype=bool)
            mask[ypix, xpix] = True
            
            # Label connected components
            from skimage.measure import label
            labeled = label(mask)
            
            # Count components, excluding those with 2 or fewer pixels
            unique_labels = np.unique(labeled[labeled > 0])
            valid_components = 0
            for label_id in unique_labels:
                component_size = np.sum(labeled == label_id)
                if component_size > 2:  # Only count components with more than 2 pixels
                    valid_components += 1
            
            component_counts[i] = valid_components
        
        return component_counts
    
    def visualize_roi_metrics(self, ellipticity_values, component_counts, 
                             initial_ellipticity_threshold=0.5, initial_components_threshold=1):
        """
        Create an interactive visualization of ROI metrics with threshold sliders.
        
        Parameters:
        -----------
        ellipticity_values : np.ndarray
            Array of ellipticity values for each ROI.
        component_counts : np.ndarray
            Array of connected component counts for each ROI.
        initial_ellipticity_threshold : float
            Initial threshold value for ellipticity.
        initial_components_threshold : int
            Initial threshold value for component counts.
        """
        # Get mean image for background
        if 'meanImg' in self.ops:
            mean_img = self.ops['meanImg']
        else:
            # Create a blank background if mean image not available
            mean_img = np.zeros((self.ops['Ly'], self.ops['Lx']))
        
        # Create color map for ROIs
        n_colors = 20  # Number of distinct colors
        cmap = plt.cm.get_cmap('tab20', n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        
        # Define figure and grid for subplots
        plt.rcParams['figure.figsize'] = [16, 9]
        fig = plt.figure(constrained_layout=True)
        grid = fig.add_gridspec(3, 8)
        
        # Create subplots according to specified layout
        # Top row: ellipticity histogram
        ax_ellipticity = fig.add_subplot(grid[0, 0:2])
        
        # Middle row: components histogram (log scale)
        ax_components = fig.add_subplot(grid[1, 0:2])
        
        # ROI visualizations
        # Selected cells
        ax_selected = fig.add_subplot(grid[0:3, 2:5])
        
        # Non-selected cells
        ax_nonselected = fig.add_subplot(grid[0:3, 5:8])
        
        # Add space at the bottom for sliders and text boxes
        plt.subplots_adjust(bottom=0.25)
        
        # Add sliders for thresholds
        # Ellipticity slider
        ax_ellipticity_slider = plt.axes([0.15, 0.15, 0.6, 0.03])
        ellipticity_slider = Slider(
            ax=ax_ellipticity_slider,
            label='Ellipticity Threshold',
            valmin=0.0,
            valmax=1.0,
            valinit=initial_ellipticity_threshold,
            valstep=0.01
        )
        
        # Ellipticity text box
        ax_ellipticity_text = plt.axes([0.77, 0.15, 0.08, 0.03])
        ellipticity_text = TextBox(
            ax_ellipticity_text,
            '',
            initial=f'{initial_ellipticity_threshold:.2f}',
            color='white',
            hovercolor='white'
        )
        
        # Components slider
        ax_components_slider = plt.axes([0.15, 0.1, 0.6, 0.03])
        components_slider = Slider(
            ax=ax_components_slider,
            label='Max Connected Components',
            valmin=1,
            valmax=min(10, np.max(component_counts)),  # Cap at 10 or max value
            valinit=initial_components_threshold,
            valstep=1
        )
        
        # Components text box
        ax_components_text = plt.axes([0.77, 0.1, 0.08, 0.03])
        components_text = TextBox(
            ax_components_text,
            '',
            initial=str(initial_components_threshold),
            color='white',
            hovercolor='white'
        )
        
        # Add save button
        ax_button = plt.axes([0.75, 0.03, 0.2, 0.05])
        save_button = Button(ax_button, 'Save Selection')
        
        # Define function to update plots when thresholds change
        def update(val=None):
            # Get current threshold values
            ellipticity_threshold = ellipticity_slider.val
            components_threshold = components_slider.val
            
            # Update iscell array based on both criteria
            self.interactive_iscell[:, 0] = ((ellipticity_values < ellipticity_threshold) & 
                                           (component_counts <= components_threshold)).astype(int)
            
            # Get cell and non-cell indices
            cell_idx = np.where(self.interactive_iscell[:, 0] == 1)[0]
            noncell_idx = np.where(self.interactive_iscell[:, 0] == 0)[0]
            
            # Clear axes for redrawing
            for ax in [ax_ellipticity, ax_components, ax_selected, ax_nonselected]:
                ax.clear()
            
            # Plot ellipticity histogram
            ax_ellipticity.hist(ellipticity_values, bins=30, color='gray', alpha=0.7)
            ax_ellipticity.axvline(x=ellipticity_threshold, color='red', linestyle='--', linewidth=2)
            ax_ellipticity.set_title(f'ROI Ellipticity (N={len(ellipticity_values)})')
            ax_ellipticity.set_ylabel('Count')
            ax_ellipticity.set_xlabel('Ellipticity')
            
            # Plot components histogram with log y-scale
            # Bin edges for components (integers)
            bin_edges = np.arange(0.5, np.max(component_counts) + 1.5, 1)
            ax_components.hist(component_counts, bins=bin_edges, color='gray', alpha=0.7)
            ax_components.axvline(x=components_threshold + 0.5, color='red', linestyle='--', linewidth=2)
            ax_components.set_title(f'Connected Components (N={len(component_counts)})')
            ax_components.set_ylabel('Count (log scale)')
            ax_components.set_xlabel('Number of Components')
            ax_components.set_yscale('log')
            
            # Plot ROIs on mean image
            # For selected cells
            ax_selected.imshow(mean_img, cmap='gray')
            for i, idx in enumerate(cell_idx):
                roi = self.stat[idx]
                color_idx = i % n_colors
                ypix = roi['ypix']
                xpix = roi['xpix']
                
                # Plot filled ROI
                for y, x in zip(ypix, xpix):
                    ax_selected.plot(x, y, '.', color=colors[color_idx], markersize=1, alpha=0.5)
                
                # Plot boundary for better visibility
                if 'yext' in roi and 'xext' in roi:
                    ax_selected.plot(roi['xext'], roi['yext'], color=colors[color_idx], linewidth=1)
                
            ax_selected.set_title(f'Selected Cells (N={len(cell_idx)})')
            ax_selected.axis('off')
            
            # For non-selected cells
            ax_nonselected.imshow(mean_img, cmap='gray')
            for i, idx in enumerate(noncell_idx):
                roi = self.stat[idx]
                color_idx = i % n_colors
                ypix = roi['ypix']
                xpix = roi['xpix']
                
                # Plot filled ROI
                for y, x in zip(ypix, xpix):
                    ax_nonselected.plot(x, y, '.', color=colors[color_idx], markersize=1, alpha=0.5)
                
                # Plot boundary for better visibility
                if 'yext' in roi and 'xext' in roi:
                    ax_nonselected.plot(roi['xext'], roi['yext'], color=colors[color_idx], linewidth=1)
                
            ax_nonselected.set_title(f'Non-selected Cells (N={len(noncell_idx)})')
            ax_nonselected.axis('off')
            
            # Display metrics about selection
            ellipticity_pass = np.sum(ellipticity_values < ellipticity_threshold)
            components_pass = np.sum(component_counts <= components_threshold)
            both_pass = np.sum((ellipticity_values < ellipticity_threshold) & 
                              (component_counts <= components_threshold))
            
            info_text = (f"Ellipticity < {ellipticity_threshold:.2f}: {ellipticity_pass} ROIs\n"
                        f"Components ≤ {components_threshold}: {components_pass} ROIs\n"
                        f"Both criteria: {both_pass} ROIs")
            
            plt.figtext(0.15, 0.01, info_text, ha='left')
            
            # Refresh the figure
            fig.canvas.draw_idle()
        
        # Define function to handle ellipticity text box submission
        def submit_ellipticity(text):
            try:
                value = float(text)
                if 0 <= value <= 1:
                    ellipticity_slider.set_val(value)
                    ellipticity_text.set_color('black')
                else:
                    ellipticity_text.set_color('red')
            except ValueError:
                ellipticity_text.set_color('red')
        
        # Define function to handle components text box submission
        def submit_components(text):
            try:
                value = int(float(text))
                if 1 <= value <= min(10, np.max(component_counts)):
                    components_slider.set_val(value)
                    components_text.set_color('black')
                else:
                    components_text.set_color('red')
            except ValueError:
                components_text.set_color('red')
        
        # Define function to save selection
        def save_selection(event):
            # Save the current selection
            np.save(self.plane_dir / 'iscell.npy', self.interactive_iscell)
            print(f"Selection saved. Cell count: {np.sum(self.interactive_iscell[:, 0])}")
            # Add a notification on the plot
            plt.figtext(0.5, 0.01, "Selection saved!", 
                      ha="center", color="green", weight="bold")
            fig.canvas.draw_idle()
        
        # Connect callbacks
        ellipticity_slider.on_changed(update)
        components_slider.on_changed(update)
        ellipticity_text.on_submit(submit_ellipticity)
        components_text.on_submit(submit_components)
        save_button.on_clicked(save_selection)
        
        # Initial update
        update()
        
        # Show the figure
        plt.tight_layout()
        plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ROI Selection Tool for Suite2p outputs.')
    parser.add_argument('plane_dir', type=str, help='Path to Suite2p plane directory (e.g., plane0)')
    return parser.parse_args()


def main():
    """Main function to run the script."""
    args = parse_args()
    
    # Initialize ROI selector
    try:
        selector = ROISelector(args.plane_dir)
        print("\nROI Selector initialized successfully.")
        
        # Automatically apply the ellipticity and components-based selection
        print("\nApplying ROI selection based on ellipticity and connected components...")
        selector.apply_selection_function('select_by_roi_ellipticity_and_components', 
                                         ellipticity_threshold=0.5,
                                         components_threshold=1)
        
        # The function above will display the interactive plot if available
    except Exception as e:
        print(f"Error in ROI selection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 