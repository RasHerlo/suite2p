import os
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider

"""
This script processes neural trace data from suite2p output files.
The input directory should be a 'plane0' folder containing the suite2p output files (.npy files).
Expected files in the directory:
    - F.npy: Raw fluorescence traces
    - Other .npy files from suite2p processing
"""

def get_stimulation_range(shape):
    """
    Determine stimulation range based on the shape of F.npy.
    
    Args:
        shape (tuple): Shape of the F.npy array
        
    Returns:
        tuple: (start_frame, end_frame) for stimulation range, or None if shape not recognized
    """
    if shape[1] == 1520:
        return (726, 733)
    elif shape[1] == 2890:
        return (1381, 1388)
    return None

def normalize_traces(F):
    """
    Normalize each row of the F matrix from 0 to 1.
    
    Args:
        F (numpy.ndarray): Input matrix
        
    Returns:
        numpy.ndarray: Normalized matrix
    """
    F_norm = np.zeros_like(F)
    for i in range(F.shape[0]):
        row = F[i, :]
        valid_data = row[~np.isnan(row)]
        if len(valid_data) > 0:
            min_val = np.nanmin(row)
            max_val = np.nanmax(row)
            if max_val > min_val:  # Only normalize if there's a range
                F_norm[i, :] = (row - min_val) / (max_val - min_val)
            else:
                F_norm[i, :] = row  # Keep original values if no range
    return F_norm

def plot_traces(F_processed, save_path=None):
    """
    Create a figure with four subplots showing the processed F matrix, its normalized version,
    and their respective mean traces.
    
    Args:
        F_processed (numpy.ndarray): Processed F matrix with NaNs
        save_path (str, optional): Path to save the figure
    """
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original processed F matrix
    im1 = ax1.imshow(F_processed, aspect='auto', cmap='jet')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Neurons')
    ax1.set_title('Raw Traces')
    plt.colorbar(im1, ax=ax1, label='Fluorescence')
    
    # Plot normalized version
    F_norm = normalize_traces(F_processed)
    im2 = ax2.imshow(F_norm, aspect='auto', cmap='jet')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Neurons')
    ax2.set_title('Norm Traces')
    plt.colorbar(im2, ax=ax2, label='Normalized Fluorescence')
    
    # Plot mean of raw traces with gaps for NaN values
    # Calculate mean only where we have valid data
    mean_raw = np.zeros(F_processed.shape[1])
    for i in range(F_processed.shape[1]):
        valid_data = F_processed[:, i][~np.isnan(F_processed[:, i])]
        if len(valid_data) > 0:
            mean_raw[i] = np.mean(valid_data)
        else:
            mean_raw[i] = np.nan
    
    masked_raw = np.ma.masked_invalid(mean_raw)
    ax3.plot(masked_raw, 'k-', linewidth=1)
    ax3.set_xlabel('Frames')
    ax3.set_ylabel('Mean Fluorescence')
    ax3.set_title('Mean Raw Traces')
    ax3.grid(True, alpha=0.3)
    
    # Plot mean of normalized traces with gaps for NaN values
    mean_norm = np.zeros(F_norm.shape[1])
    for i in range(F_norm.shape[1]):
        valid_data = F_norm[:, i][~np.isnan(F_norm[:, i])]
        if len(valid_data) > 0:
            mean_norm[i] = np.mean(valid_data)
        else:
            mean_norm[i] = np.nan
    
    masked_norm = np.ma.masked_invalid(mean_norm)
    ax4.plot(masked_norm, 'k-', linewidth=1)
    ax4.set_xlabel('Frames')
    ax4.set_ylabel('Mean Normalized Fluorescence')
    ax4.set_title('Mean Norm Traces')
    ax4.grid(True, alpha=0.3)
    
    # Add vertical lines to mark stimulation period
    stim_range = get_stimulation_range(F_processed.shape)
    if stim_range is not None:
        start_frame, end_frame = stim_range
        for ax in [ax3, ax4]:
            ax.axvspan(start_frame, end_frame, color='red', alpha=0.2, label='Stimulation')
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return F_norm  # Return normalized matrix for PCA

def plot_pca_components(F_matrix, save_path=None, is_normalized=True):
    """
    Create a figure showing the matrix and top 5 principal components.
    
    Args:
        F_matrix (numpy.ndarray): F matrix (raw or normalized)
        save_path (str, optional): Path to save the figure
        is_normalized (bool): Whether the matrix is normalized or raw
    """
    # Handle NaN values for PCA
    F_no_nan = np.nan_to_num(F_matrix, nan=0)
    
    # Perform PCA on neuron dimension
    pca = PCA(n_components=5)
    pca.fit(F_no_nan)  # Each row is a neuron, each column is a time point
    components = pca.components_  # These are now the time components
    explained_variance = pca.explained_variance_ratio_
    
    # Create figure
    fig, axs = plt.subplots(6, 1, figsize=(15, 12))
    
    # Plot matrix
    im = axs[0].imshow(F_matrix, aspect='auto', cmap='jet')
    axs[0].set_xlabel('Frames')
    axs[0].set_ylabel('Neurons')
    matrix_type = 'Normalized' if is_normalized else 'Raw'
    axs[0].set_title(f'{matrix_type} Traces')
    plt.colorbar(im, ax=axs[0], label='Fluorescence')
    
    # Plot top 5 principal components
    for i in range(5):
        axs[i+1].plot(components[i], 'k-', linewidth=1)
        axs[i+1].set_title(f'PC{i+1} - Explained Variance = {explained_variance[i]:.3f}')
        axs[i+1].grid(True, alpha=0.3)
        
        # Add stimulation period if applicable
        stim_range = get_stimulation_range(F_matrix.shape)
        if stim_range is not None:
            start_frame, end_frame = stim_range
            axs[i+1].axvspan(start_frame, end_frame, color='red', alpha=0.2, label='Stimulation')
            axs[i+1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA figure saved to {save_path}")
    
    plt.show()

def mask_stimulation_range(data):
    """
    Mask the stimulation range in F.npy with NaNs.
    
    Args:
        data (dict): Dictionary containing loaded numpy arrays
        
    Returns:
        numpy.ndarray: Processed F array with stimulation range masked
    """
    if 'F' not in data:
        print("Error: F.npy not found in the loaded data")
        return None
        
    F = data['F']
    stim_range = get_stimulation_range(F.shape)
    
    if stim_range is None:
        print(f"Warning: Unrecognized F.npy shape {F.shape}. Skipping processing.")
        return None
        
    start_frame, end_frame = stim_range
    print(f"\nF.npy shape: {F.shape}")
    print(f"Stimulation range set to frames {start_frame}-{end_frame}")
    
    # Create a copy of F to avoid modifying the original
    F_processed = F.copy()
    F_processed[:, start_frame:end_frame+1] = np.nan
    
    return F_processed

def load_npy_files(directory):
    """
    Load all .npy files from the specified directory.
    
    Args:
        directory (str): Path to the directory containing .npy files
        
    Returns:
        dict: Dictionary containing loaded numpy arrays with filenames as keys
    """
    data = {}
    directory = Path(directory)
    
    print(f"Looking for .npy files in: {directory}")
    
    # Find all .npy files in the directory
    npy_files = list(directory.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return data
    
    # Load each .npy file
    for file_path in npy_files:
        try:
            file_name = file_path.stem
            data[file_name] = np.load(str(file_path), allow_pickle=True)
            print(f"Loaded {file_name}.npy with shape: {data[file_name].shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return data

class InteractiveComponentViewer:
    def __init__(self, F_norm, pca, component_idx):
        self.F_norm = F_norm  # Normalized matrix
        self.pca = pca  # Fitted PCA object
        self.component_idx = component_idx  # Which PC we're viewing
        self.n_neurons = F_norm.shape[0]
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, 
            figsize=(15, 12), 
            height_ratios=[1, 1, 3]
        )
        
        # Add slider
        self.slider_ax = plt.axes([0.92, 0.2, 0.03, 0.6])  # Position to the right of bottom subplot
        self.slider = Slider(
            self.slider_ax,
            'Neuron',
            0,
            self.n_neurons - 1,
            valinit=0,
            valstep=1
        )
        
        # Initialize plots
        self._init_plots()
        
        # Connect slider to update function
        self.slider.on_changed(self._update)
        
    def _init_plots(self):
        # Plot PC
        self.ax1.plot(self.pca.components_[self.component_idx], 'k-', linewidth=1)
        self.ax1.set_title(f'PC{self.component_idx+1} - Explained Variance = {self.pca.explained_variance_ratio_[self.component_idx]:.3f}')
        self.ax1.grid(True, alpha=0.3)
        
        # Add stimulation period if applicable
        stim_range = get_stimulation_range(self.F_norm.shape)
        if stim_range is not None:
            start_frame, end_frame = stim_range
            self.ax1.axvspan(start_frame, end_frame, color='red', alpha=0.2, label='Stimulation')
            self.ax1.legend()
        
        # Plot initial selected neuron
        self.neuron_line, = self.ax2.plot(self.F_norm[0], 'b-', linewidth=1)
        self.ax2.set_title('Selected Neuron Trace')
        self.ax2.grid(True, alpha=0.3)
        
        # Add stimulation period to neuron trace
        if stim_range is not None:
            self.ax2.axvspan(start_frame, end_frame, color='red', alpha=0.2, label='Stimulation')
            self.ax2.legend()
        
        # Plot sorted matrix
        self.im = self.ax3.imshow(self.F_norm, aspect='auto', cmap='jet')
        self.ax3.set_xlabel('Frames')
        self.ax3.set_ylabel('Neurons (sorted by PC contribution)')
        self.ax3.set_title(f'Traces Sorted by PC{self.component_idx+1} Contribution')
        plt.colorbar(self.im, ax=self.ax3, label='Normalized Fluorescence')
        
        plt.tight_layout()
        
    def _update(self, val):
        # Update selected neuron plot
        neuron_idx = int(val)
        self.neuron_line.set_ydata(self.F_norm[neuron_idx])
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()

def plot_component_sorted_traces(F_norm, pca, save_dir=None):
    """
    Create figures showing traces sorted by their contribution to each principal component.
    
    Args:
        F_norm (numpy.ndarray): Normalized fluorescence traces
        pca: Fitted PCA object
        save_dir (str, optional): Directory to save the figures
    """
    # Handle NaN values for transform
    F_norm_no_nan = np.nan_to_num(F_norm, nan=0)
    
    # Get proper projections using transform
    # This gives us the projection of each neuron onto each PC
    projections = pca.transform(F_norm_no_nan)  # Shape: (n_neurons, n_components)
    
    # Create a figure for each component
    for i in range(5):  # For each of the first 5 PCs
        # Sort the normalized matrix based on projections onto this PC
        # Use absolute value of projections to sort by magnitude of contribution
        sort_idx = np.argsort(np.abs(projections[:, i]))
        F_sorted = F_norm[sort_idx, :]
        
        # Create interactive viewer
        viewer = InteractiveComponentViewer(F_sorted, pca, i)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process .npy files from a suite2p plane0 directory')
    parser.add_argument('directory', type=str, help='Path to the suite2p plane0 directory containing .npy files')
    args = parser.parse_args()
    
    # Convert directory path to Path object and resolve it
    directory = Path(args.directory).resolve()
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    # Check if this is a plane0 directory
    if not directory.name == 'plane0':
        print(f"Warning: Expected a 'plane0' directory, but got '{directory.name}'")
        print("Please make sure you're using the correct suite2p output directory")
    
    print(f"\nProcessing directory: {directory}")
    
    # Load the data
    data = load_npy_files(directory)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Process F.npy if it exists
    F_processed = mask_stimulation_range(data)
    
    if F_processed is not None:
        print("\nProcessing completed successfully")
        
        # Create paths for saving figures
        traces_save_path = directory / "neural_traces.png"
        pca_save_path = directory / "pca_components.png"
        
        # Plot and save original figures
        F_norm = plot_traces(F_processed, str(traces_save_path))
        plot_pca_components(F_processed, str(pca_save_path))
        
        # Create component-specific visualizations
        # First normalize the data
        F_norm = normalize_traces(F_processed)
        # Handle NaN values for PCA
        F_norm_no_nan = np.nan_to_num(F_norm, nan=0)
        # Perform PCA on normalized data
        pca = PCA(n_components=5)
        pca.fit(F_norm_no_nan)
        
        # Create the new visualizations
        plot_component_sorted_traces(F_norm, pca, str(directory))

if __name__ == "__main__":
    main() 