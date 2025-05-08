import os
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
            print(f"Loaded {file_name}.npy")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Process .npy files from a directory')
    parser.add_argument('directory', type=str, help='Directory containing .npy files')
    parser.add_argument('--save_fig', type=str, help='Path to save the figure (optional)')
    args = parser.parse_args()
    
    # Load the data
    data = load_npy_files(args.directory)
    
    # Process F.npy if it exists
    F_processed = mask_stimulation_range(data)
    
    if F_processed is not None:
        print("\nProcessing completed successfully")
        print(f"Number of NaNs in processed data: {np.isnan(F_processed).sum()}")
        
        # Create and show the first figure
        F_norm = plot_traces(F_processed, args.save_fig)
        
        # Create and show the PCA figure for normalized data
        pca_save_path = args.save_fig.replace('.png', '_pca_norm.png') if args.save_fig else None
        plot_pca_components(F_norm, pca_save_path, is_normalized=True)
        
        # Create and show the PCA figure for raw data
        pca_raw_save_path = args.save_fig.replace('.png', '_pca_raw.png') if args.save_fig else None
        plot_pca_components(F_processed, pca_raw_save_path, is_normalized=False)

if __name__ == "__main__":
    main() 