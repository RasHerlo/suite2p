import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal
import tifffile
import argparse
from pathlib import Path
from sklearn.decomposition import TruncatedSVD, NMF
import time
import warnings
from matplotlib.widgets import Button
warnings.filterwarnings('ignore', category=UserWarning)

def load_tiff(file_path):
    """Load a TIFF stack into a numpy array."""
    print(f"Loading {file_path}...")
    start_time = time.time()
    try:
        # Load the stack (time, height, width)
        stack = tifffile.imread(file_path)
        print(f"Loaded {stack.shape} stack in {time.time() - start_time:.2f}s")
        return stack
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def save_tiff(stack, output_path):
    """Save a numpy array as a TIFF stack."""
    print(f"Saving to {output_path}...")
    start_time = time.time()
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Save as TIFF
        tifffile.imwrite(output_path, stack)
        print(f"Saved in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {str(e)}")
        return False

def compute_decomposition(stack, n_components=10, method='pca'):
    """
    Compute PCA/SVD or NMF decomposition of the stack.
    
    Args:
        stack: numpy array of shape (time, height, width)
        n_components: number of components to compute
        method: 'pca' (or 'svd') or 'nmf'
        
    Returns:
        W: temporal components (time, components)
        H: spatial components (components, height*width)
        components_mean: mean of original data (for PCA)
    """
    # Get dimensions
    frames, height, width = stack.shape
    
    # Reshape to (time, pixels)
    X = stack.reshape(frames, height * width)
    
    # Handle different methods
    if method.lower() in ['pca', 'svd']:
        # Center the data
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        
        # Compute SVD
        print(f"Computing SVD/PCA with {n_components} components...")
        start_time = time.time()
        svd = TruncatedSVD(n_components=n_components)
        W = svd.fit_transform(X_centered)  # temporal components (time, components)
        S = svd.singular_values_  # singular values
        H = svd.components_  # spatial components (components, pixels)
        print(f"SVD/PCA computed in {time.time() - start_time:.2f}s")
        
        components_mean = X_mean
        
    elif method.lower() == 'nmf':
        # NMF requires non-negative data
        if np.min(X) < 0:
            print("Data contains negative values. Shifting data to be non-negative.")
            X_offset = np.min(X)
            X = X - X_offset
        else:
            X_offset = 0
        
        # Compute NMF
        print(f"Computing NMF with {n_components} components...")
        start_time = time.time()
        nmf = NMF(n_components=n_components, init='nndsvd', max_iter=1000)
        W = nmf.fit_transform(X)  # temporal components (time, components)
        H = nmf.components_  # spatial components (components, pixels)
        print(f"NMF computed in {time.time() - start_time:.2f}s")
        
        # Store offset as part of the mean
        components_mean = np.zeros_like(X[0]) + X_offset
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'nmf'.")
    
    # Print stats about decomposition
    print(f"Original data range: {np.min(stack):.2f} to {np.max(stack):.2f}")
    
    return W, H, components_mean

def reconstruct_data(W, H, components_mean, excluded_components=None, method='pca'):
    """
    Reconstruct the data using the decomposition, optionally excluding components.
    
    Args:
        W: temporal components (time, components)
        H: spatial components (components, pixels)
        components_mean: mean of original data (for PCA)
        excluded_components: list of components to exclude (0-based indices)
        method: 'pca' (or 'svd') or 'nmf'
        
    Returns:
        reconstructed: reconstructed data
    """
    if excluded_components is None:
        excluded_components = []
    
    # Make a copy of the components
    W_filtered = W.copy()
    
    # Zero out excluded components
    for comp_idx in excluded_components:
        W_filtered[:, comp_idx] = 0
    
    # Reconstruct without adding components_mean for any method
    reconstructed = W_filtered @ H
        
    return reconstructed

def compute_excluded_only(W, H, components_mean, excluded_components, method='pca'):
    """
    Compute the data containing only the excluded components.
    
    Args:
        W: temporal components (time, components)
        H: spatial components (components, pixels)
        components_mean: mean of original data
        excluded_components: list of components to exclude (0-based indices)
        method: 'pca' (or 'svd') or 'nmf'
        
    Returns:
        excluded_only: data containing only the excluded components
    """
    # Make a copy with zeros for all components
    W_excluded = np.zeros_like(W)
    
    # Set only the excluded components
    for comp_idx in excluded_components:
        W_excluded[:, comp_idx] = W[:, comp_idx]
    
    # Reconstruct using only the excluded components, without adding mean
    excluded_only = W_excluded @ H
    
    return excluded_only

def plot_components(W, H, height, width, method='pca'):
    """
    Plot the first 10 components for interactive selection.
    
    Args:
        W: temporal components (time, components)
        H: spatial components (components, pixels)
        height, width: dimensions of original frames
        method: decomposition method used
        
    Returns:
        fig: matplotlib figure
        selected_components: list to store selected component indices
    """
    n_components = min(10, H.shape[0])
    selected_components = []
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    # Compute component statistics for display
    component_stats = []
    for i in range(n_components):
        spatial = H[i].reshape(height, width)
        temporal = W[:, i]
        
        # Compute some basic stats for each component
        spatial_range = np.max(spatial) - np.min(spatial)
        temporal_range = np.max(temporal) - np.min(temporal)
        
        # For PCA/SVD, compute variance explained
        if method.lower() in ['pca', 'svd']:
            component_power = np.sum(temporal**2)
            total_power = np.sum(W**2)
            variance_explained = (component_power / total_power) * 100
        else:  # NMF
            # For NMF, use relative contribution to reconstruction
            component_contribution = np.sum(np.outer(temporal, H[i]))
            total_contribution = np.sum(W @ H)
            variance_explained = (component_contribution / total_contribution) * 100
        
        component_stats.append({
            'spatial_range': spatial_range,
            'temporal_range': temporal_range,
            'variance_explained': variance_explained
        })
    
    # Store references to the rectangles for selection
    component_rectangles = []
    
    # Plot components
    for i in range(n_components):
        ax = axes[i]
        
        # Get spatial component and normalize for display
        spatial = H[i].reshape(height, width)
        spatial_norm = (spatial - np.mean(spatial)) / (np.std(spatial) + 1e-10)
        
        # Plot spatial component
        im = ax.imshow(spatial_norm, cmap='coolwarm', vmin=-2, vmax=2)
        ax.set_title(f"Component {i+1}\n{component_stats[i]['variance_explained']:.1f}% var")
        ax.axis('off')
        
        # Add a selectable rectangle around the component
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, 
                         edgecolor='black', linewidth=0, alpha=0.8)
        ax.add_patch(rect)
        component_rectangles.append(rect)
    
    # Function to handle selection
    def on_click(event):
        if event.inaxes in axes:
            comp_idx = axes.tolist().index(event.inaxes)
            if comp_idx < n_components:
                if comp_idx in selected_components:
                    # Deselect component
                    selected_components.remove(comp_idx)
                    component_rectangles[comp_idx].set_linewidth(0)
                else:
                    # Select component
                    selected_components.append(comp_idx)
                    component_rectangles[comp_idx].set_linewidth(3)
                
                fig.canvas.draw_idle()
                print(f"Selected components: {[i+1 for i in selected_components]}")
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Add method label and instructions
    method_label = 'PCA' if method.lower() in ['pca', 'svd'] else 'NMF'
    fig.suptitle(f"{method_label} Components - Click to select components for removal", fontsize=16)
    
    # Add a "Done" button
    plt.subplots_adjust(bottom=0.15)
    done_ax = plt.axes([0.45, 0.05, 0.1, 0.04])
    done_button = Button(done_ax, 'Done')
    
    done_pressed = False
    
    def on_done(event):
        nonlocal done_pressed
        done_pressed = True
        plt.close(fig)
    
    done_button.on_clicked(on_done)
    
    # Show plot and wait for it to be closed
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show(block=True)
    
    return selected_components

def plot_method_selection():
    """
    Display a dialog to select between PCA and NMF.
    
    Returns:
        method: 'pca' or 'nmf'
    """
    fig, ax = plt.figure(figsize=(5, 3)), plt.axes([0, 0, 1, 1])
    ax.axis('off')
    
    pca_ax = plt.axes([0.2, 0.4, 0.2, 0.2])
    nmf_ax = plt.axes([0.6, 0.4, 0.2, 0.2])
    
    pca_button = Button(pca_ax, 'PCA')
    nmf_button = Button(nmf_ax, 'NMF')
    
    selected_method = [None]  # Use a list to store the selection so it can be modified in the callback
    
    def on_pca(event):
        selected_method[0] = 'pca'
        plt.close(fig)
    
    def on_nmf(event):
        selected_method[0] = 'nmf'
        plt.close(fig)
    
    pca_button.on_clicked(on_pca)
    nmf_button.on_clicked(on_nmf)
    
    plt.suptitle('Select decomposition method', fontsize=14)
    ax.text(0.5, 0.7, "Choose the method to decompose the video:", 
            horizontalalignment='center', fontsize=12)
    ax.text(0.5, 0.15, "PCA/SVD: Good for oscillatory patterns\nNMF: Good for separate additive components", 
            horizontalalignment='center', fontsize=10)
    
    plt.show(block=True)
    
    # If no method selected, default to PCA
    if selected_method[0] is None:
        print("No method selected, defaulting to PCA")
        return 'pca'
    
    return selected_method[0]

def display_component_details(W, H, stack_shape, selected_components, method='pca'):
    """
    Display detailed plots of the selected components.
    
    Args:
        W: temporal components
        H: spatial components
        stack_shape: original stack shape
        selected_components: list of selected component indices
        method: decomposition method
    """
    if not selected_components:
        print("No components selected for detailed view.")
        return
    
    frames, height, width = stack_shape
    
    for comp_idx in selected_components:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Spatial component
        spatial = H[comp_idx].reshape(height, width)
        spatial_norm = (spatial - np.mean(spatial)) / (np.std(spatial) + 1e-10)
        im1 = axes[0, 0].imshow(spatial_norm, cmap='coolwarm', vmin=-2, vmax=2)
        axes[0, 0].set_title(f"Component {comp_idx+1} - Spatial Pattern")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Temporal trace
        temporal = W[:, comp_idx]
        axes[0, 1].plot(temporal, 'k-')
        axes[0, 1].set_title(f"Component {comp_idx+1} - Temporal Pattern")
        axes[0, 1].set_xlabel("Frame")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Power spectrum
        freq, psd = signal.welch(temporal, fs=1.0, nperseg=min(256, len(temporal)))
        axes[1, 0].semilogy(freq, psd)
        axes[1, 0].set_title(f"Component {comp_idx+1} - Power Spectrum")
        axes[1, 0].set_xlabel("Frequency (cycles/frame)")
        axes[1, 0].set_ylabel("Power")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Autocorrelation (useful for detecting periodic patterns)
        max_lag = min(frames // 4, 100)  # Limit lag to avoid noise in long autocorrelations
        autocorr = np.correlate(temporal, temporal, mode='full')[frames-1-max_lag:frames+max_lag]
        norm_autocorr = autocorr / autocorr[max_lag]  # Normalize by zero-lag
        lags = np.arange(-max_lag, max_lag+1)
        axes[1, 1].plot(lags, norm_autocorr)
        axes[1, 1].set_title(f"Component {comp_idx+1} - Autocorrelation")
        axes[1, 1].set_xlabel("Lag (frames)")
        axes[1, 1].set_ylabel("Autocorrelation")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Detailed View of Component {comp_idx+1}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def print_data_stats(name, data):
    """Print statistical information about the data."""
    print(f"{name} stats:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
    print(f"  Mean: {np.mean(data):.2f}, Median: {np.median(data):.2f}")
    print(f"  5th percentile: {np.percentile(data, 5):.2f}, 95th percentile: {np.percentile(data, 95):.2f}")
    print("-" * 50)

def process_file(file_path, n_components=10):
    """
    Process a video file to remove ripple artifacts.
    
    Args:
        file_path: path to the TIFF file
        n_components: number of components to extract
    """
    # Load the file
    stack = load_tiff(file_path)
    if stack is None:
        print("Failed to load the file. Exiting.")
        return
    
    # Print original stack stats
    print_data_stats("Original stack", stack)
    
    # Get original stack properties
    frames, height, width = stack.shape
    stack_dtype = stack.dtype
    
    # Ask user to select decomposition method
    method = plot_method_selection()
    method_name = 'PCA' if method == 'pca' else 'NMF'
    print(f"Using {method_name} decomposition")
    
    # Compute decomposition
    W, H, components_mean = compute_decomposition(stack, n_components, method)
    
    # Print stats about components_mean
    print_data_stats("Components mean", components_mean)
    
    # Plot and select components to remove
    selected_components = plot_components(W, H, height, width, method)
    
    if not selected_components:
        print("No components selected for removal. Exiting.")
        return
    
    # Display detailed info for selected components
    display_component_details(W, H, stack.shape, selected_components, method)
    
    # Reconstruct without selected components
    print(f"Reconstructing video without components: {[i+1 for i in selected_components]}")
    reconstructed = reconstruct_data(W, H, components_mean, selected_components, method)
    
    # Print stats about reconstructed data
    print_data_stats("Reconstructed data (before reshaping)", reconstructed)
    
    # Also compute the components that were removed
    removed_components = compute_excluded_only(W, H, components_mean, selected_components, method)
    
    # Print stats about removed components
    print_data_stats("Removed components (before reshaping)", removed_components)
    
    # Reshape to original dimensions
    denoised_stack = reconstructed.reshape(frames, height, width)
    removed_stack = removed_components.reshape(frames, height, width)
    
    # Print stats after reshaping
    print_data_stats("Denoised stack (after reshaping)", denoised_stack)
    print_data_stats("Removed stack (after reshaping)", removed_stack)
    
    # Check for negative values and determine offsets
    denoised_min = np.min(denoised_stack)
    removed_min = np.min(removed_stack)
    
    denoised_offset = 0
    removed_offset = 0
    
    # Add offsets if necessary to handle negative values
    if denoised_min < 0:
        denoised_offset = int(abs(denoised_min)) + 1  # Add 1 for safety
        print(f"Adding offset of {denoised_offset} to denoised stack (min value was {denoised_min})")
        denoised_stack = denoised_stack + denoised_offset
    
    if removed_min < 0:
        removed_offset = int(abs(removed_min)) + 1  # Add a 1 for safety
        print(f"Adding offset of {removed_offset} to removed stack (min value was {removed_min})")
        removed_stack = removed_stack + removed_offset
    
    # Print stats after offset
    print_data_stats("Denoised stack (after offset)", denoised_stack)
    print_data_stats("Removed stack (after offset)", removed_stack)
    
    # Scale the stacks to match the original data type range
    if stack_dtype == np.uint8:
        denoised_stack = np.clip(denoised_stack, 0, 255).astype(np.uint8)
        removed_stack = np.clip(removed_stack, 0, 255).astype(np.uint8)
    elif stack_dtype == np.uint16:
        denoised_stack = np.clip(denoised_stack, 0, 65535).astype(np.uint16)
        removed_stack = np.clip(removed_stack, 0, 65535).astype(np.uint16)
    else:
        # For floating point, just convert
        denoised_stack = denoised_stack.astype(stack_dtype)
        removed_stack = removed_stack.astype(stack_dtype)
    
    # Print stats after dtype conversion
    print_data_stats("Final denoised stack", denoised_stack)
    print_data_stats("Final removed stack", removed_stack)
    
    # Create output file names with offset information
    input_file = Path(file_path)
    component_str = "_".join([str(i+1) for i in selected_components])
    
    denoised_output = str(input_file.with_name(
        f"{input_file.stem}_{method_name}_{component_str}_remain_Ofs{denoised_offset}{input_file.suffix}"))
    
    removed_output = str(input_file.with_name(
        f"{input_file.stem}_{method_name}_{component_str}_removed_Ofs{removed_offset}{input_file.suffix}"))
    
    # Save results
    save_success1 = save_tiff(denoised_stack, denoised_output)
    save_success2 = save_tiff(removed_stack, removed_output)
    
    if save_success1 and save_success2:
        print(f"Successfully saved processed files:")
        print(f"  Denoised: {denoised_output}")
        print(f"  Removed components: {removed_output}")
    else:
        print("Error saving one or more output files.")

def main():
    parser = argparse.ArgumentParser(description='Remove ripple artifacts from imaging videos')
    parser.add_argument('input_file', type=str, help='Path to input TIFF file')
    parser.add_argument('--n_components', type=int, default=10, 
                        help='Number of components to extract (default: 10)')
    args = parser.parse_args()
    
    # Process the file
    process_file(args.input_file, args.n_components)
    
if __name__ == "__main__":
    main() 