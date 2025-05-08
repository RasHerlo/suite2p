import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector
from scipy import signal
import tifffile
import argparse
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import time
import warnings
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

def compute_svd(stack, n_components=50):
    """
    Compute SVD decomposition of the stack.
    
    Args:
        stack: numpy array of shape (time, height, width)
        n_components: number of SVD components to compute
        
    Returns:
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, height*width)
    """
    # Reshape to (time, pixels)
    frames, height, width = stack.shape
    X = stack.reshape(frames, height * width)
    
    # Center the data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    
    # Compute SVD
    print(f"Computing SVD with {n_components} components...")
    start_time = time.time()
    svd = TruncatedSVD(n_components=n_components)
    U = svd.fit_transform(X_centered)  # temporal components (time, components)
    S = svd.singular_values_  # singular values
    V = svd.components_  # spatial components (components, pixels)
    print(f"SVD computed in {time.time() - start_time:.2f}s")
    
    # Print stats about decomposition
    print(f"Original data range: {np.min(X):.2f} to {np.max(X):.2f}")
    
    # Test reconstruction
    X_approx = U @ np.diag(S) @ V + X_mean
    error = np.mean(np.abs(X - X_approx))
    print(f"SVD reconstruction error: {error:.2f}")
    print(f"Reconstructed data range: {np.min(X_approx):.2f} to {np.max(X_approx):.2f}")
    
    return U, S, V, X_mean

def plot_svd_components(U, S, V, height, width, n_components=10):
    """
    Plot SVD components for analysis.
    
    Args:
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, pixels)
        height, width: original dimensions of each frame
        n_components: number of components to plot
    """
    n_components = min(n_components, len(S))
    
    # Create colormap for spatial components
    cmap = LinearSegmentedColormap.from_list('bipolar', ['blue', 'white', 'red'])
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot singular values
    plt.subplot(3, 1, 1)
    plt.semilogy(np.arange(len(S)), S, 'o-')
    plt.grid(True)
    plt.title('Singular Values')
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value (log scale)')
    
    # Plot temporal components
    plt.subplot(3, 1, 2)
    for i in range(n_components):
        plt.plot(U[:, i], label=f'Component {i+1}')
    plt.grid(True)
    plt.title('Temporal Components')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot PSD of temporal components to identify oscillatory patterns
    plt.subplot(3, 1, 3)
    for i in range(n_components):
        freq, psd = signal.welch(U[:, i], fs=1.0, nperseg=min(256, len(U[:, i])))
        plt.semilogy(freq, psd, label=f'Component {i+1}')
    plt.grid(True)
    plt.title('Power Spectral Density of Temporal Components')
    plt.xlabel('Frequency (cycles/frame)')
    plt.ylabel('PSD (log scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot spatial components
    n_rows = (n_components + 4) // 5  # Arrange in rows of 5
    fig, axs = plt.subplots(n_rows, 5, figsize=(15, 3*n_rows))
    axs = axs.flatten()
    
    for i in range(n_components):
        if i < len(axs):
            spatial_comp = V[i].reshape(height, width)
            # Normalize for better visualization
            spatial_comp = (spatial_comp - spatial_comp.mean()) / (spatial_comp.std() + 1e-10)
            im = axs[i].imshow(spatial_comp, cmap=cmap, vmin=-2, vmax=2)
            axs[i].set_title(f'Component {i+1}')
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_components, len(axs)):
        axs[i].axis('off')
        
    plt.tight_layout()
    fig.colorbar(im, ax=axs, shrink=0.8, label='Normalized Value')
    plt.show()

def reconstruct_without_components(U, S, V, X_mean, excluded_components):
    """
    Reconstruct the stack without specified components.
    
    Args:
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, pixels)
        X_mean: mean of original data
        excluded_components: list of component indices to exclude (0-based)
        
    Returns:
        X_reconstructed: reconstructed data
    """
    n_components = len(S)
    
    # Create a copy of the data for reconstruction
    S_filtered = S.copy()
    
    # Debug info
    print(f"Total components: {n_components}")
    print(f"Components to exclude: {[i+1 for i in excluded_components]}")
    print(f"Top 5 singular values before: {S[:5]}")
    
    # Calculate original reconstruction for scaling reference
    X_orig_full = U @ np.diag(S) @ V + X_mean
    min_orig = np.min(X_orig_full)
    max_orig = np.max(X_orig_full)
    
    # Zero out the excluded components
    for i in excluded_components:
        S_filtered[i] = 0
    
    print(f"Top 5 singular values after filtering: {S_filtered[:5]}")
    
    # Reconstruct using matrix multiplication
    X_reconstructed = U @ np.diag(S_filtered) @ V + X_mean
    
    # Scale to match original range if needed
    min_recon = np.min(X_reconstructed)
    max_recon = np.max(X_reconstructed)
    
    # If range is significantly different, rescale
    if abs((max_recon - min_recon) - (max_orig - min_orig)) / (max_orig - min_orig) > 0.1:
        print("Significant range difference detected. Rescaling...")
        print(f"Original range: {min_orig:.2f} to {max_orig:.2f}, span: {max_orig - min_orig:.2f}")
        print(f"Reconstructed range: {min_recon:.2f} to {max_recon:.2f}, span: {max_recon - min_recon:.2f}")
        
        # Scale to match original range
        X_reconstructed = ((X_reconstructed - min_recon) / (max_recon - min_recon)) * (max_orig - min_orig) + min_orig
        
        # Verify rescaling
        min_rescaled = np.min(X_reconstructed)
        max_rescaled = np.max(X_reconstructed)
        print(f"After rescaling: {min_rescaled:.2f} to {max_rescaled:.2f}, span: {max_rescaled - min_rescaled:.2f}")
    
    # Calculate and print actual difference
    diff = X_orig_full - X_reconstructed
    print(f"Max difference: {np.max(np.abs(diff))}")
    print(f"Mean difference: {np.mean(np.abs(diff))}")
    
    return X_reconstructed

def plot_component_difference(stack, denoised_stack, excluded_components):
    """
    Plot the difference between original and denoised stacks.
    
    Args:
        stack: original stack
        denoised_stack: denoised stack
        excluded_components: list of excluded component indices (0-based)
    """
    mid_frame = stack.shape[0] // 2
    
    # Calculate difference
    diff = stack[mid_frame] - denoised_stack[mid_frame]
    
    # Calculate difference statistics
    diff_abs_mean = np.mean(np.abs(diff))
    diff_std = np.std(diff)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine common scale
    vmin, vmax = np.percentile(stack[mid_frame], [1, 99])
    
    # For difference display, use a reasonable scale based on statistics
    diff_scale = 3 * diff_std  # Use 3 standard deviations for colormap range
    
    # Original
    axes[0].imshow(stack[mid_frame], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    # Denoised
    axes[1].imshow(denoised_stack[mid_frame], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Denoised Frame (Components {", ".join([str(i+1) for i in sorted(excluded_components)])} Removed)')
    axes[1].axis('off')
    
    # Difference
    im = axes[2].imshow(diff, cmap='coolwarm', vmin=-diff_scale, vmax=diff_scale)
    axes[2].set_title('Difference (Original - Denoised)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='Difference')
    
    # Add text with statistics
    orig_mean = np.mean(stack[mid_frame])
    orig_std = np.std(stack[mid_frame])
    denoised_mean = np.mean(denoised_stack[mid_frame])
    denoised_std = np.std(denoised_stack[mid_frame])
    
    fig.suptitle(f"Original: mean={orig_mean:.1f}, std={orig_std:.1f} | " 
                f"Denoised: mean={denoised_mean:.1f}, std={denoised_std:.1f} | "
                f"Diff: mean={diff_abs_mean:.1f}, std={diff_std:.1f}")
    
    plt.tight_layout()
    plt.show()

def plot_temporal_difference(stack, denoised_stack, excluded_components):
    """
    Plot the temporal differences between original and denoised stacks.
    
    Args:
        stack: original stack
        denoised_stack: denoised stack
        excluded_components: list of excluded component indices (0-based)
    """
    frames, height, width = stack.shape
    
    # Calculate difference for all frames
    diff = stack - denoised_stack
    
    # Calculate mean difference across frames
    mean_diff = np.mean(np.abs(diff), axis=(1, 2))
    
    # Plot temporal profile
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mean_diff, 'k-', linewidth=2)
    ax.set_title(f'Temporal Profile of Differences (Components {", ".join([str(i+1) for i in sorted(excluded_components)])} Removed)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Absolute Difference')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    max_diff = np.max(mean_diff)
    mean_total_diff = np.mean(mean_diff)
    ax.axhline(mean_total_diff, color='r', linestyle='--', label=f'Mean: {mean_total_diff:.2f}')
    
    # Find peaks in difference
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(mean_diff, height=mean_total_diff*1.5)
    if len(peaks) > 0:
        ax.plot(peaks, mean_diff[peaks], 'ro', label=f'Peaks: {len(peaks)}')
    
    ax.legend()
    plt.tight_layout()
    plt.show()

def visual_component_selection(n_components=50):
    """
    Create an interactive component selection interface.
    
    Args:
        n_components: number of components to display
        
    Returns:
        selected_components: list of selected component indices (0-based)
    """
    selected_components = []
    
    # Create a grid of clickable component boxes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, (n_components // 10) + 1 if n_components % 10 else (n_components // 10))
    
    # Draw grid of components
    component_rects = []
    for i in range(n_components):
        row = i // 10
        col = i % 10
        rect = plt.Rectangle((col, row), 0.9, 0.9, fill=True, color='lightblue', alpha=0.5)
        ax.add_patch(rect)
        ax.text(col+0.45, row+0.45, f"{i+1}", ha='center', va='center')
        component_rects.append(rect)
    
    ax.set_title('Click on components to include/exclude (blue=keep, red=remove)\nClose window when done')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        
        # Determine which component was clicked
        col = int(event.xdata)
        row = int(event.ydata)
        comp_idx = row * 10 + col
        
        if comp_idx < n_components:
            # Toggle component selection
            if comp_idx in selected_components:
                selected_components.remove(comp_idx)
                component_rects[comp_idx].set_color('lightblue')
                component_rects[comp_idx].set_alpha(0.5)
                print(f"Will keep component {comp_idx+1}")
            else:
                selected_components.append(comp_idx)
                component_rects[comp_idx].set_color('red')
                component_rects[comp_idx].set_alpha(0.7)
                print(f"Will remove component {comp_idx+1}")
            
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return selected_components

def reconstruct_single_component(U, S, V, X_mean, component_idx):
    """
    Reconstruct a stack with only the specified component.
    
    Args:
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, pixels)
        X_mean: mean of original data
        component_idx: index of component to isolate (0-based)
        
    Returns:
        X_reconstructed: reconstructed data with only one component
    """
    n_components = len(S)
    
    # Create a vector of zeros
    S_single = np.zeros_like(S)
    
    # Set only the selected component to its original value
    S_single[component_idx] = S[component_idx]
    
    # Reconstruct using only this component
    X_reconstructed = U @ np.diag(S_single) @ V + X_mean
    
    # For debugging, print component's contribution
    component_contribution = S[component_idx] / np.sum(S) * 100
    print(f"Component {component_idx+1} contributes {component_contribution:.2f}% of total variance")
    
    return X_reconstructed

def plot_component_contribution(U, S, component_indices, stack_shape):
    """
    Plot the temporal contribution of specified components.
    
    Args:
        U: temporal components (time, components)
        S: singular values 
        component_indices: indices of components to plot (0-based)
        stack_shape: shape of the original stack
    """
    frames = stack_shape[0]
    
    # Plot temporal contributions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different components
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, comp_idx in enumerate(component_indices):
        # The temporal contribution is U[:,comp_idx] scaled by singular value
        contribution = U[:, comp_idx] * S[comp_idx]
        color = colors[i % len(colors)]
        ax.plot(contribution, color=color, linewidth=2, label=f'Component {comp_idx+1}')
        
        # Calculate contribution metrics
        var_explained = S[comp_idx] / np.sum(S) * 100
        
        # Add extra info to label
        ax.text(frames-1, contribution[-1], f"  #{comp_idx+1}: {var_explained:.1f}%", 
                color=color, verticalalignment='center')
    
    ax.set_title('Temporal Contribution of Selected Components')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Contribution (a.u.)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def visualize_component(stack, U, S, V, X_mean, component_idx):
    """
    Visualize a single component as a video and spatial map.
    
    Args:
        stack: original stack
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, pixels)
        X_mean: mean of the original data
        component_idx: index of component to visualize (0-based)
    """
    frames, height, width = stack.shape
    
    # Reconstruct the video with only this component
    X_single = reconstruct_single_component(U, S, V, X_mean, component_idx)
    component_video = X_single.reshape(frames, height, width)
    
    # Calculate original video stats for scaling
    vmin_global, vmax_global = np.percentile(stack, [1, 99])
    
    # Calculate component's contribution range
    vmin_comp, vmax_comp = np.min(component_video), np.max(component_video)
    
    # Reshape spatial component for visualization
    spatial_comp = V[component_idx].reshape(height, width)
    
    # Extract temporal component
    temporal_comp = U[:, component_idx] * S[component_idx]
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spatial component map
    ax1 = fig.add_subplot(2, 2, 1)
    # Normalize for better visualization
    spatial_comp_norm = (spatial_comp - spatial_comp.mean()) / (spatial_comp.std() + 1e-10)
    im1 = ax1.imshow(spatial_comp_norm, cmap='coolwarm', vmin=-2, vmax=2)
    ax1.set_title(f'Component {component_idx+1} - Spatial Pattern')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Temporal profile
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(temporal_comp, 'k-', linewidth=2)
    ax2.set_title(f'Component {component_idx+1} - Temporal Pattern')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Contribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Power spectrum
    ax3 = fig.add_subplot(2, 2, 3)
    freq, psd = signal.welch(temporal_comp, fs=1.0, nperseg=min(256, len(temporal_comp)))
    ax3.semilogy(freq, psd)
    ax3.set_title(f'Component {component_idx+1} - Power Spectrum')
    ax3.set_xlabel('Frequency (cycles/frame)')
    ax3.set_ylabel('Power')
    ax3.grid(True, alpha=0.3)
    
    # 4. Example frame from component video
    ax4 = fig.add_subplot(2, 2, 4)
    mid_frame = frames // 2
    im4 = ax4.imshow(component_video[mid_frame], cmap='gray', 
                    vmin=vmin_comp, vmax=vmax_comp)
    ax4.set_title(f'Component {component_idx+1} - Frame {mid_frame}')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4)
    
    # Add overall title with component stats
    var_explained = S[component_idx] / np.sum(S) * 100
    intensity_range = vmax_comp - vmin_comp
    fig.suptitle(f'Component {component_idx+1}: {var_explained:.2f}% of total variance, Intensity range: {intensity_range:.2f}', 
                fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Return the component video for further analysis
    return component_video

def visualize_frame_components(stack, U, S, V, X_mean, frame_index=None, top_n=5):
    """
    Visualize how different components contribute to a specific frame.
    
    Args:
        stack: original stack
        U: temporal components
        S: singular values
        V: spatial components
        X_mean: mean of original data
        frame_index: index of frame to analyze (if None, use middle frame)
        top_n: number of top components to show
    """
    frames, height, width = stack.shape
    
    # Use middle frame if not specified
    if frame_index is None:
        frame_index = frames // 2
    
    # Get the original frame
    original_frame = stack[frame_index]
    
    # Calculate contribution of each component to this frame
    component_contributions = []
    for i in range(len(S)):
        # Create singular value vector with just this component
        S_single = np.zeros_like(S)
        S_single[i] = S[i]
        
        # Reconstruct frame with just this component
        X_single = U @ np.diag(S_single) @ V + X_mean
        frame_single = X_single.reshape(frames, height, width)[frame_index]
        
        # Calculate contribution metrics
        contribution_magnitude = np.mean(np.abs(frame_single))
        variance_explained = S[i] / np.sum(S) * 100
        temporal_weight = U[frame_index, i]
        
        component_contributions.append({
            'index': i,
            'magnitude': contribution_magnitude,
            'variance': variance_explained,
            'temporal_weight': temporal_weight,
            'frame': frame_single
        })
    
    # Sort by contribution magnitude
    component_contributions.sort(key=lambda x: x['magnitude'], reverse=True)
    
    # Plot original frame and top contributing components
    fig = plt.figure(figsize=(15, 8))
    
    # Original frame
    ax0 = fig.add_subplot(2, 3, 1)
    vmin, vmax = np.percentile(original_frame, [1, 99])
    ax0.imshow(original_frame, cmap='gray', vmin=vmin, vmax=vmax)
    ax0.set_title(f'Original Frame {frame_index}')
    ax0.axis('off')
    
    # Top components for this frame
    for i in range(min(top_n, len(component_contributions))):
        comp = component_contributions[i]
        ax = fig.add_subplot(2, 3, i+2)
        
        # For better visualization, use wider contrast
        comp_vmin, comp_vmax = np.percentile(comp['frame'], [5, 95])
        im = ax.imshow(comp['frame'], cmap='viridis', vmin=comp_vmin, vmax=comp_vmax)
        
        ax.set_title(f"Component {comp['index']+1}\n" +
                    f"Weight: {comp['temporal_weight']:.3f}\n" +
                    f"Var: {comp['variance']:.1f}%")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle(f'Component Contributions to Frame {frame_index}', fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()
    
    return component_contributions

def reconstruct_only_components(U, S, V, X_mean, component_indices):
    """
    Reconstruct a stack using only the specified components.
    
    Args:
        U: temporal components (time, components)
        S: singular values
        V: spatial components (components, pixels)
        X_mean: mean of original data
        component_indices: list of component indices to include (0-based)
        
    Returns:
        X_reconstructed: reconstructed data with only selected components
    """
    n_components = len(S)
    
    # Create a vector of zeros
    S_only_selected = np.zeros_like(S)
    
    # Set only the selected components to their original values
    for i in component_indices:
        S_only_selected[i] = S[i]
    
    # Reconstruct using only these components
    X_reconstructed = U @ np.diag(S_only_selected) @ V + X_mean
    
    # Print component contributions
    total_var = np.sum(S)
    component_var = np.sum(S[component_indices])
    print(f"Selected components contribute {component_var/total_var*100:.2f}% of total variance")
    
    return X_reconstructed

def interactive_component_removal(stack, n_components=50, method='svd'):
    """
    SVD-based denoising of the stack with interactive component selection.
    
    Args:
        stack: numpy array of shape (time, height, width)
        n_components: number of SVD components to compute
        method: 'svd' or 'pca'
        
    Returns:
        denoised_stack: denoised stack
        excluded_components: list of excluded components
    """
    frames, height, width = stack.shape
    
    # Compute SVD
    U, S, V, X_mean = compute_svd(stack, n_components)
    
    # Plot components for analysis
    plot_svd_components(U, S, V, height, width, n_components=min(10, n_components))
    
    # Plot contribution of top components
    plot_component_contribution(U, S, list(range(5)), stack.shape)
    
    # Visualize how components contribute to a specific frame
    middle_frame = frames // 2
    visualize_frame_components(stack, U, S, V, X_mean, frame_index=middle_frame, top_n=5)
    
    # Interactive component selection
    print("\nBased on the plots, select which components to exclude")
    print("Click on components you want to REMOVE (they will turn red)")
    print("Components you leave blue will be KEPT")
    excluded_components = visual_component_selection(n_components)
    
    if excluded_components:
        print(f"Excluding components: {[i+1 for i in sorted(excluded_components)]}")
        
        # Visualize each selected component
        for comp_idx in excluded_components:
            print(f"\nVisualizing component {comp_idx+1}...")
            visualize_component(stack, U, S, V, X_mean, comp_idx)
        
        # Debug info
        print("Testing reconstruction...")
        
        # Reconstruct without excluded components
        X_reconstructed = reconstruct_without_components(U, S, V, X_mean, excluded_components)
        
        # Reshape back to 3D
        denoised_stack = X_reconstructed.reshape(frames, height, width)
        
        # Ensure same data type as original
        denoised_stack = denoised_stack.astype(stack.dtype)
        
        # Plot component difference (spatial)
        plot_component_difference(stack, denoised_stack, excluded_components)
        
        # Plot temporal difference profile
        plot_temporal_difference(stack, denoised_stack, excluded_components)
        
        # Also show a visualization of just the removed components
        print("\nVisualizing the components being removed...")
        X_components = reconstruct_only_components(U, S, V, X_mean, excluded_components)
        components_stack = X_components.reshape(frames, height, width)
        
        # Plot the middle frame of the components stack
        plt.figure(figsize=(10, 8))
        comp_vmin, comp_vmax = np.percentile(components_stack[middle_frame], [1, 99])
        plt.imshow(components_stack[middle_frame], cmap='viridis', vmin=comp_vmin, vmax=comp_vmax)
        plt.colorbar(label='Component Intensity')
        plt.title(f'Components {", ".join([str(i+1) for i in sorted(excluded_components)])} Combined')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return denoised_stack, excluded_components
    else:
        print("No components excluded. Returning original stack.")
        return stack, []

def process_file(input_path, output_path=None, n_components=50, method='svd'):
    """
    Process a single TIFF file.
    
    Args:
        input_path: path to input TIFF file
        output_path: path to save output TIFF file, if None, will be auto-generated
        n_components: number of SVD components to compute
        method: 'svd' or 'pca'
        
    Returns:
        output_path: path where denoised file was saved, or None if failed
    """
    # Load TIFF stack
    stack = load_tiff(input_path)
    if stack is None:
        return None
        
    # Print original stats
    print(f"Original stack range: {np.min(stack):.2f} to {np.max(stack):.2f}")
    print(f"Original stack type: {stack.dtype}")
    
    # Interactive component removal
    denoised_stack, excluded_components = interactive_component_removal(stack, n_components, method)
    
    # Check denoised stack output
    print(f"Denoised stack range: {np.min(denoised_stack):.2f} to {np.max(denoised_stack):.2f}")
    
    # Create output paths
    input_file = Path(input_path)
    if output_path is None:
        # Save in the same directory as the input file
        output_path = str(input_file.with_name(f"{input_file.stem}_denoised{input_file.suffix}"))
    
    # Create path for components-only file
    components_path = str(input_file.with_name(f"{input_file.stem}_components{input_file.suffix}"))
    
    # Save denoised stack
    if excluded_components:
        # Ensure stack is within valid range for the data type
        if stack.dtype == np.uint8:
            denoised_stack = np.clip(denoised_stack, 0, 255)
        elif stack.dtype == np.uint16:
            denoised_stack = np.clip(denoised_stack, 0, 65535)
        
        # Ensure denoised stack is same type as original
        denoised_stack = denoised_stack.astype(stack.dtype)
        
        print(f"Final stack range: {np.min(denoised_stack):.2f} to {np.max(denoised_stack):.2f}")
        
        # Save the denoised stack
        success = save_tiff(denoised_stack, output_path)
        
        # Now save the components-only stack
        frames, height, width = stack.shape
        
        # Get the raw SVD components
        U, S, V, X_mean = compute_svd(stack, n_components)
        
        # Reconstruct using only the excluded components
        print("Reconstructing components-only stack...")
        X_components = reconstruct_only_components(U, S, V, X_mean, excluded_components)
        
        # Reshape to 3D
        components_stack = X_components.reshape(frames, height, width)
        
        # Scale to the original data range for better visualization
        orig_min, orig_max = np.min(stack), np.max(stack)
        components_stack_scaled = np.clip(components_stack, 0, None)  # Ensure no negative values
        
        # Scale the components to have the same range as original
        if np.max(components_stack_scaled) > 0:
            scale_factor = (orig_max - orig_min) / np.max(components_stack_scaled)
            components_stack_scaled = components_stack_scaled * scale_factor + orig_min
        
        # Clip and convert to original type
        if stack.dtype == np.uint8:
            components_stack_scaled = np.clip(components_stack_scaled, 0, 255)
        elif stack.dtype == np.uint16:
            components_stack_scaled = np.clip(components_stack_scaled, 0, 65535)
        
        components_stack_scaled = components_stack_scaled.astype(stack.dtype)
        
        # Save the components stack
        success_components = save_tiff(components_stack_scaled, components_path)
        
        if success and success_components:
            print(f"Components stack saved to {components_path}")
            return output_path
    else:
        print("No denoising was performed, not saving output.")
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Denoise calcium imaging videos using SVD decomposition')
    parser.add_argument('input_path', type=str, help='Path to input TIFF file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save output TIFF file')
    parser.add_argument('--n_components', type=int, default=50, help='Number of SVD components to compute')
    parser.add_argument('--method', type=str, default='svd', choices=['svd', 'pca'], help='Decomposition method')
    args = parser.parse_args()
    
    # Process file
    output_path = process_file(args.input_path, args.output_path, args.n_components, args.method)
    
    if output_path:
        print(f"Denoising completed successfully. Output saved to {output_path}")
    else:
        print("Denoising failed or was skipped.")

if __name__ == "__main__":
    main() 