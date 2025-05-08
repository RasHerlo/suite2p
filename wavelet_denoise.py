import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import tifffile
import argparse
from pathlib import Path
import time
from scipy import ndimage
from matplotlib.widgets import Slider
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

def wavelet_denoise_3d(stack, wavelet='db4', level=2, threshold=None, mode='soft'):
    """
    Apply wavelet denoising to a 3D stack.
    
    Args:
        stack: 3D numpy array (frames, height, width)
        wavelet: wavelet to use (default: 'db4')
        level: decomposition level (default: 2)
        threshold: threshold for coefficient shrinkage (None for automatic)
        mode: thresholding mode ('soft' or 'hard')
        
    Returns:
        denoised_stack: Denoised 3D stack
    """
    frames, height, width = stack.shape
    print(f"Applying wavelet denoising with {wavelet} wavelet, level {level}...")
    start_time = time.time()
    
    # Create output stack
    denoised_stack = np.zeros_like(stack)
    
    # Process each frame
    for i in range(frames):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{frames}...")
        
        # 2D wavelet transform
        coeffs = pywt.wavedec2(stack[i], wavelet, level=level)
        
        # Calculate threshold if not provided
        if threshold is None:
            # Estimate noise based on the finest detail coefficients
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(height * width))
        
        # Apply thresholding to detail coefficients
        new_coeffs = list(coeffs)
        for j in range(1, len(coeffs)):
            if isinstance(coeffs[j], tuple):
                new_detail = []
                for detail in coeffs[j]:
                    if mode == 'soft':
                        # Soft thresholding
                        thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                    else:
                        # Hard thresholding
                        thresholded = detail * (np.abs(detail) > threshold)
                    new_detail.append(thresholded)
                new_coeffs[j] = tuple(new_detail)
            else:
                if mode == 'soft':
                    # Soft thresholding
                    new_coeffs[j] = np.sign(coeffs[j]) * np.maximum(np.abs(coeffs[j]) - threshold, 0)
                else:
                    # Hard thresholding
                    new_coeffs[j] = coeffs[j] * (np.abs(coeffs[j]) > threshold)
        
        # Inverse transform
        denoised_stack[i] = pywt.waverec2(new_coeffs, wavelet)
    
    print(f"Wavelet denoising completed in {time.time() - start_time:.2f}s")
    return denoised_stack

def wavelet_denoise_timeline(signal, wavelet='sym8', level=None, threshold=None, mode='soft'):
    """
    Apply wavelet denoising to a 1D signal.
    
    Args:
        signal: 1D numpy array
        wavelet: wavelet to use (default: 'sym8')
        level: decomposition level (default: None, automatically determined)
        threshold: threshold for coefficient shrinkage (None for automatic)
        mode: thresholding mode ('soft' or 'hard')
        
    Returns:
        denoised_signal: Denoised 1D signal
    """
    # Determine max decomposition level if not provided
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        level = min(level, 5)  # Cap at level 5
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate threshold if not provided
    if threshold is None:
        # Estimate noise based on the finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply thresholding to detail coefficients
    new_coeffs = list(coeffs)
    for j in range(1, len(coeffs)):
        if mode == 'soft':
            # Soft thresholding
            new_coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
        else:
            # Hard thresholding
            new_coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='hard')
    
    # Inverse transform
    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    
    # Ensure denoised signal has same length as original
    denoised_signal = denoised_signal[:len(signal)]
    
    return denoised_signal

def temporal_denoising(stack, wavelet='sym8', level=None, threshold=None, mode='soft'):
    """
    Apply wavelet denoising along the time dimension for each pixel.
    
    Args:
        stack: 3D numpy array (frames, height, width)
        wavelet: wavelet to use (default: 'sym8')
        level: decomposition level (default: None, automatically determined)
        threshold: threshold for coefficient shrinkage (None for automatic)
        mode: thresholding mode ('soft' or 'hard')
        
    Returns:
        denoised_stack: Denoised 3D stack
    """
    frames, height, width = stack.shape
    print(f"Applying temporal wavelet denoising with {wavelet} wavelet...")
    start_time = time.time()
    
    # Create output stack
    denoised_stack = np.zeros_like(stack)
    
    # Reshape for efficient processing
    stack_reshaped = stack.reshape(frames, -1)
    denoised_reshaped = np.zeros_like(stack_reshaped)
    
    # Process each pixel's time series
    n_pixels = height * width
    update_interval = max(1, n_pixels // 100)  # Update progress every 1%
    
    for i in range(n_pixels):
        if i % update_interval == 0:
            progress = i / n_pixels * 100
            print(f"Processing pixels: {progress:.1f}% complete...")
        
        # Extract time series for this pixel
        timeline = stack_reshaped[:, i]
        
        # Apply 1D wavelet denoising
        denoised_timeline = wavelet_denoise_timeline(
            timeline, wavelet=wavelet, level=level, threshold=threshold, mode=mode
        )
        
        # Store denoised time series
        denoised_reshaped[:, i] = denoised_timeline
    
    # Reshape back to 3D
    denoised_stack = denoised_reshaped.reshape(frames, height, width)
    
    print(f"Temporal wavelet denoising completed in {time.time() - start_time:.2f}s")
    return denoised_stack

def visualize_denoising(original, denoised, frame_idx=None):
    """
    Visualize the results of denoising for a specific frame.
    
    Args:
        original: Original 3D stack
        denoised: Denoised 3D stack
        frame_idx: Index of frame to visualize (None for middle frame)
    """
    frames, height, width = original.shape
    
    # Use middle frame if not specified
    if frame_idx is None:
        frame_idx = frames // 2
    
    # Calculate difference
    diff = original[frame_idx] - denoised[frame_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine common scale
    vmin, vmax = np.percentile(original[frame_idx], [1, 99])
    
    # Original
    axes[0].imshow(original[frame_idx], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    # Denoised
    axes[1].imshow(denoised[frame_idx], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised Frame')
    axes[1].axis('off')
    
    # Difference (what was removed)
    diff_scale = 3 * np.std(diff)
    im = axes[2].imshow(diff, cmap='coolwarm', vmin=-diff_scale, vmax=diff_scale)
    axes[2].set_title('Difference (Original - Denoised)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='Difference')
    
    # Add text with statistics
    orig_mean = np.mean(original[frame_idx])
    orig_std = np.std(original[frame_idx])
    denoised_mean = np.mean(denoised[frame_idx])
    denoised_std = np.std(denoised[frame_idx])
    diff_mean = np.mean(np.abs(diff))
    diff_std = np.std(diff)
    
    fig.suptitle(f"Frame {frame_idx} | Original: mean={orig_mean:.1f}, std={orig_std:.1f} | " 
                f"Denoised: mean={denoised_mean:.1f}, std={denoised_std:.1f} | "
                f"Diff: mean={diff_mean:.1f}, std={diff_std:.1f}")
    
    plt.tight_layout()
    plt.show()

def interactive_wavelet_denoising(stack, initial_wavelet='db4', initial_level=2, initial_threshold=None, initial_mode='soft'):
    """
    Interactive wavelet denoising with parameter adjustment.
    
    Args:
        stack: 3D numpy array (frames, height, width)
        initial_wavelet: initial wavelet to use
        initial_level: initial decomposition level
        initial_threshold: initial threshold for coefficient shrinkage (None for automatic)
        initial_mode: initial thresholding mode ('soft' or 'hard')
        
    Returns:
        denoised_stack: Denoised 3D stack
        params: Dictionary of parameters used
    """
    frames, height, width = stack.shape
    
    # Select middle frame for preview
    mid_frame = frames // 2
    
    # Create figure for interactive adjustment
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    fig.subplots_adjust(bottom=0.25)
    
    # Show original frame
    axs[0].imshow(stack[mid_frame], cmap='gray')
    axs[0].set_title('Original Frame')
    axs[0].axis('off')
    
    # Apply initial denoising to get baseline
    frame_denoised = wavelet_denoise_3d(stack[mid_frame:mid_frame+1], 
                                         wavelet=initial_wavelet, 
                                         level=initial_level, 
                                         threshold=initial_threshold, 
                                         mode=initial_mode)[0]
    
    # Show denoised frame
    im_denoised = axs[1].imshow(frame_denoised, cmap='gray')
    axs[1].set_title('Denoised Frame')
    axs[1].axis('off')
    
    # Show difference
    diff = stack[mid_frame] - frame_denoised
    diff_scale = 3 * np.std(diff)
    im_diff = axs[2].imshow(diff, cmap='coolwarm', vmin=-diff_scale, vmax=diff_scale)
    axs[2].set_title('Difference (Removed Noise)')
    axs[2].axis('off')
    plt.colorbar(im_diff, ax=axs[2], label='Difference')
    
    # Add wavelets dropdown
    wavelet_list = ['haar', 'db1', 'db2', 'db4', 'db8', 'sym4', 'sym8', 'coif3', 'bior2.2', 'bior4.4']
    ax_wavelet = plt.axes([0.15, 0.15, 0.2, 0.03])
    wavelet_slider = plt.Slider(
        ax_wavelet, 'Wavelet', 0, len(wavelet_list)-1,
        valinit=wavelet_list.index(initial_wavelet),
        valstep=1
    )
    wavelet_slider.valtext.set_text(initial_wavelet)
    
    # Add level slider
    ax_level = plt.axes([0.15, 0.1, 0.2, 0.03])
    level_slider = plt.Slider(
        ax_level, 'Level', 1, 5,
        valinit=initial_level,
        valstep=1
    )
    
    # Add threshold slider
    ax_threshold = plt.axes([0.6, 0.15, 0.2, 0.03])
    
    # Calculate automatic threshold for initial setting
    coeffs = pywt.wavedec2(stack[mid_frame], initial_wavelet, level=initial_level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    auto_threshold = sigma * np.sqrt(2 * np.log(height * width))
    
    threshold_slider = plt.Slider(
        ax_threshold, 'Threshold', 0, auto_threshold*3,
        valinit=auto_threshold,
        valfmt='%0.1f'
    )
    
    # Add mode selection
    ax_mode = plt.axes([0.6, 0.1, 0.2, 0.03])
    mode_slider = plt.Slider(
        ax_mode, 'Mode (0=Soft, 1=Hard)', 0, 1,
        valinit=0 if initial_mode == 'soft' else 1,
        valstep=1
    )
    
    # Update function
    def update(val):
        wavelet = wavelet_list[int(wavelet_slider.val)]
        wavelet_slider.valtext.set_text(wavelet)
        level = int(level_slider.val)
        threshold = threshold_slider.val
        mode = 'soft' if mode_slider.val == 0 else 'hard'
        
        # Apply denoising
        frame_denoised = wavelet_denoise_3d(stack[mid_frame:mid_frame+1], 
                                           wavelet=wavelet, 
                                           level=level, 
                                           threshold=threshold, 
                                           mode=mode)[0]
        
        # Update plots
        im_denoised.set_data(frame_denoised)
        
        # Update difference
        diff = stack[mid_frame] - frame_denoised
        diff_scale = 3 * np.std(diff)
        im_diff.set_data(diff)
        im_diff.set_clim(-diff_scale, diff_scale)
        
        fig.canvas.draw_idle()
    
    # Connect update function
    wavelet_slider.on_changed(update)
    level_slider.on_changed(update)
    threshold_slider.on_changed(update)
    mode_slider.on_changed(update)
    
    # Show the window
    plt.show()
    
    # Get final parameters
    wavelet = wavelet_list[int(wavelet_slider.val)]
    level = int(level_slider.val)
    threshold = threshold_slider.val
    mode = 'soft' if mode_slider.val == 0 else 'hard'
    
    # Apply final denoising to full stack
    print(f"\nApplying final denoising with these parameters:")
    print(f"  Wavelet: {wavelet}")
    print(f"  Level: {level}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Mode: {mode}")
    
    # Ask the user which denoising approach to use
    while True:
        approach = input("\nChoose denoising approach (1=Frame-by-frame, 2=Temporal): ")
        if approach == '1':
            denoised_stack = wavelet_denoise_3d(stack, wavelet=wavelet, level=level, 
                                             threshold=threshold, mode=mode)
            break
        elif approach == '2':
            denoised_stack = temporal_denoising(stack, wavelet=wavelet, level=level, 
                                             threshold=threshold, mode=mode)
            break
        else:
            print("Invalid option. Please enter 1 or 2.")
    
    # Return the denoised stack and parameters
    params = {
        'wavelet': wavelet,
        'level': level,
        'threshold': threshold,
        'mode': mode,
        'approach': 'frame-by-frame' if approach == '1' else 'temporal'
    }
    
    return denoised_stack, params

def noise_only_stack(original, denoised):
    """
    Create a stack containing only the removed noise.
    
    Args:
        original: Original stack
        denoised: Denoised stack
        
    Returns:
        noise_stack: Stack containing only the noise
    """
    return original - denoised

def process_file(input_path, output_path=None, interactive=True):
    """
    Process a single TIFF file with wavelet denoising.
    
    Args:
        input_path: path to input TIFF file
        output_path: path to save output TIFF file, if None, will be auto-generated
        interactive: whether to use interactive parameter selection
        
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
    
    # Create output paths
    input_file = Path(input_path)
    if output_path is None:
        # Save in the same directory as the input file
        output_path = str(input_file.with_name(f"{input_file.stem}_wavelet_denoised{input_file.suffix}"))
    
    # Create path for noise-only file
    noise_path = str(input_file.with_name(f"{input_file.stem}_wavelet_noise{input_file.suffix}"))
    
    # Apply denoising
    if interactive:
        denoised_stack, params = interactive_wavelet_denoising(stack)
    else:
        # Use default parameters
        denoised_stack = wavelet_denoise_3d(stack)
        params = {
            'wavelet': 'db4',
            'level': 2,
            'threshold': None,
            'mode': 'soft',
            'approach': 'frame-by-frame'
        }
    
    # Check denoised stack output
    print(f"Denoised stack range: {np.min(denoised_stack):.2f} to {np.max(denoised_stack):.2f}")
    
    # Ensure stack is within valid range for the data type
    if stack.dtype == np.uint8:
        denoised_stack = np.clip(denoised_stack, 0, 255)
    elif stack.dtype == np.uint16:
        denoised_stack = np.clip(denoised_stack, 0, 65535)
    
    # Ensure denoised stack is same type as original
    denoised_stack = denoised_stack.astype(stack.dtype)
    
    print(f"Final stack range: {np.min(denoised_stack):.2f} to {np.max(denoised_stack):.2f}")
    
    # Save the denoised stack
    success_denoised = save_tiff(denoised_stack, output_path)
    
    # Create and save the noise-only stack
    noise_stack = noise_only_stack(stack, denoised_stack)
    
    # Scale noise to be in valid range for visualization
    noise_scale = np.std(noise_stack) * 3
    noise_stack_scaled = ((noise_stack + noise_scale) / (2 * noise_scale) * (np.max(stack) - np.min(stack))) + np.min(stack)
    noise_stack_scaled = np.clip(noise_stack_scaled, np.min(stack), np.max(stack))
    noise_stack_scaled = noise_stack_scaled.astype(stack.dtype)
    
    success_noise = save_tiff(noise_stack_scaled, noise_path)
    
    # Visualize results
    visualize_denoising(stack, denoised_stack)
    
    if success_denoised and success_noise:
        print(f"Denoised stack saved to {output_path}")
        print(f"Noise stack saved to {noise_path}")
        return output_path
    else:
        print("Error saving outputs.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Denoise calcium imaging videos using wavelet transform')
    parser.add_argument('input_path', type=str, help='Path to input TIFF file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save output TIFF file')
    parser.add_argument('--non-interactive', action='store_true', help='Run with default parameters (non-interactive)')
    args = parser.parse_args()
    
    # Process file
    output_path = process_file(args.input_path, args.output_path, interactive=not args.non_interactive)
    
    if output_path:
        print(f"Wavelet denoising completed successfully. Output saved to {output_path}")
    else:
        print("Wavelet denoising failed.")

if __name__ == "__main__":
    main() 