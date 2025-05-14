from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import matplotlib
matplotlib.use("QtAgg")
matplotlib.interactive(True)
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector, TextBox
from matplotlib.patches import Rectangle, Circle
import colorcet as cc
import sys
import os
from tkinter import Tk, filedialog

# Variables to store our data and processing state
images = None
fft_images = None
masked_fft = None
denoised_images = None
num_frames = 0
current_frame = 0
tiff_path = None
output_filename = None
mask_regions = []
circle_masks = []  # To store circular masks [inner_radius, outer_radius]
endpoints = None
midpoints = None

# GUI elements
fig = None
# Top row (logarithmic)
ax_fft = None
ax_fft_zoom = None
ax_phase = None
ax_original = None
ax_denoised = None
# Bottom row (linear)
ax_fft_linear = None
ax_fft_zoom_linear = None
ax_phase_linear = None
# Plot objects
fft_plot = None
fft_zoom_plot = None
phase_plot = None
original_plot = None
denoised_plot = None
# Linear plot objects
fft_linear_plot = None
fft_zoom_linear_plot = None
phase_linear_plot = None
# Circular mask controls
inner_radius_textbox = None
outer_radius_textbox = None
current_inner_radius = 0
current_outer_radius = 10
# Other elements
frame_slider = None
no_file_text = None
selector = None
mask_patches = []  # To store visualization rectangles
circle_patches = []  # To store visualization circles
current_mode = 'draw'  # 'draw' or 'remove'

# Function to load a TIFF file and compute FFT
def load_file(file_path):
    global images, fft_images, masked_fft, denoised_images, num_frames
    global tiff_path, output_filename, mask_regions, circle_masks, endpoints, midpoints
    
    # Store file path and set output filename
    tiff_path = file_path
    output_filename = os.path.splitext(tiff_path)[0] + "_denoised.tif"
    print(f"Processing file: {tiff_path}")
    print(f"Denoised images will be saved to: {output_filename}")
    
    # Clear existing masks
    mask_regions = []
    circle_masks = []  # Clear circular masks too
    
    # Load images
    try:
        images = imread(Path(tiff_path), is_ome=False)
        images -= images.min()
        num_frames = images.shape[0]
        
        # Compute FFT of all images
        fft_images = np.fft.fft2(images)
        fft_images = np.fft.fftshift(fft_images)
        
        # Create a copy of fft_images for manipulation
        masked_fft = fft_images.copy()
        
        # Calculate endpoints and midpoints for masking
        endpoints = np.asarray(images.shape[1:])
        midpoints = endpoints // 2
        
        # Initial mask regions removed - no default masks
        
        # Prepare initial denoised images - now without any masks applied
        reconstructed_images = np.fft.ifftshift(masked_fft)
        reconstructed_images = np.fft.ifft2(reconstructed_images)
        denoised_images = np.abs(reconstructed_images)
        
        return True
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return False

# Function to update the plots based on the current frame
def update_plots(frame):
    global fft_plot, fft_zoom_plot, phase_plot, original_plot, denoised_plot
    global fft_linear_plot, fft_zoom_linear_plot, phase_linear_plot
    
    if images is None or images.size == 0 or frame >= images.shape[0]:
        return
    
    # Update original and denoised images
    original_plot.set_data(images[frame])
    denoised_plot.set_data(denoised_images[frame])
    
    # Get dimensions
    h, w = images.shape[1], images.shape[2]
    center_y, center_x = h // 2, w // 2
    zoom_size = 20  # 20 pixels in each direction from center
    
    # Extract the region centered on the FFT center
    y_start = max(0, center_y - zoom_size)
    y_end = min(h, center_y + zoom_size + 1)
    x_start = max(0, center_x - zoom_size)
    x_end = min(w, center_x + zoom_size + 1)
    
    # Calculate FFT amplitude (log and linear)
    fft_amplitude = np.abs(fft_images[frame])
    spectrum_log = np.log(fft_amplitude + 1e-6)  # Log scale for better visualization
    spectrum_linear = fft_amplitude  # Linear scale - true amplitude values
    
    # Update logarithmic FFT plots (top row)
    fft_plot.set_data(spectrum_log)
    fft_zoom_plot.set_data(spectrum_log[y_start:y_end, x_start:x_end])
    
    # Update linear FFT plots (bottom row)
    fft_linear_plot.set_data(spectrum_linear)
    fft_zoom_linear_plot.set_data(spectrum_linear[y_start:y_end, x_start:x_end])
    
    # Update phase image
    phase_data = np.angle(fft_images[frame])
    phase_plot.set_data(phase_data)
    phase_linear_plot.set_data(phase_data)  # Same phase data for both rows
    
    # Set extents properly to fill out the subplot areas
    original_plot.set_extent([0, w, h, 0])
    denoised_plot.set_extent([0, w, h, 0])
    fft_plot.set_extent([0, w, h, 0])
    fft_linear_plot.set_extent([0, w, h, 0])
    phase_plot.set_extent([0, w, h, 0])
    phase_linear_plot.set_extent([0, w, h, 0])
    
    # For the zoom plots, explicitly set the extent in coordinates that make sense
    # for FFT visualization - with (0,0) at the center
    zoom_h, zoom_w = spectrum_log[y_start:y_end, x_start:x_end].shape
    fft_zoom_plot.set_extent([-zoom_size, zoom_size, -zoom_size, zoom_size])
    fft_zoom_linear_plot.set_extent([-zoom_size, zoom_size, -zoom_size, zoom_size])
    
    # IMPORTANT: Update color limits for the linear plots
    # For linear plots, we need strong contrast adjustment because FFT amplitude has extreme dynamic range
    # Find the DC component (center value) for reference
    dc_value = fft_amplitude[center_y, center_x]
    
    # Use a small fraction of the DC value as the maximum for better visibility of non-DC components
    linear_vmax = dc_value * 0.05  # Only show up to 5% of the DC value
    fft_linear_plot.set_clim(vmin=0, vmax=linear_vmax)
    
    # For zoomed linear plot, use an even smaller range to see the details
    zoom_max = np.max(spectrum_linear[y_start:y_end, x_start:x_end])
    if zoom_max > 0:
        fft_zoom_linear_plot.set_clim(vmin=0, vmax=zoom_max * 0.5)  # Use 50% of max value in zoom region
    
    # Redraw
    fig.canvas.draw_idle()

# Slider callback
def on_slider_change(val):
    global current_frame
    current_frame = int(min(val, num_frames-1))
    update_plots(current_frame)

# Function to add or remove mask regions and update the visuals
def apply_masks():
    global masked_fft, denoised_images
    
    if fft_images is None or fft_images.size == 0:
        return
    
    # Start with a fresh copy of the FFT images
    masked_fft = fft_images.copy()
    
    # Apply all rectangular masks
    for region in mask_regions:
        y1, y2, x1, x2 = region
        masked_fft[:, y1:y2, x1:x2] = 0
    
    # Apply all circular masks
    if len(circle_masks) > 0:
        # Get image dimensions
        h, w = images.shape[1], images.shape[2]
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids centered at FFT center
        y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
        
        # Apply each circular mask
        for inner_radius, outer_radius in circle_masks:
            # Calculate squared distances for efficiency
            dist_squared = x*x + y*y
            
            # Create mask: 1 for areas to keep, 0 for areas to mask
            # Mask between inner_radius and outer_radius
            mask = (dist_squared < inner_radius**2) | (dist_squared > outer_radius**2)
            
            # Apply mask to all frames
            for i in range(masked_fft.shape[0]):
                masked_fft[i][~mask] = 0
    
    # Recalculate denoised images
    reconstructed_images = np.fft.ifftshift(masked_fft)
    reconstructed_images = np.fft.ifft2(reconstructed_images)
    denoised_images = np.abs(reconstructed_images)
    
    # Update the denoised plot
    denoised_plot.set_data(denoised_images[current_frame])
    
    # Update the FFT plot
    masked_spectrum = np.log(np.abs(masked_fft[current_frame]) + 1e-6)
    fft_plot.set_data(masked_spectrum)
    
    # Update the phase plot
    masked_phase = np.angle(masked_fft[current_frame])
    phase_plot.set_data(masked_phase)
    
    fig.canvas.draw_idle()

# Function to show mask visualizations
def show_mask_visualizations():
    global mask_patches
    
    # Clear any existing visualizations
    clear_mask_visualizations()
    
    # Add a rectangle for each mask region
    for region in mask_regions:
        y1, y2, x1, x2 = region
        width = x2 - x1
        height = y2 - y1
        
        # Create a semi-transparent red rectangle
        rect = Rectangle((x1, y1), width, height, 
                         fill=True, alpha=0.3, color='red', 
                         linestyle='dashed', linewidth=1)
        
        # Add it to the plot
        ax_fft.add_patch(rect)
        mask_patches.append(rect)
    
    fig.canvas.draw_idle()

# Function to clear mask visualizations
def clear_mask_visualizations():
    global mask_patches
    
    # Remove all visualization rectangles
    for patch in mask_patches:
        patch.remove()
    
    mask_patches = []
    fig.canvas.draw_idle()

# Function to show circular mask visualizations with semi-transparent green fill
def show_circle_visualizations():
    global circle_patches
    
    # Clear any existing circle visualizations
    clear_circle_visualizations()
    
    # Add circle patches for all circular masks
    for inner_radius, outer_radius in circle_masks:
        # Create circles for the zoomed views
        # Inner circle - solid blue outline
        ax_fft_zoom.add_patch(plt.Circle((0, 0), inner_radius, 
                                       fill=False, color='blue', 
                                       linestyle='solid', linewidth=1.5))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), inner_radius, 
                                            fill=False, color='blue', 
                                            linestyle='solid', linewidth=1.5))
        
        # Outer circle - solid red outline
        ax_fft_zoom.add_patch(plt.Circle((0, 0), outer_radius, 
                                       fill=False, color='red', 
                                       linestyle='solid', linewidth=1.5))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), outer_radius, 
                                            fill=False, color='red', 
                                            linestyle='solid', linewidth=1.5))
        
        # Fill between inner and outer with semi-transparent green
        # This creates a filled ring shape
        donut = plt.Circle((0, 0), outer_radius, 
                           fill=True, color='green', alpha=0.3)
        inner_mask = plt.Circle((0, 0), inner_radius, 
                                fill=True, color='white', alpha=1.0)
        
        # Add the semi-transparent mask to zoomed views
        ax_fft_zoom.add_patch(donut)
        ax_fft_zoom.add_patch(inner_mask)
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), outer_radius, 
                                            fill=True, color='green', alpha=0.3))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), inner_radius, 
                                            fill=True, color='white', alpha=1.0))
        
        # Store reference to patches for later removal
        circle_patches.extend([donut, inner_mask])
        
        # Also add circles to main FFT views (need to translate coordinates)
        if images is not None and images.size > 0:
            h, w = images.shape[1], images.shape[2]
            center_y, center_x = h // 2, w // 2
            
            # Main views - solid outlines
            ax_fft.add_patch(plt.Circle((center_x, center_y), inner_radius, 
                                      fill=False, color='blue', 
                                      linestyle='solid', linewidth=1.5))
            ax_fft.add_patch(plt.Circle((center_x, center_y), outer_radius, 
                                      fill=False, color='red', 
                                      linestyle='solid', linewidth=1.5))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), inner_radius, 
                                          fill=False, color='blue', 
                                          linestyle='solid', linewidth=1.5))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), outer_radius, 
                                          fill=False, color='red', 
                                          linestyle='solid', linewidth=1.5))
            
            # Semi-transparent green mask in main views
            main_donut = plt.Circle((center_x, center_y), outer_radius, 
                                  fill=True, color='green', alpha=0.3)
            main_inner_mask = plt.Circle((center_x, center_y), inner_radius, 
                                       fill=True, color='white', alpha=1.0)
            
            ax_fft.add_patch(main_donut)
            ax_fft.add_patch(main_inner_mask)
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), outer_radius, 
                                          fill=True, color='green', alpha=0.3))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), inner_radius, 
                                          fill=True, color='white', alpha=1.0))
            
            # Store reference to main patches
            circle_patches.extend([main_donut, main_inner_mask])
            
            print(f"Added circle visualization: inner={inner_radius:.1f}, outer={outer_radius:.1f}")
    
    # Update the display
    fig.canvas.draw_idle()

# Function to clear circular mask visualizations
def clear_circle_visualizations():
    global circle_patches
    
    # Remove all visualization circles
    for patch in circle_patches:
        try:
            patch.remove()
        except:
            # Some patches might have been removed already, just continue
            pass
    
    circle_patches = []
    fig.canvas.draw_idle()

# Function to add a new circular mask from the textbox values
def add_circle_mask():
    global current_inner_radius, current_outer_radius, circle_masks
    
    # Check that inner radius is less than outer radius
    if current_inner_radius >= current_outer_radius:
        print("Inner radius must be less than outer radius")
        return
    
    # Add the new mask and update
    circle_masks.append([current_inner_radius, current_outer_radius])
    print(f"Added circle mask: inner={current_inner_radius:.1f}, outer={current_outer_radius:.1f}")
    
    # Update visualizations and apply the mask
    show_circle_visualizations()
    apply_masks()

# Callback for inner radius textbox
def on_inner_radius_change(text):
    global current_inner_radius
    try:
        value = float(text)
        if value >= 0:
            current_inner_radius = value
            print(f"Inner radius set to {value:.1f}")
        else:
            print("Inner radius must be positive")
    except ValueError:
        print("Please enter a valid number for inner radius")

# Callback for outer radius textbox
def on_outer_radius_change(text):
    global current_outer_radius
    try:
        value = float(text)
        if value > 0:  # Must be positive and non-zero
            current_outer_radius = value
            print(f"Outer radius set to {value:.1f}")
        else:
            print("Outer radius must be positive")
    except ValueError:
        print("Please enter a valid number for outer radius")

# Function to clear all circle masks
def clear_circle_masks():
    global circle_masks
    circle_masks = []
    clear_circle_visualizations()
    apply_masks()
    print("All circle masks cleared")

# Function to show both rectangular and circular mask visualizations
def show_all_mask_visualizations():
    show_mask_visualizations()
    show_circle_visualizations()
    
# Function to clear all mask visualizations
def clear_all_mask_visualizations():
    clear_mask_visualizations()
    clear_circle_visualizations()

# Function to load a new TIFF file
def on_load_click(event):
    global no_file_text, frame_slider
    
    # Create root Tk window and hide it
    root = Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select TIFF File",
        filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")]
    )
    
    # Close the Tk instance
    root.destroy()
    
    # If a file was selected
    if file_path:
        # Initialize the file
        success = load_file(file_path)
        
        if success:
            # Hide the "no file loaded" text
            no_file_text.set_visible(False)
            
            # Reset slider
            frame_slider.valmax = num_frames - 1
            frame_slider.ax.set_xlim(0, num_frames - 1)
            frame_slider.set_val(0)
            
            # Update plots with the new data
            # Recalculate display ranges for intensity scaling
            ivmin = np.percentile(images, 0)
            ivmax = np.percentile(images, 99)
            spectrum = np.log(np.abs(fft_images[0]) + 1e-6)
            fvmin = np.percentile(spectrum, 0)
            fvmax = np.percentile(spectrum, 99)
            
            # Update the data limits for the plots
            original_plot.set_clim(vmin=ivmin, vmax=ivmax)
            denoised_plot.set_clim(vmin=ivmin, vmax=ivmax)
            fft_plot.set_clim(vmin=fvmin, vmax=fvmax)
            fft_zoom_plot.set_clim(vmin=fvmin, vmax=fvmax)
            phase_plot.set_clim(vmin=-np.pi, vmax=np.pi)
            
            # Set the data for all plots
            update_plots(0)
            
            # Force redraw
            plt.draw()
            fig.canvas.flush_events()
            
            # Clear any mask visualizations
            clear_all_mask_visualizations()
            
            print(f"Loaded file: {file_path}")
        else:
            print("Failed to load file")
    else:
        print("No file selected")

# Callback functions for the buttons
def on_draw_click(event):
    global current_mode, selector
    current_mode = 'draw'
    if selector:
        selector.set_active(True)
    # Clear mask visualizations when switching to draw mode
    clear_all_mask_visualizations()
    print("Draw mode activated")

def on_remove_click(event):
    global current_mode, selector
    current_mode = 'remove'
    if selector:
        selector.set_active(True)
    # Show mask visualizations when switching to remove mode
    show_all_mask_visualizations()
    print("Remove mode activated")

def on_circle_click(event):
    global current_mode, selector
    current_mode = 'circle'
    if selector:
        selector.set_active(False)  # Disable rectangle selector
    # Clear mask visualizations when switching to circle mode
    clear_all_mask_visualizations()
    print("Circle draw mode activated")

def on_adjust_circle_click(event):
    global current_mode, selector
    current_mode = 'adjust_circle'
    if selector:
        selector.set_active(False)  # Disable rectangle selector
    # Show circle visualizations when switching to adjust mode
    show_circle_visualizations()
    print("Circle adjust mode activated")

def on_update_click(event):
    global current_inner_radius, current_outer_radius
    
    # Update the circular mask values from textboxes
    try:
        inner_val = float(inner_radius_textbox.text)
        outer_val = float(outer_radius_textbox.text)
        
        # Validate values
        if inner_val >= 0 and outer_val > 0 and inner_val < outer_val:
            current_inner_radius = inner_val
            current_outer_radius = outer_val
            print(f"Updated radius values: inner={inner_val:.1f}, outer={outer_val:.1f}")
        else:
            print("Invalid radius values: inner must be ≥ 0, outer must be > 0, and inner must be < outer")
    except ValueError:
        print("Invalid radius input. Please enter valid numbers.")
    
    # Clear visualizations before updating
    clear_all_mask_visualizations()
    
    # Apply masks
    apply_masks()
    
    # If we're in remove mode, redraw the visualizations after update
    if current_mode == 'remove':
        show_all_mask_visualizations()
    
    print("Masks updated")

# Function to save the denoised and FFT images
def save_images():
    if tiff_path == "No file loaded" or denoised_images is None:
        print("No file to save")
        return
    
    # Save denoised images
    imwrite(output_filename, denoised_images.astype(np.float32))
    print(f"Saved denoised images to {output_filename}")
    
    # Prepare log-amplitude FFT images
    fft_stack = np.zeros_like(images, dtype=np.float32)
    for i in range(num_frames):
        fft_stack[i] = np.log(np.abs(fft_images[i]) + 1e-6)
    
    # Create output filename for FFT stack
    fft_output_filename = os.path.splitext(tiff_path)[0] + "_FFT.tif"
    
    # Save FFT stack
    imwrite(fft_output_filename, fft_stack.astype(np.float32))
    print(f"Saved FFT amplitude spectrum to {fft_output_filename}")
    
    # Prepare phase FFT images
    phase_stack = np.zeros_like(images, dtype=np.float32)
    for i in range(num_frames):
        phase_stack[i] = np.angle(fft_images[i])
    
    # Create output filename for phase stack
    phase_output_filename = os.path.splitext(tiff_path)[0] + "_phase.tif"
    
    # Save phase stack
    imwrite(phase_output_filename, phase_stack.astype(np.float32))
    print(f"Saved FFT phase spectrum to {phase_output_filename}")

# Callback for the save button
def on_save_click(event):
    save_images()

# Function to create a rectangle selector
def toggle_selector(event):
    print(' Key pressed:', event.key)
    if event.key == 't':
        if selector.active:
            print('Selector deactivated')
            selector.set_active(False)
        else:
            print('Selector activated')
            selector.set_active(True)

# Callback for the rectangle selector
def on_select(eclick, erelease):
    global mask_regions
    
    if eclick.xdata is None or erelease.xdata is None:
        return
        
    # Get coordinates in data space
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    # Ensure x1 < x2 and y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Add or remove the region based on the current mode
    if current_mode == 'draw':
        print(f"Adding mask region: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
        mask_regions.append([y1, y2, x1, x2])
    else:  # Remove mode
        # First, clear existing visualizations
        clear_mask_visualizations()
        
        # Find all regions that overlap with the selection
        regions_to_remove = []
        for i, region in enumerate(mask_regions):
            ry1, ry2, rx1, rx2 = region
            # Check for overlap
            if (rx1 < x2 and rx2 > x1 and ry1 < y2 and ry2 > y1):
                regions_to_remove.append(i)
        
        # Remove the regions (in reverse order to avoid index issues)
        for i in sorted(regions_to_remove, reverse=True):
            print(f"Removing mask region: {mask_regions[i]}")
            del mask_regions[i]
        
        # Redraw the remaining mask visualizations
        show_mask_visualizations()

# Set up a callback to save when the figure is closed
def on_close(event):
    if tiff_path != "No file loaded" and denoised_images is not None:
        print("Figure closed. Saving images...")
        # Clear any visualizations before saving
        clear_all_mask_visualizations()
        save_images()

# Function to handle mouse press for circle drawing
def on_press(event):
    global drawing_circle, current_circle
    
    # Only respond to events in the linear zoom window and in circle mode
    if event.inaxes != ax_fft_zoom_linear or current_mode not in ['circle', 'adjust_circle']:
        return
    
    # Calculate distance from center (0,0)
    dist = np.sqrt(event.xdata**2 + event.ydata**2)
    print(f"Mouse press at distance {dist:.2f} from center")
    
    # Check if we're in adjust mode and close to an existing circle
    if current_mode == 'adjust_circle' and circle_masks:
        # Find the closest circle edge to adjust
        closest_dist = float('inf')
        closest_idx = -1
        closest_is_inner = True
        
        for idx, (inner_r, outer_r) in enumerate(circle_masks):
            # Calculate distance from click to inner and outer circles
            inner_dist = abs(dist - inner_r)
            outer_dist = abs(dist - outer_r)
            
            # Update if this is closer than previous closest
            if inner_dist < closest_dist:
                closest_dist = inner_dist
                closest_idx = idx
                closest_is_inner = True
            
            if outer_dist < closest_dist:
                closest_dist = outer_dist
                closest_idx = idx
                closest_is_inner = False
        
        # If we're close enough to a circle edge (within 2 pixels)
        if closest_dist < 2.0 and closest_idx >= 0:
            drawing_circle = True
            # Store the current circle info for adjustment
            current_circle = {
                'index': closest_idx,
                'is_inner': closest_is_inner,
                'other_radius': circle_masks[closest_idx][0 if not closest_is_inner else 1]
            }
            print(f"Adjusting {'inner' if closest_is_inner else 'outer'} radius of circle {closest_idx}")
            # Clear visualizations while adjusting
            clear_circle_visualizations()
            return
    
    # For drawing a new circle in 'circle' mode
    if current_mode == 'circle':
        drawing_circle = True
        current_circle = {
            'index': -1,  # New circle
            'start_radius': dist,
            'is_inner': event.button == 3  # Right-click (button 3) for inner radius
        }
        print(f"Starting to draw {'inner' if event.button == 3 else 'outer'} radius at {dist:.2f}")

# Function to handle mouse release for circle drawing
def on_release(event):
    global drawing_circle, current_circle, circle_masks
    
    if not drawing_circle or event.inaxes != ax_fft_zoom_linear:
        return
    
    # Calculate final radius
    radius = np.sqrt(event.xdata**2 + event.ydata**2)
    print(f"Mouse release at distance {radius:.2f} from center")
    
    # Different behavior based on mode
    if current_mode == 'circle' and current_circle and 'start_radius' in current_circle:
        # For new circle: determine inner and outer radii
        start_radius = current_circle['start_radius']
        is_inner = current_circle['is_inner']
        
        inner_radius = min(start_radius, radius) if is_inner else 0
        outer_radius = max(start_radius, radius) if not is_inner else radius
        
        # Enforce minimum difference between inner and outer
        if outer_radius - inner_radius < 2:
            outer_radius = inner_radius + 2
        
        # Add new circle mask
        circle_masks.append([inner_radius, outer_radius])
        print(f"Added circle mask: inner={inner_radius:.1f}, outer={outer_radius:.1f}")
    
    elif current_mode == 'adjust_circle' and current_circle:
        # For adjusting existing circle
        idx = current_circle['index']
        is_inner = current_circle['is_inner']
        other_radius = current_circle['other_radius']
        
        if is_inner:
            inner_radius = radius
            outer_radius = other_radius
            # Ensure inner < outer
            if inner_radius >= outer_radius:
                inner_radius = outer_radius - 2
        else:
            inner_radius = other_radius
            outer_radius = radius
            # Ensure outer > inner
            if outer_radius <= inner_radius:
                outer_radius = inner_radius + 2
        
        # Update the circle mask
        circle_masks[idx] = [inner_radius, outer_radius]
        print(f"Adjusted circle mask: inner={inner_radius:.1f}, outer={outer_radius:.1f}")
    
    # Reset drawing state
    drawing_circle = False
    current_circle = None
    
    # Apply the masks to update the denoised image
    apply_masks()
    
    # Show updated visualizations
    if current_mode == 'adjust_circle':
        show_circle_visualizations()

# Function to handle mouse motion for circle drawing
def on_motion(event):
    global drawing_circle, current_circle, circle_patches
    
    if not drawing_circle or event.inaxes != ax_fft_zoom_linear:
        return
    
    # Calculate current radius
    radius = np.sqrt(event.xdata**2 + event.ydata**2)
    
    # Clear previous temporary visualization
    clear_circle_visualizations()
    
    # Show all existing circles
    for inner_r, outer_r in circle_masks:
        # Skip the one being adjusted
        if current_circle and current_circle.get('index', -1) >= 0:
            idx = current_circle['index']
            if [inner_r, outer_r] == circle_masks[idx]:
                continue
                
        # Create visualization for other circles
        inner_circle_zoom = plt.Circle((0, 0), inner_r, fill=False, color='blue', linestyle='dashed')
        outer_circle_zoom = plt.Circle((0, 0), outer_r, fill=False, color='red', linestyle='dashed')
        
        # Add to zoom plots - create new circle objects for each plot
        ax_fft_zoom.add_patch(plt.Circle((0, 0), inner_r, fill=False, color='blue', linestyle='dashed'))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), inner_r, fill=False, color='blue', linestyle='dashed'))
        ax_fft_zoom.add_patch(plt.Circle((0, 0), outer_r, fill=False, color='red', linestyle='dashed'))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), outer_r, fill=False, color='red', linestyle='dashed'))
        
        # Store for later removal
        circle_patches.extend([inner_circle_zoom, outer_circle_zoom])
        
        # Also add circles to main FFT plots (need to convert coordinates)
        if images is not None and images.size > 0:
            h, w = images.shape[1], images.shape[2]
            center_y, center_x = h // 2, w // 2
            
            # Add circles to main FFT log plot
            inner_circle_main = plt.Circle((center_x, center_y), inner_r, 
                                           fill=False, color='blue', 
                                           linestyle='dashed', linewidth=1)
            outer_circle_main = plt.Circle((center_x, center_y), outer_r, 
                                            fill=False, color='red', 
                                            linestyle='dashed', linewidth=1)
            
            ax_fft.add_patch(inner_circle_main)
            ax_fft.add_patch(outer_circle_main)
            
            # Add circles to main FFT linear plot
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), inner_r, 
                                              fill=False, color='blue', 
                                              linestyle='dashed', linewidth=1))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), outer_r, 
                                              fill=False, color='red', 
                                              linestyle='dashed', linewidth=1))
            
            # Store for later removal
            circle_patches.extend([inner_circle_main, outer_circle_main])
    
    # Different behavior based on the operation
    if current_mode == 'circle' and current_circle and 'start_radius' in current_circle:
        # For new circle
        start_radius = current_circle['start_radius']
        is_inner = current_circle['is_inner']
        
        inner_radius = min(start_radius, radius) if is_inner else 0
        outer_radius = max(start_radius, radius) if not is_inner else radius
        
        # Show the current drawing - create new circle objects for each plot
        # Zoomed views
        ax_fft_zoom.add_patch(plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
        ax_fft_zoom.add_patch(plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
        
        # Store temporary circle objects for later removal
        new_inner_circle = plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2)
        new_outer_circle = plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2)
        circle_patches.extend([new_inner_circle, new_outer_circle])
        
        # Also add to main FFT plots
        if images is not None and images.size > 0:
            h, w = images.shape[1], images.shape[2]
            center_y, center_x = h // 2, w // 2
            
            # Main views
            ax_fft.add_patch(plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
            ax_fft.add_patch(plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
            
            # Store main circle objects for later removal
            main_inner_circle = plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2)
            main_outer_circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2)
            circle_patches.extend([main_inner_circle, main_outer_circle])
    
    elif current_mode == 'adjust_circle' and current_circle:
        # For adjusting existing circle
        idx = current_circle['index']
        is_inner = current_circle['is_inner']
        other_radius = current_circle['other_radius']
        
        if is_inner:
            inner_radius = radius
            outer_radius = other_radius
        else:
            inner_radius = other_radius
            outer_radius = radius
        
        # Show the current adjustment - create new circle objects for each plot
        # Zoomed views
        ax_fft_zoom.add_patch(plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
        ax_fft_zoom.add_patch(plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
        ax_fft_zoom_linear.add_patch(plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
        
        # Store temporary circle objects for later removal
        adj_inner_circle = plt.Circle((0, 0), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2)
        adj_outer_circle = plt.Circle((0, 0), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2)
        circle_patches.extend([adj_inner_circle, adj_outer_circle])
        
        # Also add to main FFT plots
        if images is not None and images.size > 0:
            h, w = images.shape[1], images.shape[2]
            center_y, center_x = h // 2, w // 2
            
            # Main views
            ax_fft.add_patch(plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
            ax_fft.add_patch(plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2))
            ax_fft_linear.add_patch(plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2))
            
            # Store main circle objects for later removal
            main_inner_circle = plt.Circle((center_x, center_y), inner_radius, fill=False, color='blue', linestyle='solid', linewidth=2)
            main_outer_circle = plt.Circle((center_x, center_y), outer_radius, fill=False, color='red', linestyle='solid', linewidth=2)
            circle_patches.extend([main_inner_circle, main_outer_circle])
    
    # Update the display
    fig.canvas.draw_idle()

# Function to handle removing circular masks
def remove_circle_at_position(x, y):
    if not circle_masks:
        return False
    
    # Calculate distance from center
    dist = np.sqrt(x**2 + y**2)
    
    # Find circles that contain this point
    circles_to_remove = []
    for idx, (inner_r, outer_r) in enumerate(circle_masks):
        if inner_r <= dist <= outer_r:
            circles_to_remove.append(idx)
    
    # Remove the circles (in reverse order to avoid index issues)
    for idx in sorted(circles_to_remove, reverse=True):
        print(f"Removing circular mask: inner={circle_masks[idx][0]:.1f}, outer={circle_masks[idx][1]:.1f}")
        del circle_masks[idx]
    
    return len(circles_to_remove) > 0

def main():
    global fig, ax_fft, ax_fft_zoom, ax_phase, ax_original, ax_denoised
    global ax_fft_linear, ax_fft_zoom_linear, ax_phase_linear
    global fft_plot, fft_zoom_plot, phase_plot, original_plot, denoised_plot
    global fft_linear_plot, fft_zoom_linear_plot, phase_linear_plot
    global frame_slider, no_file_text, selector, current_mode
    global inner_radius_textbox, outer_radius_textbox
    global images, num_frames, tiff_path, output_filename
    
    # Initialize empty data if no file is provided
    images = np.zeros((1, 100, 100), dtype=np.float32)
    num_frames = 1
    tiff_path = "No file loaded"
    output_filename = "No file loaded"
    
    # These will be initialized when a file is loaded
    global fft_images, masked_fft, denoised_images, endpoints, midpoints
    fft_images = np.zeros_like(images, dtype=np.complex64)
    masked_fft = fft_images.copy()
    denoised_images = np.zeros_like(images)
    endpoints = np.asarray(images.shape[1:])
    midpoints = endpoints // 2
    
    # Check for command-line arguments
    initial_file = None
    if len(sys.argv) > 1:
        initial_file = sys.argv[1]
        if not os.path.exists(initial_file):
            print(f"Error: File {initial_file} does not exist")
            sys.exit(1)
    
    # Create the main figure - made taller for two rows
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Interactive FFT-based Noise Removal", fontsize=16)
    
    # Create a 2x5 grid of subplots
    grid_shape = (2, 5)
    
    # Top row (logarithmic FFT)
    ax_fft = plt.subplot2grid(grid_shape, (0, 0))
    ax_fft_zoom = plt.subplot2grid(grid_shape, (0, 1))
    ax_phase = plt.subplot2grid(grid_shape, (0, 2))
    ax_original = plt.subplot2grid(grid_shape, (0, 3))
    ax_denoised = plt.subplot2grid(grid_shape, (0, 4))
    
    # Bottom row (linear FFT + empty spaces)
    ax_fft_linear = plt.subplot2grid(grid_shape, (1, 0))
    ax_fft_zoom_linear = plt.subplot2grid(grid_shape, (1, 1))
    ax_phase_linear = plt.subplot2grid(grid_shape, (1, 2))
    # Leave (1, 3) and (1, 4) empty
    
    # Set up axes to fill their areas better
    for ax in [ax_fft, ax_phase, ax_original, ax_denoised, ax_fft_zoom, 
               ax_fft_linear, ax_fft_zoom_linear, ax_phase_linear]:
        ax.set_xticks([])
        ax.set_yticks([])
        # Use 'equal' for FFT plots to maintain proper scale
        # Use 'auto' for image plots to fill the subplot area
        if ax in [ax_original, ax_denoised]:
            ax.set_aspect('auto')
        else:
            ax.set_aspect('equal', adjustable='box')
    
    # Add some spacing between subplots
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Leave space at bottom for controls
    
    # Set titles for the top row (logarithmic)
    ax_fft.set_title("Log-Amplitude Spectrum\n(CLICK HERE TO MASK)", color='red')
    ax_fft_zoom.set_title("Zoomed Center Region\n(±20 pixels)")
    ax_phase.set_title("Phase Spectrum")
    ax_original.set_title("Original Image")
    ax_denoised.set_title("Denoised Image")
    
    # Set titles for the bottom row (linear)
    ax_fft_linear.set_title("Linear Amplitude Spectrum")
    ax_fft_zoom_linear.set_title("Linear Zoomed Center")
    ax_phase_linear.set_title("Phase Spectrum")
    
    # Create default data for plots
    spectrum_reference = np.zeros((100, 100))
    phase_reference = np.zeros((100, 100))
    fvmin, fvmax = 0, 1
    ivmin, ivmax = 0, 1
    
    # Create initial plots for top row (logarithmic)
    fft_plot = ax_fft.imshow(spectrum_reference, 
                           cmap=cc.m_CET_L17_r,
                           interpolation="bicubic",
                           vmin=fvmin,
                           vmax=fvmax)
    
    # Create zoomed FFT plot (logarithmic)
    zoom_size = 20
    
    # Create zoomed plot with same colormap as main FFT plot (logarithmic)
    fft_zoom_plot = ax_fft_zoom.imshow(np.zeros((2*zoom_size+1, 2*zoom_size+1)), 
                                      cmap=cc.m_CET_L17_r,
                                      interpolation="bicubic",
                                      vmin=fvmin,
                                      vmax=fvmax,
                                      extent=[-zoom_size, zoom_size, -zoom_size, zoom_size])
    
    # Set tick marks and labels for every 5 pixels, centered on (0,0)
    ticks = np.arange(-zoom_size, zoom_size+1, 5)
    ax_fft_zoom.set_xticks(ticks)
    ax_fft_zoom.set_yticks(ticks)
    ax_fft_zoom.set_xticklabels(ticks)
    ax_fft_zoom.set_yticklabels(ticks)
    
    # Draw a crosshair at the center of the zoomed plot (at 0,0)
    ax_fft_zoom.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax_fft_zoom.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Create bottom row plots (linear amplitude)
    fft_linear_plot = ax_fft_linear.imshow(spectrum_reference, 
                                         cmap=cc.m_CET_L17_r,
                                         interpolation="bicubic")
    
    # Create zoomed linear FFT plot
    fft_zoom_linear_plot = ax_fft_zoom_linear.imshow(np.zeros((2*zoom_size+1, 2*zoom_size+1)), 
                                                  cmap=cc.m_CET_L17_r,
                                                  interpolation="bicubic",
                                                  extent=[-zoom_size, zoom_size, -zoom_size, zoom_size])
    
    # Set tick marks and labels for every 5 pixels on the linear zoomed plot
    ax_fft_zoom_linear.set_xticks(ticks)
    ax_fft_zoom_linear.set_yticks(ticks)
    ax_fft_zoom_linear.set_xticklabels(ticks)
    ax_fft_zoom_linear.set_yticklabels(ticks)
    
    # Draw a crosshair at the center of the linear zoomed plot
    ax_fft_zoom_linear.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax_fft_zoom_linear.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Create phase plots for both rows
    phase_plot = ax_phase.imshow(phase_reference, 
                               cmap='hsv',
                               interpolation="bicubic",
                               vmin=-np.pi,
                               vmax=np.pi)
    
    phase_linear_plot = ax_phase_linear.imshow(phase_reference, 
                                           cmap='hsv',
                                           interpolation="bicubic",
                                           vmin=-np.pi,
                                           vmax=np.pi)
    
    # Create the original image plot
    original_plot = ax_original.imshow(images[current_frame], 
                                    cmap=cc.m_CET_L1_r,
                                    interpolation="bicubic",
                                    vmin=ivmin,
                                    vmax=ivmax)
    
    # Create the denoised image plot
    denoised_plot = ax_denoised.imshow(denoised_images[current_frame], 
                                    cmap=cc.m_CET_L1_r,
                                    interpolation="bicubic",
                                    vmin=ivmin,
                                    vmax=ivmax)
    
    # Add a text annotation for when no file is loaded
    no_file_text = ax_original.text(0.5, 0.5, "No file loaded\nUse 'Load TIFF' button", 
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  transform=ax_original.transAxes,
                                  fontsize=14,
                                  color='white')
    
    # Create controls area
    slider_height = 0.03
    button_height = 0.04
    button_width = 0.12
    button_spacing = 0.02
    textbox_width = 0.08
    bottom_margin = 0.05
    
    # Create frame slider
    ax_slider = plt.axes([0.25, bottom_margin, 0.5, slider_height])
    frame_slider = Slider(ax_slider, 'Frame', 0, max(num_frames-1, 1), valinit=0, valstep=1)
    
    # Create button axes - first row
    ax_load = plt.axes([0.05, bottom_margin + slider_height + button_spacing, button_width, button_height])
    ax_draw = plt.axes([0.25, bottom_margin + slider_height + button_spacing, button_width, button_height])
    ax_remove = plt.axes([0.25 + button_width + button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])
    ax_update = plt.axes([0.25 + 2*button_width + 2*button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])
    ax_save = plt.axes([0.25 + 3*button_width + 3*button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])
    
    # Create second row for circle mask controls
    row2_y = bottom_margin + slider_height + 2*button_spacing + button_height
    
    # Label for circle mask controls
    ax_circle_label = plt.axes([0.05, row2_y, textbox_width, button_height])
    ax_circle_label.text(0.5, 0.5, "Circle Mask:", horizontalalignment='center',
                        verticalalignment='center', transform=ax_circle_label.transAxes)
    ax_circle_label.set_xticks([])
    ax_circle_label.set_yticks([])
    
    # Labels and textboxes for inner radius
    ax_inner_label = plt.axes([0.05 + textbox_width + button_spacing, row2_y, textbox_width, button_height/2])
    ax_inner_label.text(0, 0.5, "Inner Radius:", horizontalalignment='left',
                       verticalalignment='center', transform=ax_inner_label.transAxes)
    ax_inner_label.set_xticks([])
    ax_inner_label.set_yticks([])
    
    ax_inner_textbox = plt.axes([0.05 + 2*textbox_width + button_spacing, row2_y, textbox_width, button_height/2])
    inner_radius_textbox = TextBox(ax_inner_textbox, '', initial='0')
    
    # Labels and textboxes for outer radius  
    ax_outer_label = plt.axes([0.05 + textbox_width + button_spacing, row2_y - button_height/2, textbox_width, button_height/2])
    ax_outer_label.text(0, 0.5, "Outer Radius:", horizontalalignment='left',
                       verticalalignment='center', transform=ax_outer_label.transAxes)
    ax_outer_label.set_xticks([])
    ax_outer_label.set_yticks([])
    
    ax_outer_textbox = plt.axes([0.05 + 2*textbox_width + button_spacing, row2_y - button_height/2, textbox_width, button_height/2])
    outer_radius_textbox = TextBox(ax_outer_textbox, '', initial='10')
    
    # Circle control buttons
    ax_add_circle = plt.axes([0.05 + 3*textbox_width + 2*button_spacing, row2_y, button_width, button_height/2])
    ax_clear_circles = plt.axes([0.05 + 3*textbox_width + 2*button_spacing, row2_y - button_height/2, button_width, button_height/2])
    
    # Create buttons
    btn_load = Button(ax_load, 'Load TIFF')
    btn_draw = Button(ax_draw, 'Draw Rect')
    btn_remove = Button(ax_remove, 'Remove Mask')
    btn_update = Button(ax_update, 'Update')
    btn_save = Button(ax_save, 'Save')
    
    # Create circle control buttons
    btn_add_circle = Button(ax_add_circle, 'Add Circle')
    btn_clear_circles = Button(ax_clear_circles, 'Clear Circles')
    
    # Connect callbacks
    frame_slider.on_changed(on_slider_change)
    btn_load.on_clicked(on_load_click)
    btn_draw.on_clicked(on_draw_click)
    btn_remove.on_clicked(on_remove_click)
    btn_update.on_clicked(on_update_click)
    btn_save.on_clicked(on_save_click)
    btn_add_circle.on_clicked(lambda event: add_circle_mask())
    btn_clear_circles.on_clicked(lambda event: clear_circle_masks())
    inner_radius_textbox.on_submit(on_inner_radius_change)
    outer_radius_textbox.on_submit(on_outer_radius_change)
    
    # Create the rectangle selector
    selector = RectangleSelector(ax_fft, on_select, 
                               interactive=True, button=[1, 3])
    selector.set_active(True)
    
    # Connect key press event for toggling the selector
    plt.connect('key_press_event', toggle_selector)
    
    # Set up a callback to save when the figure is closed
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Add colorbars to make scaling differences clear
    plt.colorbar(fft_plot, ax=ax_fft, label='Log Amplitude', fraction=0.046, pad=0.04)
    plt.colorbar(fft_zoom_plot, ax=ax_fft_zoom, label='Log Amplitude (Zoom)', fraction=0.046, pad=0.04)
    plt.colorbar(fft_linear_plot, ax=ax_fft_linear, label='Linear Amplitude', fraction=0.046, pad=0.04)
    plt.colorbar(fft_zoom_linear_plot, ax=ax_fft_zoom_linear, label='Linear Amplitude (Zoom)', fraction=0.046, pad=0.04)
    
    # If a file was provided as a command line argument, load it
    if initial_file:
        no_file_text.set_visible(False)
        if load_file(initial_file):
            # Update slider
            frame_slider.valmax = num_frames - 1
            frame_slider.ax.set_xlim(0, num_frames - 1)
            
            # Update plot ranges based on loaded data
            ivmin = np.percentile(images, 0)
            ivmax = np.percentile(images, 99)
            spectrum = np.log(np.abs(fft_images[0]) + 1e-6)
            fvmin = np.percentile(spectrum, 0)
            fvmax = np.percentile(spectrum, 99)
            
            # Update plot limits for all plots
            original_plot.set_clim(vmin=ivmin, vmax=ivmax)
            denoised_plot.set_clim(vmin=ivmin, vmax=ivmax)
            fft_plot.set_clim(vmin=fvmin, vmax=fvmax)
            fft_zoom_plot.set_clim(vmin=fvmin, vmax=fvmax)
            
            # For linear FFT plots, set appropriate limits based on data
            fft_amplitude_0 = np.abs(fft_images[0])
            dc_value = fft_amplitude_0[fft_amplitude_0.shape[0]//2, fft_amplitude_0.shape[1]//2]
            linear_vmax = dc_value * 0.05  # Show up to 5% of DC component
            
            fft_linear_plot.set_clim(vmin=0, vmax=linear_vmax)
            
            # For zoomed region, use a different scale
            center_y, center_x = fft_amplitude_0.shape[0]//2, fft_amplitude_0.shape[1]//2
            zoom_size = 20
            y_start = max(0, center_y - zoom_size)
            y_end = min(fft_amplitude_0.shape[0], center_y + zoom_size + 1)
            x_start = max(0, center_x - zoom_size)
            x_end = min(fft_amplitude_0.shape[1], center_x + zoom_size + 1)
            
            zoom_max = np.max(fft_amplitude_0[y_start:y_end, x_start:x_end])
            fft_zoom_linear_plot.set_clim(vmin=0, vmax=zoom_max * 0.5)  # 50% of max value in zoom region
            
            phase_plot.set_clim(vmin=-np.pi, vmax=np.pi)
            phase_linear_plot.set_clim(vmin=-np.pi, vmax=np.pi)
            
            # Update the plots
            update_plots(0)
    
    # Show the figure (non-blocking)
    plt.show(block=False)
    
    # Keep the figure alive
    plt.pause(0.1)
    input("Press Enter to close the figure and save the result...")
    plt.close(fig)

if __name__ == "__main__":
    main()
