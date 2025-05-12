from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import matplotlib
matplotlib.use("QtAgg")
matplotlib.interactive(True)
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from matplotlib.patches import Rectangle
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
endpoints = None
midpoints = None

# Function to initialize a file
def initialize_file(file_path):
    global images, fft_images, masked_fft, denoised_images, num_frames
    global tiff_path, output_filename, mask_regions, endpoints, midpoints
    
    # Store file path and set output filename
    tiff_path = file_path
    output_filename = os.path.splitext(tiff_path)[0] + "_denoised.tif"
    print(f"Processing file: {tiff_path}")
    print(f"Denoised images will be saved to: {output_filename}")
    
    # Clear existing masks
    mask_regions = []
    
    # Load images
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
    
    # Initial mask regions - these are the ones from the original script
    box_offset = (15, 150)  # y, x
    box_size = [10, 200]    # y, x
    mask_regions.append([
        midpoints[0]+box_offset[0],
        midpoints[0]+box_offset[0]+box_size[0],
        midpoints[1]-box_offset[1],
        midpoints[1]-box_offset[1]+box_size[1]
    ])
    mask_regions.append([
        midpoints[0]-box_offset[0]-box_size[0],
        midpoints[0]-box_offset[0],
        midpoints[1]+box_offset[1]-box_size[1],
        midpoints[1]+box_offset[1]
    ])
    
    # Apply initial masks
    for region in mask_regions:
        y1, y2, x1, x2 = region
        masked_fft[:, y1:y2, x1:x2] = 0
    
    # Prepare initial denoised images
    reconstructed_images = np.fft.ifftshift(masked_fft)
    reconstructed_images = np.fft.ifft2(reconstructed_images)
    denoised_images = np.abs(reconstructed_images)
    
    # Reset plot extents to match new image dimensions
    h, w = images.shape[1], images.shape[2]
    original_plot.set_extent([0, w, h, 0])
    denoised_plot.set_extent([0, w, h, 0])
    fft_plot.set_extent([0, w, h, 0])
    phase_plot.set_extent([0, w, h, 0])
    
    # Ensure axes adapt to new image size
    ax_original.autoscale()
    ax_denoised.autoscale()
    ax_fft.autoscale()
    ax_phase.autoscale()
    
    return True

# Check for command-line arguments
if len(sys.argv) > 1:
    # Use the provided path
    initial_file = sys.argv[1]
    if not os.path.exists(initial_file):
        print(f"Error: File {initial_file} does not exist")
        sys.exit(1)
    initialize_file(initial_file)
else:
    # Empty initialization to start
    images = np.zeros((1, 100, 100), dtype=np.float32)
    num_frames = 1
    tiff_path = "No file loaded"
    output_filename = "No file loaded"
    
    # These will be initialized when a file is loaded
    fft_images = np.zeros_like(images, dtype=np.complex64)
    masked_fft = fft_images.copy()
    denoised_images = np.zeros_like(images)
    endpoints = np.asarray(images.shape[1:])
    midpoints = endpoints // 2

# Create the main figure
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Interactive FFT-based Noise Removal", fontsize=16)

# Create subplots with more spacing and better layout
ax_fft = fig.add_subplot(141)
ax_phase = fig.add_subplot(142)
ax_original = fig.add_subplot(143)
ax_denoised = fig.add_subplot(144)

# Set up axes
for ax in [ax_fft, ax_phase, ax_original, ax_denoised]:
    ax.set_xticks([])
    ax.set_yticks([])
    # Use 'auto' instead of 'equal' to better fill available space
    ax.set_aspect('auto')

# Add some spacing between subplots
plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Leave space at bottom for controls

# Set titles
ax_fft.set_title("Log-Amplitude Spectrum\n(Draw/Remove Masks Here)")
ax_phase.set_title("Phase Spectrum")
ax_original.set_title("Original Image")
ax_denoised.set_title("Denoised Image")

# Calculate initial spectrum and vmin/vmax values
if images is not None and images.size > 0:
    spectrum_reference = np.log(np.abs(fft_images[current_frame]) + 1e-6)
    phase_reference = np.angle(fft_images[current_frame])
    fvmin = np.percentile(spectrum_reference, 0)
    fvmax = np.percentile(spectrum_reference, 99)
    
    ivmin = np.percentile(images, 0)
    ivmax = np.percentile(images, 99)
else:
    # Default values if no image is loaded
    spectrum_reference = np.zeros((100, 100))
    phase_reference = np.zeros((100, 100))
    fvmin, fvmax = 0, 1
    ivmin, ivmax = 0, 1

# Create initial plots
fft_plot = ax_fft.imshow(spectrum_reference, 
                        cmap=cc.m_CET_L17_r,
                        interpolation="bicubic",
                        vmin=fvmin,
                        vmax=fvmax)

phase_plot = ax_phase.imshow(phase_reference, 
                           cmap='hsv',
                           interpolation="bicubic",
                           vmin=-np.pi,
                           vmax=np.pi)

original_plot = ax_original.imshow(images[current_frame], 
                                cmap=cc.m_CET_L1_r,
                                interpolation="bicubic",
                                vmin=ivmin,
                                vmax=ivmax)

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

# Hide the text when a file is loaded
if len(sys.argv) > 1:
    no_file_text.set_visible(False)

# Create a slider axis and the slider
slider_height = 0.03
button_height = 0.04
button_width = 0.12
button_spacing = 0.02
bottom_margin = 0.05

ax_slider = plt.axes([0.25, bottom_margin, 0.5, slider_height])
frame_slider = Slider(ax_slider, 'Frame', 0, max(num_frames-1, 1), valinit=0, valstep=1)

# Create button axes and buttons
ax_load = plt.axes([0.05, bottom_margin + slider_height + button_spacing, button_width, button_height])
ax_draw = plt.axes([0.25, bottom_margin + slider_height + button_spacing, button_width, button_height])
ax_remove = plt.axes([0.25 + button_width + button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])
ax_update = plt.axes([0.25 + 2*button_width + 2*button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])
ax_save = plt.axes([0.25 + 3*button_width + 3*button_spacing, bottom_margin + slider_height + button_spacing, button_width, button_height])

btn_load = Button(ax_load, 'Load TIFF')
btn_draw = Button(ax_draw, 'Draw Mask')
btn_remove = Button(ax_remove, 'Remove Mask')
btn_update = Button(ax_update, 'Update')
btn_save = Button(ax_save, 'Save')

# Variables to track the current mode and rectangle selector
current_mode = 'draw'  # 'draw' or 'remove'
selector = None
temp_regions = []
mask_patches = []  # To store visualization rectangles

# Function to update the plots based on the current frame
def update_plots(frame):
    if images is None or images.size == 0 or frame >= images.shape[0]:
        return
    
    # Update original and denoised images
    original_plot.set_data(images[frame])
    denoised_plot.set_data(denoised_images[frame])
    
    # Update spectrum image
    spectrum = np.log(np.abs(fft_images[frame]) + 1e-6)
    fft_plot.set_data(spectrum)
    
    # Update phase image
    phase_data = np.angle(fft_images[frame])
    phase_plot.set_data(phase_data)
    
    # Reset extents for all plots to ensure proper display
    h, w = images.shape[1], images.shape[2]
    original_plot.set_extent([0, w, h, 0])
    denoised_plot.set_extent([0, w, h, 0])
    fft_plot.set_extent([0, w, h, 0])
    phase_plot.set_extent([0, w, h, 0])
    
    # Remove any explicit limits to let the image fill the available space
    for ax in [ax_fft, ax_phase, ax_original, ax_denoised]:
        ax.autoscale()
    
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
    
    # Apply all masks
    for region in mask_regions:
        y1, y2, x1, x2 = region
        masked_fft[:, y1:y2, x1:x2] = 0
    
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

# Function to load a new TIFF file
def on_load_click(event):
    global no_file_text
    
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
        success = initialize_file(file_path)
        
        if success:
            # Hide the "no file loaded" text
            no_file_text.set_visible(False)
            
            # Reset slider
            frame_slider.valmax = num_frames - 1
            frame_slider.ax.set_xlim(0, num_frames - 1)
            frame_slider.set_val(0)
            
            # Update plots with the new data
            # Recalculate display ranges for intensity scaling
            global ivmin, ivmax, fvmin, fvmax
            ivmin = np.percentile(images, 0)
            ivmax = np.percentile(images, 99)
            spectrum = np.log(np.abs(fft_images[0]) + 1e-6)
            fvmin = np.percentile(spectrum, 0)
            fvmax = np.percentile(spectrum, 99)
            
            # Update the data limits for the plots
            original_plot.set_clim(vmin=ivmin, vmax=ivmax)
            denoised_plot.set_clim(vmin=ivmin, vmax=ivmax)
            fft_plot.set_clim(vmin=fvmin, vmax=fvmax)
            phase_plot.set_clim(vmin=-np.pi, vmax=np.pi)
            
            # Set the data for all plots
            update_plots(0)
            
            # Force redraw
            plt.draw()
            fig.canvas.flush_events()
            
            # Clear any mask visualizations
            clear_mask_visualizations()
            
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
    clear_mask_visualizations()
    print("Draw mode activated")

def on_remove_click(event):
    global current_mode, selector
    current_mode = 'remove'
    if selector:
        selector.set_active(True)
    # Show mask visualizations when switching to remove mode
    show_mask_visualizations()
    print("Remove mode activated")

def on_update_click(event):
    # Clear visualizations before updating
    clear_mask_visualizations()
    apply_masks()
    # If in remove mode, redraw the visualizations after update
    if current_mode == 'remove':
        show_mask_visualizations()
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

# Connect callbacks
frame_slider.on_changed(on_slider_change)
btn_load.on_clicked(on_load_click)
btn_draw.on_clicked(on_draw_click)
btn_remove.on_clicked(on_remove_click)
btn_update.on_clicked(on_update_click)
btn_save.on_clicked(on_save_click)

# Create the rectangle selector
selector = RectangleSelector(ax_fft, on_select, 
                           interactive=True, button=[1, 3])
selector.set_active(True)

# Connect key press event for toggling the selector
plt.connect('key_press_event', toggle_selector)

# Set up a callback to save when the figure is closed
def on_close(event):
    if tiff_path != "No file loaded" and denoised_images is not None:
        print("Figure closed. Saving images...")
        # Clear any visualizations before saving
        clear_mask_visualizations()
        save_images()

fig.canvas.mpl_connect('close_event', on_close)

# Show the figure (non-blocking)
plt.show(block=False)

# Keep the figure alive
plt.pause(0.1)
input("Press Enter to close the figure and save the result...")
plt.close(fig)
