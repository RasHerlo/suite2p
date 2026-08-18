import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_and_save_images(ops_path, output_dir=None):
    # Load the ops file
    ops = np.load(ops_path, allow_pickle=True).item()
    
    # If no output directory specified, use the same directory as ops file
    if output_dir is None:
        output_dir = Path(ops_path).parent
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to normalize image for visualization
    def normalize_image(img):
        return (img - img.min()) / (img.max() - img.min())
    
    # Save mean image
    if 'meanImg' in ops:
        plt.figure(figsize=(10, 10))
        plt.imshow(normalize_image(ops['meanImg']), cmap='viridis')
        plt.colorbar()
        plt.title('Mean Image')
        plt.savefig(output_dir / 'mean_image.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mean image to {output_dir / 'mean_image.png'}")
    
    # Save reference image
    if 'refImg' in ops:
        plt.figure(figsize=(10, 10))
        plt.imshow(normalize_image(ops['refImg']), cmap='viridis')
        plt.colorbar()
        plt.title('Reference Image')
        plt.savefig(output_dir / 'reference_image.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved reference image to {output_dir / 'reference_image.png'}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ops_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        visualize_and_save_images(ops_path, output_dir)
    else:
        print("Please provide the path to the ops.npy file as an argument") 