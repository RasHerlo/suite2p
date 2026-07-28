import numpy as np
from tifffile import imwrite
from pathlib import Path

def extract_registered_stack(ops_path, output_filename='registered_stack.tif'):
    """
    Extract the motion-corrected stack from Suite2p binary file.
    
    Parameters:
    -----------
    ops_path : str or Path
        Path to ops.npy file
    output_filename : str
        Name for output TIFF file
    """
    ops_path = Path(ops_path)
    
    # Load ops file
    print(f"Loading ops from: {ops_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    
    # Get the binary file path
    plane_dir = ops_path.parent
    bin_file = plane_dir / 'data.bin'
    
    if not bin_file.exists():
        print(f"ERROR: Binary file not found at {bin_file}")
        return None
    
    print(f"Found binary file: {bin_file}")
    print(f"  Dimensions: {ops['Ly']} x {ops['Lx']} pixels")
    print(f"  Number of frames: {ops['nframes']}")
    
    # Read binary file directly
    print("Reading registered frames from binary file...")
    Ly = ops['Ly']
    Lx = ops['Lx']
    nframes = ops['nframes']
    
    # Read the binary file as int16 (Suite2p default format)
    with open(bin_file, 'rb') as f:
        registered_stack = np.fromfile(f, dtype='int16')
    
    # Reshape to (nframes, Ly, Lx)
    registered_stack = registered_stack.reshape((nframes, Ly, Lx))
    
    # Save as TIFF
    output_path = plane_dir / output_filename
    print(f"Saving registered stack to: {output_path}")
    print(f"  Stack shape: {registered_stack.shape}")
    print(f"  Stack dtype: {registered_stack.dtype}")
    print(f"  Value range: [{registered_stack.min():.2f}, {registered_stack.max():.2f}]")
    
    # Save as int16 to preserve data and save space
    imwrite(str(output_path), registered_stack.astype('int16'))
    
    print(f"\n✓ Successfully saved registered stack!")
    print(f"  Output: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_registered_stack.py <path_to_ops.npy> [output_filename]")
        print("\nExample:")
        print('  python extract_registered_stack.py "F:\\path\\to\\suite2p\\plane0\\ops.npy"')
        print('  python extract_registered_stack.py "F:\\path\\to\\suite2p\\plane0\\ops.npy" my_registered.tif')
        sys.exit(1)
    
    ops_path = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else 'registered_stack.tif'
    
    extract_registered_stack(ops_path, output_filename)

