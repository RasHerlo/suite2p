#!/usr/bin/env python3
"""
Standalone Traces Pickle Generation Script

Generate pickle file with selected traces from suite2p output.

Usage:
    python generate_traces_pickle.py /path/to/suite2p/plane0 /path/to/output/dir
    python generate_traces_pickle.py /path/to/suite2p/plane0 /path/to/output/dir --filename custom_traces.pkl
"""

import sys
import argparse
import numpy as np
import pickle
from pathlib import Path

def generate_pickle_file(suite2p_dir, output_dir, filename='selected_traces.pkl'):
    """
    Generate pickle file with selected traces data.
    
    Parameters:
    -----------
    suite2p_dir : str
        Path to suite2p output directory (plane0 folder)
    output_dir : str
        Directory to save pickle file
    filename : str
        Name of the pickle file to create
    """
    suite2p_dir = Path(suite2p_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required files
    required_files = ['F.npy', 'iscell.npy']
    missing_files = []
    
    for file in required_files:
        if not (suite2p_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files in {suite2p_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Load data
    print(f"Loading data from: {suite2p_dir}")
    F = np.load(suite2p_dir / 'F.npy')
    iscell = np.load(suite2p_dir / 'iscell.npy')
    
    # Get selected cells
    selected_mask = iscell[:, 0] == 1
    selected_cells = F[selected_mask]
    selected_indices = np.where(selected_mask)[0]
    
    print(f"Total ROIs: {len(F)}")
    print(f"Selected cells: {len(selected_cells)}")
    print(f"Trace length: {F.shape[1]} timepoints")
    
    if len(selected_cells) == 0:
        print("ERROR: No cells selected! Check iscell.npy file.")
        return False
    
    # Create and save pickle file with data
    data_dict = {
        'roi_indices': selected_indices,
        'traces': selected_cells,
        'n_total_rois': len(F),
        'n_selected_rois': len(selected_cells),
        'n_timepoints': F.shape[1]
    }
    
    pickle_path = output_dir / filename
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Pickle file saved to: {pickle_path}")
    
    # Print summary
    print("\nPickle file contents:")
    print(f"  - roi_indices: indices of selected ROIs ({len(selected_indices)})")
    print(f"  - traces: fluorescence traces ({selected_cells.shape})")
    print(f"  - n_total_rois: total number of ROIs ({len(F)})")
    print(f"  - n_selected_rois: number of selected ROIs ({len(selected_cells)})")
    print(f"  - n_timepoints: number of timepoints ({F.shape[1]})")
    
    return True

def load_and_inspect_pickle(pickle_path):
    """Load and inspect the contents of a pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded pickle file: {pickle_path}")
        print(f"Keys: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: array shape {value.shape}, dtype {value.dtype}")
                if value.size <= 10:
                    print(f"    values: {value}")
                else:
                    print(f"    sample values: {value.flat[:5]}...")
            else:
                print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate pickle file with selected traces')
    parser.add_argument('suite2p_dir', help='Path to suite2p output directory (plane0 folder)')
    parser.add_argument('output_dir', help='Directory to save pickle file')
    parser.add_argument('--filename', default='selected_traces.pkl',
                       help='Name of the pickle file (default: selected_traces.pkl)')
    parser.add_argument('--inspect', help='Path to existing pickle file to inspect')
    
    args = parser.parse_args()
    
    if args.inspect:
        # Just inspect an existing pickle file
        success = load_and_inspect_pickle(args.inspect)
        return 0 if success else 1
    
    success = generate_pickle_file(
        args.suite2p_dir, 
        args.output_dir,
        args.filename
    )
    
    if success:
        print("Pickle file generation completed successfully!")
        return 0
    else:
        print("Pickle file generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 