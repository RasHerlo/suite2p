#!/usr/bin/env python3
"""
Suite2p Functions Runner

Run individual functions from the data processing pipeline.

Usage:
    python run_s2p_functions.py --list
    python run_s2p_functions.py roi-selection /path/to/suite2p/plane0 /path/to/output
    python run_s2p_functions.py rasterplots /path/to/suite2p/plane0 /path/to/output
    python run_s2p_functions.py pickle /path/to/suite2p/plane0 /path/to/output
    python run_s2p_functions.py --help <function_name>
"""

import sys
import argparse
import subprocess
from pathlib import Path

def list_functions():
    """List available functions and their descriptions."""
    functions = {
        'roi-selection': {
            'description': 'Apply ROI selection based on ellipticity and components',
            'script': 'run_roi_selection.py',
            'example': 'python run_s2p_functions.py roi-selection /path/to/suite2p/plane0 /path/to/output --ellipticity 0.8 --components 2'
        },
        'rasterplots': {
            'description': 'Generate rasterplots (default and rastermap-sorted)',
            'script': 'run_rasterplots.py',
            'example': 'python run_s2p_functions.py rasterplots /path/to/suite2p/plane0 /path/to/output --no-rastermap'
        },
        'pickle': {
            'description': 'Generate pickle file with selected traces',
            'script': 'generate_traces_pickle.py',
            'example': 'python run_s2p_functions.py pickle /path/to/suite2p/plane0 /path/to/output --filename my_traces.pkl'
        }
    }
    
    print("Available functions:")
    print("===================")
    for name, info in functions.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Script: {info['script']}")
        print(f"  Example: {info['example']}")
    
    return functions

def run_function(function_name, args):
    """Run a specific function with the given arguments."""
    functions = {
        'roi-selection': 'run_roi_selection.py',
        'rasterplots': 'run_rasterplots.py',
        'pickle': 'generate_traces_pickle.py'
    }
    
    if function_name not in functions:
        print(f"ERROR: Unknown function '{function_name}'")
        print(f"Available functions: {list(functions.keys())}")
        return False
    
    script_name = functions[function_name]
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script {script_name} not found in {script_path.parent}")
        return False
    
    # Run the script with the provided arguments
    cmd = [sys.executable, str(script_path)] + args
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Function failed with return code {e.returncode}")
        return False

def get_function_help(function_name):
    """Get help for a specific function."""
    functions = {
        'roi-selection': 'run_roi_selection.py',
        'rasterplots': 'run_rasterplots.py',
        'pickle': 'generate_traces_pickle.py'
    }
    
    if function_name not in functions:
        print(f"ERROR: Unknown function '{function_name}'")
        return False
    
    script_name = functions[function_name]
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script {script_name} not found")
        return False
    
    # Run the script with --help
    cmd = [sys.executable, str(script_path), '--help']
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    if len(sys.argv) == 1:
        print(__doc__)
        return 1
    
    # Handle --list
    if '--list' in sys.argv:
        list_functions()
        return 0
    
    # Handle --help <function>
    if '--help' in sys.argv and len(sys.argv) >= 3:
        function_name = sys.argv[2]
        success = get_function_help(function_name)
        return 0 if success else 1
    
    # Parse function name
    if len(sys.argv) < 2:
        print("ERROR: No function specified")
        print(__doc__)
        return 1
    
    function_name = sys.argv[1]
    function_args = sys.argv[2:]
    
    # Special handling for help requests
    if function_name == '--help' or function_name == '-h':
        print(__doc__)
        print()
        list_functions()
        return 0
    
    success = run_function(function_name, function_args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 