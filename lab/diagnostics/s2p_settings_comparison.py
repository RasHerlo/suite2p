#!/usr/bin/env python3
"""
Suite2p Settings Comparison Tool

A GUI tool for comparing the contents of two suite2p output folders.
Provides side-by-side viewing of .npy files and other content with
difference highlighting.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Set, Any, Optional, Tuple


class Suite2pComparator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Suite2p Settings Comparison Tool")
        self.root.geometry("1400x800")
        
        # Store folder paths and common files
        self.folder1_path = None
        self.folder2_path = None
        self.common_files = {}  # relative_path -> (full_path1, full_path2)
        
        # GUI elements
        self.setup_gui()
        
        # Ask for folders on startup
        self.root.after(100, self.select_folders)
    
    def setup_gui(self):
        """Set up the main GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create three-panel layout
        # Left panel for file tree
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        # File tree
        ttk.Label(left_frame, text="Common Files", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_tree = ttk.Treeview(tree_frame)
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind file selection
        self.file_tree.bind('<<TreeviewSelect>>', self.on_file_select)
        
        # Right panel for content viewers
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create two content viewers
        self.viewer1_frame = ttk.LabelFrame(right_frame, text="Folder 1", padding=5)
        self.viewer1_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.viewer2_frame = ttk.LabelFrame(right_frame, text="Folder 2", padding=5)
        self.viewer2_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Content text areas
        self.content1 = scrolledtext.ScrolledText(self.viewer1_frame, wrap=tk.WORD, 
                                                  font=("Courier", 10))
        self.content1.pack(fill=tk.BOTH, expand=True)
        
        self.content2 = scrolledtext.ScrolledText(self.viewer2_frame, wrap=tk.WORD, 
                                                  font=("Courier", 10))
        self.content2.pack(fill=tk.BOTH, expand=True)
        
        # Configure text highlighting tags
        self.content1.tag_configure("diff", background="#ffcccc")
        self.content1.tag_configure("same", background="#ccffcc")
        self.content2.tag_configure("diff", background="#ffcccc")
        self.content2.tag_configure("same", background="#ccffcc")
        
        # Resize configuration
        left_frame.configure(width=300)
        
        # Menu bar
        self.create_menu()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Folders", command=self.select_folders)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def select_folders(self):
        """Open folder selection dialogs"""
        # Select first folder
        folder1 = filedialog.askdirectory(title="Select first suite2p folder")
        if not folder1:
            self.root.quit()
            return
        
        # Select second folder
        folder2 = filedialog.askdirectory(title="Select second suite2p folder")
        if not folder2:
            self.root.quit()
            return
        
        self.folder1_path = Path(folder1)
        self.folder2_path = Path(folder2)
        
        # Update viewer frame titles
        self.viewer1_frame.configure(text=f"Folder 1: {self.folder1_path.name}")
        self.viewer2_frame.configure(text=f"Folder 2: {self.folder2_path.name}")
        
        # Scan for common files
        self.scan_common_files()
        
        # Populate file tree
        self.populate_file_tree()
    
    def scan_common_files(self):
        """Scan both folders and find common files"""
        self.common_files = {}
        
        # Get all files from both folders
        files1 = self.get_all_files(self.folder1_path)
        files2 = self.get_all_files(self.folder2_path)
        
        # Find common files
        common_relative_paths = files1.keys() & files2.keys()
        
        for rel_path in common_relative_paths:
            self.common_files[rel_path] = (files1[rel_path], files2[rel_path])
        
        print(f"Found {len(self.common_files)} common files")
    
    def get_all_files(self, folder_path: Path) -> Dict[str, Path]:
        """Get all files in a folder with relative paths as keys"""
        files = {}
        
        if not folder_path.exists():
            return files
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(folder_path)
                files[str(rel_path)] = file_path
        
        return files
    
    def populate_file_tree(self):
        """Populate the file tree with common files"""
        # Clear existing tree
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Build tree structure
        tree_items = {}  # path -> item_id
        
        # Sort files to ensure proper hierarchy
        sorted_files = sorted(self.common_files.keys())
        
        for rel_path in sorted_files:
            parts = Path(rel_path).parts
            current_path = ""
            
            for i, part in enumerate(parts):
                if i == 0:
                    current_path = part
                else:
                    current_path = str(Path(current_path) / part)
                
                if current_path not in tree_items:
                    parent_path = str(Path(current_path).parent) if i > 0 else ""
                    parent_id = tree_items.get(parent_path, "")
                    
                    # Determine if this is a file or folder
                    is_file = (current_path == rel_path)
                    display_name = part
                    
                    if is_file:
                        # Add file extension info
                        ext = Path(part).suffix
                        if ext:
                            display_name = f"{part} ({ext})"
                    
                    item_id = self.file_tree.insert(parent_id, tk.END, text=display_name, 
                                                    values=(current_path, is_file))
                    tree_items[current_path] = item_id
        
        # Expand all folders
        for item_id in tree_items.values():
            self.file_tree.item(item_id, open=True)
    
    def on_file_select(self, event):
        """Handle file selection in tree"""
        selection = self.file_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.file_tree.item(item, 'values')
        
        if len(values) < 2:
            return
        
        rel_path, is_file = values[0], values[1] == 'True'
        
        if is_file and rel_path in self.common_files:
            self.load_file_content(rel_path)
    
    def load_file_content(self, rel_path: str):
        """Load and display content of selected file"""
        file1_path, file2_path = self.common_files[rel_path]
        
        try:
            content1 = self.load_file(file1_path)
            content2 = self.load_file(file2_path)
            
            self.display_content_with_diff(content1, content2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file {rel_path}:\n{str(e)}")
    
    def load_file(self, file_path: Path) -> str:
        """Load file content based on file type"""
        if file_path.suffix == '.npy':
            return self.load_npy_file(file_path)
        else:
            # Try to load as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
    
    def load_npy_file(self, file_path: Path) -> str:
        """Load and format .npy file content"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Handle different data types
            if isinstance(data, np.ndarray):
                if data.dtype == object:
                    # Might be a dictionary or other object
                    if data.shape == ():
                        # Single object (often a dictionary)
                        obj = data.item()
                        return self.format_object(obj)
                    else:
                        # Array of objects
                        return self.format_array(data)
                else:
                    # Regular numpy array
                    return self.format_array(data)
            else:
                # Other types
                return self.format_object(data)
        
        except Exception as e:
            return f"Error loading .npy file: {str(e)}"
    
    def format_object(self, obj: Any) -> str:
        """Format Python object for display"""
        if isinstance(obj, dict):
            return self.format_dict(obj)
        elif isinstance(obj, (list, tuple)):
            return self.format_list(obj)
        else:
            return str(obj)
    
    def format_dict(self, d: dict, indent: int = 0) -> str:
        """Format dictionary for display"""
        lines = []
        indent_str = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(self.format_dict(value, indent + 1))
            elif isinstance(value, (list, tuple, np.ndarray)):
                lines.append(f"{indent_str}{key}: {self.format_array_compact(value)}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        
        return "\n".join(lines)
    
    def format_list(self, lst: list) -> str:
        """Format list for display"""
        if len(lst) <= 10:
            return str(lst)
        else:
            return f"[{lst[0]}, {lst[1]}, ..., {lst[-1]}] (length: {len(lst)})"
    
    def format_array(self, arr: np.ndarray) -> str:
        """Format numpy array for display"""
        if arr.size <= 100:
            return str(arr)
        else:
            return f"Array shape: {arr.shape}, dtype: {arr.dtype}\nFirst few values:\n{arr.flat[:10]}"
    
    def format_array_compact(self, arr) -> str:
        """Format array in compact form"""
        if isinstance(arr, np.ndarray):
            if arr.size <= 5:
                return str(arr.tolist())
            else:
                return f"Array({arr.shape}, dtype={arr.dtype})"
        elif isinstance(arr, (list, tuple)):
            if len(arr) <= 5:
                return str(arr)
            else:
                return f"List(length={len(arr)})"
        else:
            return str(arr)
    
    def display_content_with_diff(self, content1: str, content2: str):
        """Display content with difference highlighting"""
        # Clear previous content
        self.content1.delete(1.0, tk.END)
        self.content2.delete(1.0, tk.END)
        
        # Split content into lines
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        # Pad shorter content with empty lines
        max_lines = max(len(lines1), len(lines2))
        lines1.extend([''] * (max_lines - len(lines1)))
        lines2.extend([''] * (max_lines - len(lines2)))
        
        # Display with highlighting
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            line_num = i + 1
            
            # Add line to both text widgets
            self.content1.insert(tk.END, line1 + '\n')
            self.content2.insert(tk.END, line2 + '\n')
            
            # Apply highlighting
            if line1 != line2:
                # Different lines - highlight in red
                self.content1.tag_add("diff", f"{line_num}.0", f"{line_num}.end")
                self.content2.tag_add("diff", f"{line_num}.0", f"{line_num}.end")
            else:
                # Same lines - highlight in green
                self.content1.tag_add("same", f"{line_num}.0", f"{line_num}.end")
                self.content2.tag_add("same", f"{line_num}.0", f"{line_num}.end")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        return
    
    app = Suite2pComparator()
    app.run()


if __name__ == "__main__":
    main() 