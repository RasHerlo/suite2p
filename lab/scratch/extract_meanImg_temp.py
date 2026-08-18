import numpy as np
from tifffile import imwrite
from pathlib import Path

# Path to ops.npy file
ops_path = Path(r"F:\Rasmus_CTN_bPAC_2309xx\EMx1_Dbl_Labelling\230929\take1_Chans\ChanA\SUPPORT\20240321_122448\denoised_cut\suite2p\plane0\ops.npy")

# Load ops file
print(f"Loading ops file from: {ops_path}")
ops = np.load(ops_path, allow_pickle=True).item()

# Get meanImg
meanImg = ops['meanImg']

# Save in same directory as ops.npy
output_path = ops_path.parent / 'meanImg.tif'
imwrite(str(output_path), meanImg.astype('float32'))

print(f"\nSuccessfully extracted meanImg")
print(f"  Saved to: {output_path}")
print(f"  Image shape: {meanImg.shape}")
print(f"  Image range: [{meanImg.min():.2f}, {meanImg.max():.2f}]")

