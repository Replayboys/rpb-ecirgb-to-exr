#!/usr/bin/env python3

import OpenImageIO as oiio
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def eci_to_srgb_matrix():
    """
    Returns the transformation matrix from ECI RGB v2 to linear sRGB.
    Using the Nuke Bradford matrix for direct conversion.
    """
    # Direct ECI RGB v2 to sRGB matrix from Nuke
    return np.array([
        [ 1.46066,   -0.384547,  -0.0761167],
        [-0.0266179,  0.965383,   0.0612355],
        [-0.0263729, -0.041449,   1.06782  ]
    ])

def linearize_lstar(img):
    """
    Linearize the L* encoding of ECI RGB v2.
    This converts from perceptual L* space to linear RGB.
    
    Args:
        img (numpy.ndarray): Input image array in L* space
    
    Returns:
        numpy.ndarray: Linearized image
    """
    # Ensure we're working with float values
    img = img.astype(np.float32)
    
    # Print debug info about input values
    print(f"Input image range: min={np.min(img)}, max={np.max(img)}")
    
    # Scale input values to 0-100 range if they're not already
    if np.max(img) <= 1.0:
        img = img * 100.0
    
    # L* to linear conversion
    # For values below 8, use a linear segment
    mask = img < 8
    linearized = np.zeros_like(img)
    
    # Linear segment (for dark values)
    linearized[mask] = img[mask] / 903.3
    
    # L* segment (for brighter values)
    linearized[~mask] = np.power((img[~mask] + 16) / 116, 3)
    
    # Print debug info about output values
    print(f"Linearized image range: min={np.min(linearized)}, max={np.max(linearized)}")
    
    return linearized

def convert_image(input_path, output_path):
    """
    Convert an ECI RGB v2 TIFF image to linear sRGB.
    
    Args:
        input_path (str): Path to input ECI RGB v2 TIFF file
        output_path (str): Path to output linear sRGB file
    """
    try:
        # Read input image
        inp = oiio.ImageInput.open(input_path)
        if not inp:
            print(f"Error: Could not open {input_path}")
            return False
        
        spec = inp.spec()
        print(f"Input image spec: width={spec.width}, height={spec.height}, channels={spec.nchannels}")
        print(f"Input image format: {spec.format}")
        
        img = inp.read_image()
        inp.close()
        
        # Convert to float32 for processing
        img = img.astype(np.float32)
        
        # Linearize the L* encoding (ECI RGB v2 uses L* instead of gamma)
        img = linearize_lstar(img)
        
        # Get transformation matrix
        transform_matrix = eci_to_srgb_matrix()
        
        # Apply color transformation
        # Reshape image for matrix multiplication
        pixels = img.reshape(-1, 3)
        transformed = np.dot(pixels, transform_matrix.T)
        
        # Reshape back to image dimensions
        transformed = transformed.reshape(img.shape)
        
        # Print debug info about transformed values
        print(f"Transformed image range: min={np.min(transformed)}, max={np.max(transformed)}")
        
        # Create output image
        out = oiio.ImageOutput.create(output_path)
        if not out:
            print(f"Error: Could not create {output_path}")
            return False
        
        # Set output specification
        out_spec = oiio.ImageSpec(spec.width, spec.height, 3, oiio.FLOAT)
        out_spec.attribute("oiio:ColorSpace", "Linear")
        
        # Set EXR compression to DWAA
        if output_path.lower().endswith('.exr'):
            out_spec.attribute("compression", "dwaa")
            out_spec.attribute("dwaCompressionLevel", 45)  # Default quality level
        
        # Write the image
        out.open(output_path, out_spec)
        out.write_image(transformed)
        out.close()
        
        print(f"Successfully converted {input_path} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def process_directory(input_dir, overwrite=False):
    """
    Process all TIFF files in the input directory.
    
    Args:
        input_dir (str): Path to directory containing TIFF files
        overwrite (bool): Whether to overwrite existing EXR files
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        return False
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process all TIFF files in the directory
    for tiff_file in input_path.glob("*.tif"):
        print(f"Found TIFF file: {tiff_file}")
        # Create output filename
        output_file = tiff_file.with_name(f"{tiff_file.stem}_linear-rec709.exr")
        
        # Skip if output exists and overwrite is False
        if output_file.exists() and not overwrite:
            print(f"Skipping {tiff_file.name} - output already exists")
            skipped_count += 1
            continue
        
        print(f"Processing {tiff_file.name}...")
        if convert_image(str(tiff_file), str(output_file)):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully converted: {success_count} files")
    print(f"Skipped (already exist): {skipped_count} files")
    print(f"Failed conversions: {error_count} files")
    
    return error_count == 0

class TIFFHandler(FileSystemEventHandler):
    def __init__(self, overwrite=False):
        self.overwrite = overwrite
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.tif'):
            print(f"\nNew TIFF file detected: {event.src_path}")
            output_file = Path(event.src_path).with_name(f"{Path(event.src_path).stem}_linear-rec709.exr")
            convert_image(event.src_path, str(output_file))

def watch_folder(input_dir, overwrite=False):
    """
    Watch a folder for new TIFF files and convert them automatically.
    Also processes any existing TIFF files in the folder.
    
    Args:
        input_dir (str): Path to directory to watch
        overwrite (bool): Whether to overwrite existing EXR files
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        return False

    print(f"\nProcessing existing TIFF files in: {input_dir}")
    # First process any existing TIFF files
    process_directory(input_dir, overwrite)
    
    # Set up the file system observer
    event_handler = TIFFHandler(overwrite)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()
    
    print(f"\nWatching folder: {input_dir}")
    print("Press Ctrl+C to stop watching...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping watch folder...")
    
    observer.join()

def main():
    parser = argparse.ArgumentParser(description='Convert ECI RGB v2 TIFF files to linear sRGB EXR')
    parser.add_argument('input_dir', help='Input directory containing TIFF files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing EXR files')
    parser.add_argument('--watch', action='store_true', help='Watch folder for new TIFF files')
    
    args = parser.parse_args()
    
    if args.watch:
        watch_folder(args.input_dir, args.overwrite)
    else:
        if not process_directory(args.input_dir, args.overwrite):
            sys.exit(1)

if __name__ == "__main__":
    main() 