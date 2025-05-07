# ECI RGB to ACEScg Converter

This tool converts ECI RGB TIFF files to ACEScg color space using OpenImageIO (OIIO). The conversion is performed using a direct matrix transformation without requiring an external OCIO configuration.

## Requirements

- Python 3.6 or higher
- OpenImageIO
- NumPy

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic conversion:
```bash
python eci_to_acescg.py <input_directory>
```

Watch folder mode (processes existing files and watches for new ones):
```bash
python eci_to_acescg.py <input_directory> --watch
```

Additional options:
```bash
python eci_to_acescg.py <input_directory> --overwrite  # Overwrite existing EXR files
python eci_to_acescg.py <input_directory> --watch --overwrite  # Watch folder and overwrite existing files
```

Example:
```bash
python eci_to_acescg.py ./input_folder --watch
```

## Notes

- The input files must be TIFF files in ECI RGB color space
- The output will be EXR files in linear sRGB color space
- The conversion uses a direct matrix transformation from ECI RGB to linear sRGB
- No external OCIO configuration is required
- In watch folder mode, the script will process all existing TIFF files and automatically convert any new TIFF files added to the folder

## Technical Details

The conversion is performed using a two-step matrix transformation:
1. ECI RGB to XYZ
2. XYZ to ACEScg

The matrices are combined into a single transformation for efficiency. The output preserves the full dynamic range and out-of-gamut colors in the EXR format. 