# Install instructions for symmetrical-dollop

conda create -n py2DTM python=3.10
conda activate py2DTM
conda install numpy scipy matplotlib scikit-learn 
conda install ipykernel scikit-image
pip install mrcfile


# Documentation for 

05-Parallel_notebook.ipynb

## Overview
This Jupyter Notebook demonstrates a parallel pipeline for processing 3D volumes and performing cross-correlation with 2D images "so called Template Matching".
The notebook includes functions for reading MRC files, generating angles using Healpix sampling, applying band-pass filters, rotating and projecting volumes, and performing cross-correlation using both CPU and GPU acceleration.

## Dependencies
The notebook requires the following libraries:
- `numpy`
- `mrcfile`
- `healpy`
- `matplotlib`
- `scipy`
- `skimage`
- `multiprocessing`
- `joblib`
- `cupy` (for GPU acceleration)

## Sections

### 1. Import Libraries
The notebook starts by importing necessary libraries for numerical operations, file handling, plotting, signal processing, and parallel computing.

### 2. Define Functions
Several functions are defined to handle various tasks:
- `read_mrc_file(file_path)`: Reads a 3D volume from an MRC file.
- `generate_healpix_pixel_bounds(nside)`: Generates boundaries of theta and phi for each Healpix pixel.
- `generate_angles_in_pixel(theta_min, theta_max, phi_min, phi_max, angular_spacing_rad)`: Generates theta and phi angles within specified bounds.
- `generate_all_angles(nside, angular_spacing_deg)`: Generates all theta and phi angles using Healpix sampling.
- `bandpass_filter(img, low_sigma, high_sigma)`: Applies a band-pass filter to an image.
- `rotate_and_project_opt(volume, phi, theta, psi)`: Efficiently rotates a 3D volume and projects it along the Z-axis.
- `cross_correlate_fourier_gpu(image, template)`: Performs cross-correlation using GPU acceleration.
- `process_combination(volume, phi, theta, psi, I_filtered, projection_shape, ncc_max, ncc_mean, ncc_M2, n)`: Processes a single combination of angles and updates relevant statistics.
- `parallelize_processing(volume, phi_angles, theta_angles, psi_angles, I_filtered, projection_shape)`: Parallelizes the processing of angle combinations.

### 3. Main Execution
The main execution block initializes necessary variables, reads the input volume and image files, generates angles, and performs parallel processing to compute cross-correlation maps.

### 4. Display Results
The results are displayed using `matplotlib`:
- Maximum Cross-Correlation Map
- Mean Cross-Correlation Map
- Variance Cross-Correlation Map
- Aggregated MIP Map

### 5. Post-Processing
The notebook includes code to count the number of elements above a threshold in the aggregated MIP map and to display the row and column indices of these elements on the filtered image.

## Usage
1. Ensure all dependencies are installed.
2. Run the notebook cells sequentially.
3. Provide the necessary input parameters when prompted (e.g., file paths, NSIDE parameter, angular spacing, psi angular step).
4. View the generated cross-correlation maps and post-processing results.

## Notes
- The notebook leverages GPU acceleration using `cupy` for faster cross-correlation computations.
- Parallel processing is implemented using `joblib` to handle multiple angle combinations efficiently.
- The notebook is designed to handle large datasets and perform computationally intensive tasks in a reasonable time frame.