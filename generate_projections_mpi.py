import numpy as np
import mrcfile
import healpy as hp
from scipy.ndimage import rotate
import sys
import os
import multiprocessing as mp
import time  # Added for timing

# Global variable for the shared volume
shared_volume = None
volume_shape = None

def init_worker(volume_array, shape):
    """Initialize worker process by setting the global shared volume."""
    global shared_volume
    global volume_shape
    shared_volume = volume_array
    volume_shape = shape

def read_mrc_file(file_path):
    """Reads a volume from an MRC file."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            volume = mrc.data.astype(np.float32)
            header = mrc.header
            if volume.ndim != 3:
                print("Error: Input volume is not 3-dimensional.")
                sys.exit(1)
            if volume.shape[0] != volume.shape[1] or volume.shape[1] != volume.shape[2]:
                print("Warning: Volume dimensions are not equal.")
    except Exception as e:
        print(f"Error reading MRC file: {e}")
        sys.exit(1)
    return volume, header

def write_mrcs_file(file_path, data_stack, header):
    """Writes a stack of images to an .mrcs file."""
    try:
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(data_stack)
            mrc.header = header
            mrc.voxel_size = header.cella / header.mx
    except Exception as e:
        print(f"Error writing MRCS file: {e}")
        sys.exit(1)

def generate_healpix_angles(nside):
    """Generates phi and theta angles using Healpix sampling."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Convert theta from colatitude to latitude
    theta = np.pi/2 - theta
    # Convert radians to degrees
    phi_deg = np.degrees(phi)
    theta_deg = np.degrees(theta)
    return phi_deg, theta_deg

def rotate_and_project_worker(args):
    """Worker function to rotate and project the volume."""
    global shared_volume
    global volume_shape

    phi, theta, psi, index = args

    # Reconstruct the volume from shared memory
    volume_np = np.ctypeslib.as_array(shared_volume)
    volume = volume_np.reshape(volume_shape)

    # Rotate volume by theta around Y-axis
    rotated = rotate(volume, angle=theta, axes=(0, 2), reshape=False, order=1)
    # Rotate volume by phi around Z-axis
    rotated = rotate(rotated, angle=phi, axes=(1, 0), reshape=False, order=1)
    # Rotate volume by psi around Z-axis (in-plane rotation)
    rotated = rotate(rotated, angle=psi, axes=(1, 0), reshape=False, order=1)
    # Project along Z-axis
    projection = np.sum(rotated, axis=0)

    # Normalize projection (optional)
    projection = projection - np.mean(projection)

    return index, projection

def check_volume_dimensions(volume):
    """Checks if the volume dimensions are consistent."""
    if volume.ndim != 3:
        print("Error: Input volume is not 3-dimensional.")
        sys.exit(1)
    if volume.shape[0] != volume.shape[1] or volume.shape[1] != volume.shape[2]:
        print("Warning: Volume dimensions are not equal.")
    return volume.shape

def main():
    # Start the timer
    start_time = time.time()

    # User inputs
    volume_file = input("Enter the path to the input MRC volume file: ").strip()
    output_file = input("Enter the path for the output MRCS stack file: ").strip()
    angular_spacing = float(input("Enter the angular spacing for Healpix sampling (in degrees): "))
    psi_step = float(input("Enter the in-plane rotation step size (psi) (in degrees): "))

    # Read the volume
    volume, header = read_mrc_file(volume_file)
    vol_shape = check_volume_dimensions(volume)

    # Copy the header to make it writable
    header = header.copy()

    # Determine NSIDE parameter based on desired angular spacing
    # The approximate resolution in radians per pixel is given by:
    # resolution ≈ sqrt(π / (3 * nside^2))
    desired_resolution_rad = np.radians(angular_spacing)
    nside_float = np.sqrt(np.pi / (3 * desired_resolution_rad**2))
    # NSIDE must be a power of 2 integer
    nside = 2 ** int(np.ceil(np.log2(nside_float)))
    if not hp.isnsideok(nside):
        print(f"Error: Calculated NSIDE {nside} is not valid.")
        sys.exit(1)
    print(f"Using NSIDE = {nside} for Healpix sampling.")

    # Generate phi and theta angles
    phi_angles, theta_angles = generate_healpix_angles(nside)
    print(f"Generated {len(phi_angles)} orientations using Healpix.")

    # Generate psi angles
    psi_angles = np.arange(0, 360, psi_step)
    print(f"Using {len(psi_angles)} in-plane rotations (psi angles).")

    # Total number of projections
    total_projections = len(phi_angles) * len(psi_angles)
    print(f"Total number of projections to generate: {total_projections}")

    # Initialize the projection shape
    projection_shape = (vol_shape[1], vol_shape[2])  # Assuming projection along Z-axis

    # Create a shared memory array for the volume
    volume_np = np.asarray(volume, dtype=np.float32)
    volume_shared = mp.Array('f', volume_np.flatten(), lock=False)

    # Create argument list for worker function
    args_list = []
    index = 0
    for i, (phi, theta) in enumerate(zip(phi_angles, theta_angles)):
        for psi in psi_angles:
            args_list.append((phi, theta, psi, index))
            index += 1

    # Use multiprocessing Pool to compute projections in parallel
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes for parallel computation.")

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(volume_shared, vol_shape)) as pool:
        # Map the worker function to the arguments
        results = pool.map(rotate_and_project_worker, args_list)

    # Sort results by index to ensure correct order
    results.sort(key=lambda x: x[0])

    # Extract projections from results
    projections = np.zeros((total_projections, *projection_shape), dtype=np.float32)
    for idx, projection in results:
        projections[idx] = projection

    # Update header for projections
    header.nz = total_projections
    header.mz = total_projections
    header.cella.z = total_projections * header.cella.x / header.mx  # Assuming isotropic voxels

    # Write projections to .mrcs file
    write_mrcs_file(output_file, projections, header)
    print(f"Projections saved to '{output_file}'.")

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total computation time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
