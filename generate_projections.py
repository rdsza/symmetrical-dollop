import numpy as np
import mrcfile
import healpy as hp
from scipy.ndimage import rotate
import sys
import os

def read_mrc_file(file_path):
    """Reads a volume from an MRC file."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            volume = mrc.data.astype(np.float32)
            voxel_size = mrc.voxel_size
            if volume.ndim != 3:
                print("Error: Input volume is not 3-dimensional.")
                sys.exit(1)
    except Exception as e:
        print(f"Error reading MRC file: {e}")
        sys.exit(1)
    return volume, voxel_size

def write_mrcs_file(file_path, data_stack, voxel_size):
    """Writes a stack of images to an .mrcs file."""
    try:
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(data_stack)
            mrc.voxel_size = voxel_size
            # Set header fields
            mrc.header.map = b'MAP '
            mrc.header.machst = [68, 65, 0, 0]  # Machine stamp
            mrc.header.nx = data_stack.shape[2]
            mrc.header.ny = data_stack.shape[1]
            mrc.header.nz = data_stack.shape[0]
            mrc.header.mode = 2  # Mode 2 for float32 data
            mrc.header.mx = data_stack.shape[2]
            mrc.header.my = data_stack.shape[1]
            mrc.header.mz = data_stack.shape[0]
            mrc.header.cella = [voxel_size[0] * data_stack.shape[2],
                                voxel_size[1] * data_stack.shape[1],
                                voxel_size[2] * data_stack.shape[0]]
            mrc.header.mapc = 1
            mrc.header.mapr = 2
            mrc.header.maps = 3
            mrc.update_header_from_data()
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

def rotate_and_project(volume, phi, theta, psi):
    """Rotates the volume and projects it along the Z-axis."""
    # Rotate volume by theta around Y-axis
    rotated = rotate(volume, angle=theta, axes=(0, 2), reshape=False, order=1)
    # Rotate volume by phi around Z-axis
    rotated = rotate(rotated, angle=phi, axes=(1, 0), reshape=False, order=1)
    # Rotate volume by psi around Z-axis (in-plane rotation)
    rotated = rotate(rotated, angle=psi, axes=(1, 0), reshape=False, order=1)
    # Project along Z-axis
    projection = np.sum(rotated, axis=0)
    return projection

def check_volume_dimensions(volume):
    """Checks if the volume dimensions are consistent."""
    if volume.ndim != 3:
        print("Error: Input volume is not 3-dimensional.")
        sys.exit(1)
    if volume.shape[0] != volume.shape[1] or volume.shape[1] != volume.shape[2]:
        print("Warning: Volume dimensions are not equal.")
    return volume.shape

def main():
    # User inputs
    volume_file = input("Enter the path to the input MRC volume file: ").strip()
    output_file = input("Enter the path for the output MRCS stack file: ").strip()
    angular_spacing = float(input("Enter the angular spacing for Healpix sampling (in degrees): "))
    psi_step = float(input("Enter the in-plane rotation step size (psi) (in degrees): "))

    # Read the volume
    volume, voxel_size = read_mrc_file(volume_file)
    vol_shape = check_volume_dimensions(volume)

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

    # Initialize the projection stack
    projection_shape = (vol_shape[1], vol_shape[2])  # Assuming projection along Z-axis
    projections = np.zeros((total_projections, *projection_shape), dtype=np.float32)

    # Generate projections
    index = 0
    for i, (phi, theta) in enumerate(zip(phi_angles, theta_angles)):
        for psi in psi_angles:
            # Rotate and project
            projection = rotate_and_project(volume, phi, theta, psi)
            # Check dimensions
            if projection.shape != projection_shape:
                print(f"Warning: Projection dimensions {projection.shape} do not match expected shape {projection_shape}.")
            # Normalize projection (optional)
            projection = projection - np.mean(projection)
            # Store in the stack
            projections[index] = projection
            index += 1
            # Progress update (optional)
            if index % 100 == 0 or index == total_projections:
                print(f"Generated {index}/{total_projections} projections.")

    # Write projections to .mrcs file
    # Use voxel size from input volume
    write_mrcs_file(output_file, projections, voxel_size)
    print(f"Projections saved to '{output_file}'.")

if __name__ == "__main__":
    main()
