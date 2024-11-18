# Initialize arrays to store the max elements, mean, and variance
ncc_max = np.full_like(I_filtered, -np.inf, dtype=np.float32)
ncc_mean = np.zeros_like(I_filtered, dtype=np.float32)
ncc_M2 = np.zeros_like(I_filtered, dtype=np.float32)  # M2 is the sum of squares of differences from the current mean
n = 0  # Counter for the number of templates processed

comp_start = time.time()

# Loop through each template and update the max, mean, and variance
#for idx, template in enumerate(templates_rotated):
#index = 0
for i, (phi, theta) in enumerate(zip(phi_angles, theta_angles)):    
    for psi in psi_angles:
        #if n == 0: 
        #        start_time = time.time()
        # Rotate and project
        start_time = time.time()
        projection = rotate_and_project(volume, phi, theta, psi)
        end_time = time.time()
        print(f"Time taken to rotate and project : {end_time-start_time} seconds")
        # Check Dimensions
        if projection.shape != projection_shape:
                print(f"Warning: Projection dimensions {projection.shape} do not match expected shape {projection_shape}.")
        # Normalize projection
        projection = projection - np.mean(projection)
        # Band pass filter the projection
        proj_filtered = bandpass_filter(projection, low_sigma=1, high_sigma=5)
        # Normalize the filtered projection
        T_mean = np.mean(proj_filtered)
        T_std = np.std(proj_filtered)
        T_norm = (proj_filtered - T_mean) / T_std
        # Perform cross correlation
        start_time = time.time()
        cross_corr = cross_correlate_top_left(I_filtered, T_norm)
        end_time = time.time()
        print(f"Time taken to calculate cross-correlation : {end_time-start_time} seconds")
        
        
        # Update max elements
        ncc_max = np.maximum(ncc_max, cross_corr)
        
        # Update mean and variance using Welford's algorithm
        n += 1
        delta = cross_corr - ncc_mean
        ncc_mean += delta / n
        delta2 = cross_corr - ncc_mean
        ncc_M2 += delta * delta2


# Calculate the variance
ncc_variance = ncc_M2 / (n - 1) if n > 1 else np.zeros_like(I_filtered, dtype=np.float32)

final = time.time()
elapsed_time = final - comp_start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Cross-correlation computation taken for all {n} templates: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")