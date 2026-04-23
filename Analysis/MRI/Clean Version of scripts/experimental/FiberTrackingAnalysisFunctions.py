import numpy as np
import matplotlib.pyplot as plt





# ============================================================
# 1. COMPRESSION AND FILTER OF TRACTOGRAPHY
# ============================================================
from dipy.tracking.streamline import length as streamline_length

# --- compress_streamlines replacement ---
# Reduces the number of points per streamline while preserving shape
# error_threshold controls how much simplification is allowed (in mm)
def compress_streamlines(streamlines, error_threshold=0.5):
    compressed = []
    for s in streamlines:
        if len(s) <= 2:
            compressed.append(s)
            continue
        # Ramer-Douglas-Peucker style: keep points that deviate > threshold
        keep = [0]  # always keep start
        for i in range(1, len(s) - 1):
            # Distance from point to line between last kept point and end
            v = s[-1] - s[keep[-1]]
            w = s[i] - s[keep[-1]]
            proj = s[keep[-1]] + v * (np.dot(w, v) / np.dot(v, v))
            dist = np.linalg.norm(s[i] - proj)
            if dist > error_threshold:
                keep.append(i)
        keep.append(len(s) - 1)  # always keep end
        compressed.append(s[keep])
    return compressed

# --- select_by_length replacement ---
# Filter streamlines by min/max length
def select_by_length(streamlines, min_length=5.0, max_length=200.0):
    lengths = np.array([streamline_length(s) for s in streamlines])
    mask = (lengths >= min_length) & (lengths <= max_length)
    filtered = [s for s, keep in zip(streamlines, mask) if keep]
    return filtered, lengths[mask], mask


# ============================================================
# 2. ORIENTATION ANALYSIS (per-segment)
# ============================================================
def compute_segment_orientations(streamlines):
    """
    For each segment in each streamline, compute the unit tangent vector.
    Returns a list of arrays, one per streamline, each (N-1, 3).
    """
    orientations = []
    for s in streamlines:
        if len(s) < 2:
            continue
        diffs = np.diff(s, axis=0)                          # (N-1, 3)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms[norms == 0] = 1                               # avoid div by zero
        orientations.append(diffs / norms)
    return orientations

# ============================================================
# 3. ORIENTATION DISTRIBUTION FUNCTION (ODF) — spherical histogram
# ============================================================
def direction_to_spherical(directions):
    """Convert (N,3) unit vectors to (theta, phi) in radians."""
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))          # polar angle from Z
    phi   = np.arctan2(y, x)                       # azimuthal
    return theta, phi

# ============================================================
# 4. PER-STREAMLINE SUMMARY ORIENTATION
# ============================================================
def streamline_mean_orientation(streamlines):
    """
    For each streamline, compute the mean tangent direction
    (endpoint-to-endpoint as an alternative is also common).
    Returns (N, 3) array of unit vectors.
    """
    mean_orientations = []
    for s in streamlines:
        if len(s) < 2:
            mean_orientations.append(np.array([0, 0, 0]))
            continue
        diffs = np.diff(s, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_diffs = diffs / norms
        mean_dir = unit_diffs.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        mean_orientations.append(mean_dir / norm if norm > 0 else np.array([0,0,0]))
    return np.array(mean_orientations)


# ============================================================
# 5. CURVATURE ANALYSIS
# ============================================================
def compute_curvature(streamline):
    """
    Compute curvature at each interior point of a streamline.
    Curvature = |dT/ds| where T is the unit tangent.
    Returns array of curvature values (length N-2).
    """
    if len(streamline) < 3:
        return np.array([])
    diffs = np.diff(streamline, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tangents = diffs / norms

    # Change in tangent
    dtangents = np.diff(tangents, axis=0)           # (N-2, 3)
    dtangent_norms = np.linalg.norm(dtangents, axis=1)

    # Segment lengths (average of adjacent)
    seg_lengths = (norms[:-1, 0] + norms[1:, 0]) / 2
    seg_lengths[seg_lengths == 0] = 1

    return dtangent_norms / seg_lengths


# ============================================================
# 6. TORTUOSITY (ratio of streamline length to straight-line distance)
# ============================================================
def compute_tortuosity(streamlines):
    """
    Tortuosity = arc_length / straight_line_distance.
    1.0 = perfectly straight.
    """
    tortuosities = []
    for s in streamlines:
        if len(s) < 2:
            continue
        arc_len = streamline_length(s)
        straight = np.linalg.norm(s[-1] - s[0])
        if straight > 0:
            tortuosities.append(arc_len / straight)
    return np.array(tortuosities)

def visualize_streamlines_stats(filtered_streamlines, filtered_lengths, plot_each = False, summary_title=None, fig_save_path = None):
    
    # ============================================================
    # 1. LENGTH ANALYSIS
    # ============================================================
    # Get length of every streamline in mm
    print(f"Number of streamlines:  {len(filtered_streamlines)}")
    print(f"Mean length:            {filtered_lengths.mean():.2f} mm")
    print(f"Median length:          {np.median(filtered_lengths):.2f} mm")
    print(f"Min length:             {filtered_lengths.min():.2f} mm")
    print(f"Max length:             {filtered_lengths.max():.2f} mm")
    print(f"Std length:             {filtered_lengths.std():.2f} mm")

    # Filter by length if needed (e.g. remove short spurious streamlines)
    print(f"Streamlines after length filter: {len(filtered_streamlines)}")

    # ============================================================
    # 2. ORIENTATION ANALYSIS (per-segment)
    # ============================================================
    segment_orientations = compute_segment_orientations(filtered_streamlines)

    # Flatten all segment directions into one array for global stats
    all_directions = np.vstack(segment_orientations)  # (total_segments, 3)

    # Mean direction (note: orientations are sign-ambiguous, so flip if needed)
    mean_dir = all_directions.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    print(f"\nMean segment direction (unit): {mean_dir}")


    # ============================================================
    # 3. ORIENTATION DISTRIBUTION FUNCTION (ODF) — spherical histogram
    # ============================================================
    theta, phi = direction_to_spherical(all_directions)
    if plot_each == True:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Polar (theta) distribution ---
        axes[0].hist(np.degrees(theta), bins=90, color='steelblue', edgecolor='k', linewidth=0.5)
        axes[0].set_xlabel('Polar angle θ (degrees from Z)')
        axes[0].set_ylabel('Segment count')
        axes[0].set_title('Polar Angle Distribution')
        axes[0].axvline(90, color='r', linestyle='--', label='XY plane')
        axes[0].legend()

        # --- Azimuthal (phi) distribution ---
        axes[1].hist(np.degrees(phi), bins=180, color='coral', edgecolor='k', linewidth=0.5)
        axes[1].set_xlabel('Azimuthal angle φ (degrees)')
        axes[1].set_ylabel('Segment count')
        axes[1].set_title('Azimuthal Angle Distribution')

        plt.tight_layout()
        plt.savefig('orientation_distributions.png', dpi=150)
        plt.show()

    # ============================================================
    # 4. PER-STREAMLINE SUMMARY ORIENTATION
    # ============================================================
    per_streamline_dirs = streamline_mean_orientation(filtered_streamlines)


    # ============================================================
    # 5. CURVATURE ANALYSIS
    # ============================================================
    all_curvatures = np.concatenate([compute_curvature(s) for s in filtered_streamlines if len(s) >= 3])

    print(f"\nMean curvature:   {all_curvatures.mean():.4f} mm^-1")
    print(f"Median curvature: {np.median(all_curvatures):.4f} mm^-1")
    if plot_each == True:
        plt.figure(figsize=(8, 4))
        plt.hist(all_curvatures, bins=100, color='mediumseagreen', edgecolor='k', linewidth=0.5)
        plt.xlabel('Curvature (mm⁻¹)')
        plt.ylabel('Segment count')
        plt.title('Curvature Distribution')
        plt.savefig('curvature_distribution.png', dpi=150)
        plt.show()


    # ============================================================
    # 6. TORTUOSITY (ratio of streamline length to straight-line distance)
    # ============================================================
    tortuosities = compute_tortuosity(filtered_streamlines)

    print(f"\nMean tortuosity:   {tortuosities.mean():.3f}")
    print(f"Median tortuosity: {np.median(tortuosities):.3f}")
    if plot_each == True:
        plt.figure(figsize=(8, 4))
        plt.hist(tortuosities, bins=80, color='mediumpurple', edgecolor='k', linewidth=0.5)
        plt.xlabel('Tortuosity (arc length / chord length)')
        plt.ylabel('Streamline count')
        plt.title('Tortuosity Distribution')
        plt.axvline(1.0, color='r', linestyle='--', label='Perfectly straight')
        plt.legend()
        plt.savefig('tortuosity_distribution.png', dpi=150)
        plt.show()


    # ============================================================
    # 7. COMBINED SUMMARY PLOT
    # ============================================================
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # Top row: two equal-width plots
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])

    # --- Azimuthal (phi) distribution ---
    ax_top_left.hist(np.degrees(phi), bins=180, color='coral', edgecolor='k', linewidth=0.5)
    ax_top_left.set_xlabel('Azimuthal angle φ (degrees)')
    ax_top_left.set_ylabel('Segment count')
    ax_top_left.set_title('Azimuthal Angle Distribution')

    # --- Polar (theta) distribution ---
    ax_top_right.hist(np.degrees(theta), bins= 180, color='coral', edgecolor='k', linewidth=0.5)
    ax_top_right.set_xlabel('Polar angle θ (deg from Z)')
    ax_top_right.set_title('Segment Orientation (Polar)')
    ax_top_right.axvline(90, color='r', linestyle='--', label='XY plane')
    ax_top_right.legend()

    # Bottom row: three equal-width plots across the full width
    gs_bottom = gs[1, :].subgridspec(1, 3, wspace=0.25)
    ax_len = fig.add_subplot(gs_bottom[0, 0])
    ax_curv = fig.add_subplot(gs_bottom[0, 1])
    ax_tort = fig.add_subplot(gs_bottom[0, 2])

    # --- Streamline length distribution ---
    ax_len.hist(filtered_lengths, bins=80, color='steelblue', edgecolor='k', linewidth=0.5)
    ax_len.set_xlabel('Length (mm)')
    ax_len.set_title('Streamline Length')
    ax_len.set_xlim(0, 25)  # Assuming we filtered out streamlines > 200mm

    # --- Curvature distribution ---
    ax_curv.hist(all_curvatures, bins=100, color='mediumseagreen', edgecolor='k', linewidth=0.5)
    ax_curv.set_xlabel('Curvature (mm⁻¹)')
    ax_curv.set_title('Curvature')
    ax_curv.set_xlim(0, 4)  # Focus on lower curvature values for better visualization
    
    # --- Tortuosity distribution ---
    ax_tort.hist(tortuosities, bins=80, color='mediumpurple', edgecolor='k', linewidth=0.5)
    ax_tort.set_xlabel('Tortuosity (arc length / chord length)')
    ax_tort.set_title('Tortuosity')
    ax_tort.axvline(1.0, color='r', linestyle='--', label='Perfectly straight')
    ax_tort.set_xlim(0, 10)  # Focus on tortuosity > 1 for better visualization
    ax_tort.legend()

    plt.suptitle('Tractography Analysis Summary', fontsize=14, fontweight='bold')
    if summary_title is not None:
        plt.suptitle(summary_title, fontsize=14, fontweight='bold')

    
    plt.tight_layout()
    plt.savefig('tractography_summary.png', dpi=150)

    if fig_save_path is not None:
        plt.savefig(fig_save_path, dpi=300)
    plt.show()



    # ============================================================
# 1. FILTER BY SEED LOCATION (if you saved seed info)
# ============================================================
# This only works if you stored per_streamline_seeds during tractography
# Check if your sft has seed data:
def filter_streamlines_by_seed_z(sft, affine, z_min_vox, z_max_vox):
    if not hasattr(sft, 'data_per_streamline') or 'seeds' not in sft.data_per_streamline:
        raise ValueError("Tractogram does not contain seed information.")
    
    seeds = sft.data_per_streamline['seeds']
    inv_affine = np.linalg.inv(affine)
    
    filtered = []
    for i, s in enumerate(sft.streamlines):
        seed_world = np.append(seeds[i], 1)  # homogeneous coords
        seed_vox = inv_affine @ seed_world
        z_vox = seed_vox[2]
        
        if z_min_vox <= z_vox <= z_max_vox:
            filtered.append(s)
    
    return filtered

# ============================================================
# 2. FILTER BY STREAMLINES PASSING THROUGH A Z-SLICE (most useful)
# ============================================================
def filter_streamlines_by_z_slice(streamlines, affine, z_slice_vox, tolerance_vox=0.5):
    """
    Keep only streamlines that pass through a given z-slice (in voxel coords).
    
    Parameters
    ----------
    streamlines : list
        List of streamlines in world (RASMM) coordinates
    affine : ndarray (4, 4)
        Affine transform from voxel to world
    z_slice_vox : float
        Z slice index in voxel space
    tolerance_vox : float
        How close streamline must pass to the slice (in voxels)
    
    Returns
    -------
    filtered : list
        Streamlines that pass through the slice
    indices : list
        Original indices of kept streamlines
    """
    inv_affine = np.linalg.inv(affine)
    filtered = []
    indices = []
    
    for idx, s in enumerate(streamlines):
        # Convert streamline from world to voxel coords
        s_homogeneous = np.column_stack([s, np.ones(len(s))])  # (N, 4)
        s_vox = (inv_affine @ s_homogeneous.T).T[:, :3]        # (N, 3)
        
        # Check if any point has z within tolerance of target slice
        z_coords = s_vox[:, 2]
        if np.any(np.abs(z_coords - z_slice_vox) <= tolerance_vox):
            filtered.append(s)
            indices.append(idx)
    
    return filtered, indices

# ============================================================
# 3. FILTER BY Z-SLICE RANGE (streamlines that stay within a slab)
# ============================================================
def filter_streamlines_in_z_range(streamlines, affine, z_min_vox, z_max_vox):
    """
    Keep only streamlines where ALL points are within [z_min, z_max].
    Useful for isolating a specific spinal cord segment.
    """
    inv_affine = np.linalg.inv(affine)
    filtered = []
    indices = []
    
    for idx, s in enumerate(streamlines):
        s_homogeneous = np.column_stack([s, np.ones(len(s))])
        s_vox = (inv_affine @ s_homogeneous.T).T[:, :3]
        
        z_coords = s_vox[:, 2]
        if np.all((z_coords >= z_min_vox) & (z_coords <= z_max_vox)):
            filtered.append(s)
            indices.append(idx)
    
    return filtered, indices


# ============================================================
# 4. FILTER BY MASK (e.g., only streamlines passing through WM or GM)
# ============================================================
def filter_streamlines_by_mask(streamlines, affine, mask, min_points=1):
    """
    Keep streamlines that have at least min_points inside the mask.
    
    Parameters
    ----------
    mask : ndarray (bool)
        Binary mask in voxel space
    min_points : int
        Minimum number of streamline points that must be in mask
    """
    inv_affine = np.linalg.inv(affine)
    filtered = []
    indices = []
    
    for idx, s in enumerate(streamlines):
        s_homogeneous = np.column_stack([s, np.ones(len(s))])
        s_vox = (inv_affine @ s_homogeneous.T).T[:, :3]
        
        # Round to nearest voxel
        s_vox_int = np.round(s_vox).astype(int)
        
        # Check bounds
        valid = (
            (s_vox_int[:, 0] >= 0) & (s_vox_int[:, 0] < mask.shape[0]) &
            (s_vox_int[:, 1] >= 0) & (s_vox_int[:, 1] < mask.shape[1]) &
            (s_vox_int[:, 2] >= 0) & (s_vox_int[:, 2] < mask.shape[2])
        )
        s_vox_int = s_vox_int[valid]
        
        # Count points in mask
        if len(s_vox_int) > 0:
            in_mask = mask[s_vox_int[:, 0], s_vox_int[:, 1], s_vox_int[:, 2]]
            if np.sum(in_mask) >= min_points:
                filtered.append(s)
                indices.append(idx)
    
    return filtered, indices