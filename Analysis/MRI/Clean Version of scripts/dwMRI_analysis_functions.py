# Dipy
import dipy as dp
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.data as dpd
from dipy.tracking import utils

#Preproc
from dipy.align import motion_correction
import dipy.direction.peaks as dpp
#from dipy.viz import window, actor
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq


#DIPY Plot
from dipy.viz import window, actor, colormap
from dipy.data import get_sphere, default_sphere, get_fnames

#Regular Packages
import keyboard  # For detecting keypresses
import IPython
import pickle #loading pkl files (dictonairy)
import pandas as pd

from pathlib import Path


import numpy as np
import os
from pathlib import Path
from time import time
import time  # For simulating work


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib import colormaps


import scipy.io

#Plot packages
import napari

#Model Packages
from dipy.direction import peaks_from_model
from dipy.reconst.dti import color_fa
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import CsaOdfModel
import dipy.reconst.sfm as sfm
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.tracking.stopping_criterion import (ActStoppingCriterion,
                                              BinaryStoppingCriterion,
                                              ThresholdStoppingCriterion)

def rotate_peaks_per_slice_fast(peaks, rotation_angles):
    """
    Fully vectorized rotation of peak directions (fastest version).
    """
    # Convert all angles to radians
    theta = np.radians(rotation_angles)
    
    # Pre-compute rotation matrices for all slices (n_slices, 3, 3)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    n_slices = len(rotation_angles)
    
    rotation_matrices = np.zeros((n_slices, 3, 3))
    rotation_matrices[:, 0, 0] = cos_theta
    rotation_matrices[:, 0, 1] = -sin_theta
    rotation_matrices[:, 1, 0] = sin_theta
    rotation_matrices[:, 1, 1] = cos_theta
    rotation_matrices[:, 2, 2] = 1.0
    
    # Get all peak directions
    peak_dirs = peaks.peak_dirs  # (nx, ny, nz, n_peaks, 3)
    nx, ny, nz, n_peaks, _ = peak_dirs.shape
    
    # Reshape for vectorized rotation
    # (nx, ny, nz, n_peaks, 3) -> (nx, ny, nz, n_peaks, 3)
    # Apply rotation per z-slice: einsum over batch dimensions
    # 'xyzpc,zcd->xyzpd' where x,y=spatial, z=slice, p=peak, c,d=coords
    rotated = np.einsum('xyzpc,zcd->xyzpd', peak_dirs, rotation_matrices)
    
    # Normalize all peaks at once
    norms = np.linalg.norm(rotated, axis=-1, keepdims=True)
    mask = norms[..., 0] > 1e-10
    rotated[mask] = rotated[mask] / norms[mask]
    
    # Update peaks
    peaks.peak_dirs[:] = rotated
    
    return peaks

from scipy.ndimage import affine_transform

def rotate_peaks_spatial_and_directional(peaks, rotation_angles, data_shape):
    """
    Rotate both spatial positions AND peak directions per slice.
    
    Parameters:
    -----------
    peaks : PeaksAndMetrics object
        Output from peaks_from_model
    rotation_angles : array
        Rotation angle per slice in degrees
    data_shape : tuple
        Shape of the original data (nx, ny, nz)
    
    Returns:
    --------
    peaks_rotated : PeaksAndMetrics object
        Peaks with rotated positions and directions
    """
    from dipy.direction import PeaksAndMetrics
    
    nx, ny, nz = data_shape
    n_peaks = peaks.peak_dirs.shape[3]
    
    # Initialize new peaks arrays
    new_peak_dirs = np.zeros_like(peaks.peak_dirs)
    new_peak_values = np.zeros_like(peaks.peak_values)
    new_peak_indices = np.zeros_like(peaks.peak_indices)
    
    for z in range(nz):
        angle = rotation_angles[z]
        theta = np.radians(angle)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 2D rotation matrix for directions
        rotation_matrix_3d = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ])
        
        # 2D affine matrix for spatial transformation
        # Rotate around center of the slice
        center = np.array([nx/2, ny/2])
        
        rotation_matrix_2d = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        
        # Apply spatial rotation to each array in the slice
        for peak_idx in range(n_peaks):
            # Rotate peak directions
            peak_dir_slice = peaks.peak_dirs[:, :, z, peak_idx, :]
            rotated_dirs = np.einsum('xyc,cd->xyd', peak_dir_slice, rotation_matrix_3d)
            
            # Normalize
            norms = np.linalg.norm(rotated_dirs, axis=-1, keepdims=True)
            mask = norms[..., 0] > 1e-10
            rotated_dirs[mask] = rotated_dirs[mask] / norms[mask]
            
            # Rotate spatial positions of peak values
            peak_val_slice = peaks.peak_values[:, :, z, peak_idx]
            
            # Create affine transformation matrix for scipy
            # We need to rotate around the center
            offset_before = center
            offset_after = center - rotation_matrix_2d @ center
            
            # Apply spatial transformation (inverse rotation for resampling)
            rotated_vals = affine_transform(
                peak_val_slice,
                rotation_matrix_2d.T,  # inverse rotation
                offset=offset_after,
                order=1,
                mode='constant',
                cval=0
            )
            
            new_peak_dirs[:, :, z, peak_idx, :] = rotated_dirs
            new_peak_values[:, :, z, peak_idx] = rotated_vals
    
    # Create new peaks object
    peaks_rotated = PeaksAndMetrics()
    peaks_rotated.peak_dirs = new_peak_dirs
    peaks_rotated.peak_values = new_peak_values
    peaks_rotated.peak_indices = peaks.peak_indices  # These don't need rotation
    peaks_rotated.shm_coeff = peaks.shm_coeff if hasattr(peaks, 'shm_coeff') else None
    peaks_rotated.gfa = peaks.gfa
    
    return peaks_rotated

import numpy as np


# =========================================================================== #
#  PEAKS                                                                       #
# =========================================================================== #

def flatten_peaks(peak_dirs, peak_vals=None, roi_mask=None):
    """
    Returns:
        dirs_flat   (N, Np, 3)
        vals_flat   (N, Np) or None
        roi_flat    (N, 1)  or None
    """
    X, Y, Z, Np, _ = peak_dirs.shape
    dirs = peak_dirs.reshape(-1, Np, 3)
    vals = peak_vals.reshape(-1, Np)          if peak_vals is not None else None
    roi  = roi_mask.reshape(-1, 1)            if roi_mask  is not None else None
    return dirs, vals, roi


def valid_peak_mask(dirs, vals=None, roi=None, min_peak_value=0.0):
    """
    Returns bool mask (N, Np) — True where peak is valid.
    """
    norm  = np.linalg.norm(dirs, axis=-1)
    valid = norm > 1e-6



    if vals is not None:
        valid &= vals > min_peak_value
    if roi is not None:
        valid &= roi          # broadcasts over Np

    return valid


def az_el_from_dirs(dirs, valid_mask):
    """
    Standard hemisphere convention:
        Azimuth  : 0° – 360°  (arctan2 around z-axis, East-of-North compass)
        Elevation: 0° –  90°  (angle up from transverse plane)

    Vectors pointing into the lower hemisphere (z < 0) are flipped
    before angle computation so every peak maps to the upper hemisphere.

    Parameters
    ----------
    dirs      : (N, Np, 3)  — raw (possibly un-normalised) peak directions
    valid_mask: (N, Np) bool

    Returns
    -------
    az, el : 1-D arrays (radians), length = valid_mask.sum()
    """
    # Normalise
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    v    = dirs / (norm + 1e-8)

    # Enforce upper hemisphere
    flip    = v[..., 2] < 0
    v[flip] = -v[flip]

    # Azimuth: full 0 – 2π  (use x,y components; % 2π wraps negatives)
    az = np.arctan2(v[..., 0], v[..., 1]) % (2 * np.pi)
    # Elevation: 0 – π/2  (z >= 0 guaranteed, so arcsin or arctan2 both work)
    el = np.arctan2(v[..., 2], np.sqrt(v[..., 0]**2 + v[..., 1]**2))

    return az[valid_mask], el[valid_mask]


def compute_az_el_fast(
    peak_dirs,
    peak_vals=None,
    roi_mask=None,
    min_peak_value=0.0,
    degrees=False,
):
    """
    Compute azimuth / elevation for CSD/DTI peak directions.

    Returns
    -------
    az, el, weights                          – all valid peaks
    az_has_transverse, el_has_transverse,
    weights_transverse                       – subset with a transverse component
    """
    dirs, vals, roi = flatten_peaks(peak_dirs, peak_vals, roi_mask)
    valid           = valid_peak_mask(dirs, vals, roi, min_peak_value)

    az, el = az_el_from_dirs(dirs, valid)

    # Subset: peaks that have a meaningful transverse (x,y) component
    # Evaluated on the *valid* subset that az_el_from_dirs already returned
    dirs_valid       = dirs[valid]                                   # (M, 3)
    transverse_norm  = np.linalg.norm(dirs_valid[:, :2], axis=-1)
    has_transverse   = transverse_norm > 1e-8
    

    az_has_transverse = az[has_transverse]
    el_has_transverse = el[has_transverse]

    if vals is not None:
        weights              = vals[valid]
        weights_transverse   = weights[has_transverse]
    else:
        weights              = None
        weights_transverse   = None

    if degrees:
        az                 = np.degrees(az)
        el                 = np.degrees(el)
        az_has_transverse  = np.degrees(az_has_transverse)
        el_has_transverse  = np.degrees(el_has_transverse)
    output_dict = {"az": az, "el": el, "weights": weights, "az_has_transverse": az_has_transverse, "el_has_transverse": el_has_transverse, "weights_transverse": weights_transverse, "has_transverse": has_transverse}
    return output_dict


# =========================================================================== #
#  fODF                                                                        #
# =========================================================================== #

# =========================================================================== #
#  fODF                                                                        #
# =========================================================================== #

def flatten_odf(odf, sphere, roi_mask=None):
    """
    odf      : (X, Y, Z, N_dirs)
    sphere   : dipy Sphere with .vertices (N_dirs, 3)
    roi_mask : (X, Y, Z) bool, optional

    Returns
    -------
    dirs_flat : (N_voxels, N_dirs, 3)
    amps_flat : (N_voxels, N_dirs)
    roi_flat  : (N_voxels, 1) or None
    """
    X, Y, Z, N_dirs = odf.shape
    amps_flat = odf.reshape(-1, N_dirs)
    dirs_flat = np.tile(sphere.vertices, (amps_flat.shape[0], 1, 1))
    roi_flat  = roi_mask.reshape(-1, 1) if roi_mask is not None else None
    return dirs_flat, amps_flat, roi_flat


def valid_odf_mask(amps, roi=None, min_amp=0.0):
    """
    amps : (N_voxels, N_dirs)
    Returns bool mask of same shape.
    """
    valid = amps > min_amp
    if roi is not None:
        valid &= roi
    return valid


def az_el_from_odf(sphere, amps, valid_mask, min_transverse=1e-2):
    """
    Standard hemisphere convention for fODF directions.

    Sphere vertices with z < 0 are mapped to their antipode before
    angle computation. Near-pole directions (transverse norm < min_transverse)
    are flagged via has_az.

    Parameters
    ----------
    sphere         : dipy Sphere
    amps           : (N_voxels, N_dirs)
    valid_mask     : (N_voxels, N_dirs) bool
    min_transverse : pole exclusion threshold

    Returns
    -------
    az, el   : (M,) radians
    weights  : (M,) fODF amplitudes
    has_az   : (M,) bool — True where azimuth is meaningful
    """
    v = sphere.vertices.copy()   # (N_dirs, 3) unit vectors

    # Enforce upper hemisphere
    flip    = v[:, 2] < 0
    v[flip] = -v[flip]

    # Azimuth 0 – 2π
    az = np.arctan2(v[:, 0], v[:, 1]) % (2 * np.pi)   # (N_dirs,)

    # Elevation 0 – π/2
    transverse_norm = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    el = np.arctan2(v[:, 2], transverse_norm)           # (N_dirs,)

    # Pole mask per direction (broadcast to match valid_mask shape)
    has_az_dirs = transverse_norm > min_transverse       # (N_dirs,)
    has_az_full = np.broadcast_to(has_az_dirs, amps.shape)  # (N_voxels, N_dirs)

    # Broadcast angles over voxels
    az_full = np.broadcast_to(az, amps.shape)
    el_full = np.broadcast_to(el, amps.shape)

    return (
        az_full[valid_mask],
        el_full[valid_mask],
        amps[valid_mask],
        has_az_full[valid_mask],
    )


def compute_az_el_odf(
    odf,
    sphere,
    roi_mask=None,
    min_amp=0.0,
    min_transverse=1e-2,
    degrees=True,
):
    """
    Main entry point for fODF data — mirrors compute_az_el_fast().

    Parameters
    ----------
    odf            : (X, Y, Z, N_dirs)
    sphere         : dipy Sphere object
    roi_mask       : (X, Y, Z) bool, optional
    min_amp        : threshold to ignore near-zero amplitudes
    min_transverse : pole exclusion threshold
    degrees        : return degrees if True, radians if False

    Returns
    -------
    az, el, weights : all valid directions
    az_az, el_az, w_az : subset where azimuth is meaningful (has_az=True)
    has_az          : (M,) bool mask on the valid subset
    """
    dirs_flat, amps_flat, roi_flat = flatten_odf(odf, sphere, roi_mask)
    valid = valid_odf_mask(amps_flat, roi_flat, min_amp)

    az, el, weights, has_az = az_el_from_odf(sphere, amps_flat, valid, min_transverse)

    az_az = az[has_az]
    el_az = el[has_az]
    w_az  = weights[has_az]

    if degrees:
        az    = np.degrees(az)
        el    = np.degrees(el)
        az_az = np.degrees(az_az)
        el_az = np.degrees(el_az)
    output_dict = {"az": az, "el": el, "weights": weights, "az_az": az_az, "el_az": el_az, "w_az": w_az, "has_az": has_az}
    return output_dict



###################################################################
########################## Plot Function ##########################
###################################################################
def plot_az_el_summary_deg(
    az_deg,
    el_deg,
    weights=None,
    az_bins=36,
    el_bins=36,
    figsize=(4, 12), 
    save_path=None,
    title="Azimuth and Elevation Summary",
    cmap="viridis",
    dpi=1200,
    colorbar_fraction = False,
    horizontal_layout= False,
    show = False
):
    """
    az_deg : array-like, azimuth angles in degrees [0, 180]
    el_deg : array-like, elevation angles in degrees [-90, 90]
    weights : optional weights (e.g. peak amplitudes)
    """
    az_edges = np.linspace(0, 180, az_bins + 1)
    el_edges = np.linspace(0, 90, el_bins + 1)

    if horizontal_layout:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(3, 1, figsize=figsize)

    # -----------------------
    # Azimuth histogram
    # -----------------------
    az_counts, _ = np.histogram(az_deg, bins=az_edges, weights=weights)
    az_total = az_counts.sum()
    az_frac = az_counts / az_total if az_total > 0 else az_counts

    bin_centers_az = 0.5 * (az_edges[:-1] + az_edges[1:])
    bin_width_az = np.diff(az_edges)

    # Left axis: fraction
    axes[0].bar(bin_centers_az, az_frac, width=bin_width_az, edgecolor="k", align="center")
    axes[0].axvline(90, color="k", linestyle="--", linewidth=0.5, label="90° reference (dorsal-ventral)")
    axes[0].legend()
    axes[0].set_xlabel("Azimuth (°)")
    axes[0].set_ylabel("Fraction")
    axes[0].set_xlim(0, 180)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Azimuth distribution")

    # Right axis: count
    ax0_right = axes[0].twinx()
    ax0_right.set_ylim(0, az_counts.max())
    ax0_right.set_ylabel("Weighted count" if weights is not None else "Count")

    # -----------------------
    # Elevation histogram
    # -----------------------
    el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)
    el_total = el_counts.sum()
    el_frac = el_counts / el_total if el_total > 0 else el_counts

    bin_centers_el = 0.5 * (el_edges[:-1] + el_edges[1:])
    bin_width_el = np.diff(el_edges)

    axes[1].bar(bin_centers_el, el_frac, width=bin_width_el, edgecolor="k", align="center")
    axes[1].axvline(0, color="k", linestyle="--", linewidth=0.5, label="0° (Transverse plane)")
    axes[1].legend()
    axes[1].set_xlabel("Elevation (°)")
    axes[1].set_ylabel("Fraction")
    axes[1].set_xlim(0, 90)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Elevation distribution")

    ax1_right = axes[1].twinx()
    ax1_right.set_ylim(0, el_counts.max())
    ax1_right.set_ylabel("Weighted count" if weights is not None else "Count")

    # -----------------------
    # Azimuth–Elevation 2D density
    # -----------------------
    H, _, _ = np.histogram2d(az_deg, el_deg, bins=[az_edges, el_edges], weights=weights)
    H_frac = H / H.sum() if H.sum() > 0 else H  # Convert to fraction
    
    #Clip away 0 to avoid log(0) issues
    #H = np.clip(H, 1e-6, None) 
    #H_frac = np.clip(H_frac, 1e-10, None)

    if colorbar_fraction == True:
        im = axes[2].imshow(H_frac.T,  # transpose for correct orientation
            origin="lower",
            aspect="equal",
            extent=[0, 180, 0, 90],
            cmap=cmap,
            interpolation="lanczos",
            #norm=LogNorm(vmin=H_frac.min(), vmax=H_frac.max()),
            )
        

        axes[2].set_xlabel("Azimuth (°)")
        axes[2].set_ylabel("Elevation (°)")
        axes[2].set_title("Azimuth–Elevation density (fraction)")
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Fraction")
        cbar.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
    else:
        im = axes[2].imshow( H.T, origin="lower", aspect="equal", extent=[0, 180, 0, 90], cmap=cmap, interpolation="lanczos",
                            #norm=LogNorm(vmin=H.min(), vmax=H.max())
                            ) 
        axes[2].set_xlabel("Azimuth (°)") 
        axes[2].set_ylabel("Elevation (°)") 
        axes[2].set_title("Azimuth–Elevation density") 
        cbar = fig.colorbar( im, ax=axes[2], fraction=0.046, pad=0.04 ) 
        cbar.set_label("Weighted count" if weights is not None else "Count")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format="svg", bbox_inches="tight")
    if show == True:
        plt.show()



def plot_az_el_summary_deg_individual(
    az_deg,
    el_deg,
    weights=None,
    az_bins=36,
    el_bins=36,
    figsize=(5, 5),
    save_path=None,
    title="Azimuth and Elevation Summary",
    dpi=1200
):
    """
    az_deg : array-like, azimuth angles in degrees [0, 180]
    el_deg : array-like, elevation angles in degrees [-90, 90]
    weights : optional weights (e.g. peak amplitudes)
    """

    az_edges = np.linspace(0, 180, az_bins + 1)
    el_edges = np.linspace(-90, 90, el_bins + 1)

    save_path = Path(save_path) if save_path is not None else None

    # =========================
    # Azimuth histogram
    # =========================
    az_counts, _ = np.histogram(
        az_deg,
        bins=az_edges,
        weights=weights
    )

    az_total = az_counts.sum()
    az_frac = az_counts / az_total if az_total > 0 else az_counts

    fig, ax_frac = plt.subplots(figsize=figsize)

    bin_centers = 0.5 * (az_edges[:-1] + az_edges[1:])
    bin_width = np.diff(az_edges)

    ax_frac.bar(
        bin_centers,
        az_frac,
        width=bin_width,
        edgecolor="k",
        align="center"
    )
    ax_frac.set_xlabel("Azimuth (°)")
    ax_frac.axvline(90, color="k", linestyle="--", linewidth=0.5, label="90° reference (dorsal-ventral)")
    ax_frac.legend()

    ax_frac.set_ylabel("Fraction")
    ax_frac.set_title("Azimuth distribution")
    ax_frac.set_xlim(0, 180)
    ax_frac.set_ylim(0, 1)
    
    ax_cnt = ax_frac.twinx()
    ax_cnt.set_ylabel("Weighted count" if weights is not None else "Count")
    ax_cnt.set_ylim(0, az_counts.max())


    if save_path is not None:
        plt.savefig(
            save_path.with_name(save_path.stem + "_azimuth.svg"),
            dpi=dpi,
            bbox_inches="tight"
        )
    plt.show()
    plt.close()

    # =========================
    # Elevation histogram
    # =========================
    el_counts, _ = np.histogram(
        el_deg,
        bins=el_edges,
        weights=weights
    )

    el_total = el_counts.sum()
    el_frac = el_counts / el_total if el_total > 0 else el_counts

    fig, ax_frac = plt.subplots(figsize=figsize)

    bin_centers = 0.5 * (el_edges[:-1] + el_edges[1:])
    bin_width = np.diff(el_edges)

    ax_frac.bar(
        bin_centers,
        el_frac,
        width=bin_width,
        edgecolor="k",
        align="center"
    )
    ax_frac.set_xlabel("Elevation (°)")
    ax_frac.set_ylabel("Fraction")
    ax_frac.set_title("Elevation distribution")
    ax_frac.set_xlim(-90, 90)
    ax_frac.set_ylim(0, 1)

    ax_cnt = ax_frac.twinx()
    ax_cnt.set_ylabel("Weighted count" if weights is not None else "Count")
    ax_cnt.set_ylim(0, el_counts.max())

    if save_path is not None:
        plt.savefig(
            save_path.with_name(save_path.stem + "_elevation.svg"),
            dpi=dpi,
            bbox_inches="tight"
        )
    plt.close()
    plt.show()
    # =========================
    # Azimuth–Elevation density
    # =========================
    H, _, _ = np.histogram2d(
        az_deg,
        el_deg,
        bins=[az_edges, el_edges],
        weights=weights
    )

    plt.figure(figsize=figsize)
    im = plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[0, 180, -90, 90],
        interpolation="lanczos"
    )
    plt.xlabel("Azimuth (°)")
    plt.ylabel("Elevation (°)")
    plt.title("Azimuth–Elevation density")

    cbar = plt.colorbar(im)
    cbar.set_label("Weighted count" if weights is not None else "Count")

    if save_path is not None:
        plt.savefig(
            save_path.with_name(save_path.stem + "_az_el_density.svg"),
            dpi=dpi,
            bbox_inches="tight"
        )
    plt.show()
    plt.close()


from matplotlib.colors import Normalize

def plot_az_el_summary_polar(
    az_deg,
    el_deg,
    weights=None,
    weights_transverse = None,
    az_bins=36,
    el_bins=36,
    figsize=(14, 5),
    save_path=None,
    title="Azimuth and Elevation Summary (Polar)",
    cmap="viridis",
    dpi=300,
    colorbar_fraction=False,
    show=False,
):
    """
    Polar-plot version of the azimuth/elevation summary.

    Parameters
    ----------
    az_deg : array-like
        Azimuth angles in degrees [0, 180].
    el_deg : array-like
        Elevation angles in degrees [-90, 90].
    weights : array-like, optional
        Per-sample weights (e.g. peak amplitudes).
    az_bins : int
        Number of azimuth histogram bins.
    el_bins : int
        Number of elevation histogram bins.
    figsize : tuple
        Figure size (width, height). If height >= width, panels are stacked
        vertically (3x1); otherwise they are placed side-by-side (1x3).
    save_path : str or None
        If given, save the figure to this path as SVG.
    title : str
        Figure suptitle.
    cmap : str
        Colormap name for all panels.
    dpi : int
        Resolution used when saving.
    colorbar_fraction : bool
        If True, colour-encode the 2-D panel by fraction; otherwise by count.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)

    # ------------------------------------------------------------------ #
    #  Histogram data                                                      #
    # ------------------------------------------------------------------ #
    az_edges = np.linspace(0, 180, az_bins + 1)
    el_edges = np.linspace(0, 90, el_bins + 1)

    
    #az_counts, _ = np.histogram(az_deg, bins=az_edges, weights=weights)
    #el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)

    if weights is None and weights_transverse is None:
        az_counts, _ = np.histogram(az_deg, bins=az_edges)
        el_counts, _ = np.histogram(el_deg, bins=el_edges)
        
    if weights is not None:
        try:
            az_counts, _ = np.histogram(az_deg, bins=az_edges, weights=weights)
            el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)
        except ValueError:
            print("Shape is not matching, check the shape of weights and angles")

    if weights_transverse is not None:
        az_counts, _ = np.histogram(az_deg, bins=az_edges, weights = weights_transverse)
        el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)

    az_total = az_counts.sum()
    el_total = el_counts.sum()
    az_frac = az_counts / az_total if az_total > 0 else az_counts
    el_frac = el_counts / el_total if el_total > 0 else el_counts

    bin_centers_az = np.deg2rad(0.5 * (az_edges[:-1] + az_edges[1:]))
    bin_width_az   = np.deg2rad(np.diff(az_edges))

    bin_centers_el = np.deg2rad(0.5 * (el_edges[:-1] + el_edges[1:]))
    bin_width_el   = np.deg2rad(np.diff(el_edges))

    count_label = "Weighted count" if weights is not None else "Count"

    # 2-D density for panel 3 (|elevation| as radius)
    el_abs  = np.abs(el_deg)
    r_edges = np.linspace(0, 90, el_bins + 1)
    # ------------------------------------------------------------------ #
    #  Figure layout – vertical when figsize is portrait, else horizontal #
    # ------------------------------------------------------------------ #
    w, h     = figsize
    vertical = h >= w

    fig = plt.figure(figsize=figsize)
    if vertical:
        ax_az = fig.add_subplot(311, projection="polar")
        ax_el = fig.add_subplot(312, projection="polar")
    else:
        ax_az = fig.add_subplot(131, projection="polar")
        ax_el = fig.add_subplot(132, projection="polar")

    # Resolve colormap once (compatible with all matplotlib versions)
    cmap_obj = colormaps[cmap]

    # ------------------------------------------------------------------ #
    #  Panel 1 – Azimuth polar bar                                        #
    # ------------------------------------------------------------------ #
    norm_az   = Normalize(vmin=0, vmax=az_frac.max() if az_frac.max() > 0 else 1)
    colors_az = cmap_obj(norm_az(az_frac))

    ax_az.bar(
        bin_centers_az,
        az_frac,
        width=bin_width_az,
        bottom=0,
        color=colors_az,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )
    ax_az.axvline(np.deg2rad(90), color="crimson", linestyle="--",
                  linewidth=1.0, label="90° (dorsal–ventral)")


    ax_az.set_thetamin(0)
    ax_az.set_thetamax(180)
    ax_az.set_theta_zero_location("W")
    ax_az.set_theta_direction(-1)
    ax_az.set_rlabel_position(45)
    ax_az.set_title("Azimuth distribution", va="bottom", pad=18)
    ax_az.legend(loc="upper center", fontsize=7)

    for label in ax_az.get_yticklabels():   # yticklabels = radial (fraction) ticks
        label.set_rotation(-45)          # rotate to match bars; negative for clockwise
        label.set_ha("right")               # realign after rotation so they don't drift

    # ------------------------------------------------------------------ #
    #  Panel 2 – Elevation polar bar                                      #
    # ------------------------------------------------------------------ #
    norm_el   = Normalize(vmin=0, vmax=el_frac.max() if el_frac.max() > 0 else 1)
    colors_el = cmap_obj(norm_el(el_frac))

    ax_el.bar(
        bin_centers_el,
        el_frac,
        width=bin_width_el,
        bottom=0,
        color=colors_el,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )

    ax_el.axvline(0, color="crimson", linestyle="--",
                  linewidth=1.0, label="0° (transverse plane)")

    ax_el.set_thetamin(0)
    ax_el.set_thetamax(90)
    ax_el.set_theta_zero_location("W")
    ax_el.set_theta_direction(-1)
    ax_el.set_rlabel_position(22)

    ax_el.set_title("Elevation distribution", va="bottom", pad=18)
    ax_el.legend(loc="upper left", bbox_to_anchor=(.25, 1.155), fontsize=7)

    # ------------------------------------------------------------------ #
    #  Finish                                                              #
    # ------------------------------------------------------------------ #
    for label in ax_el.get_yticklabels():
        label.set_va("bottom")   # anchor at bottom → text grows upward

    ax_az.tick_params(axis="y", pad=10, labelrotation = 45)
    ax_el.tick_params(axis="y", pad=10, labelrotation = 0 )
    from matplotlib.ticker import FuncFormatter

    percent_fmt = FuncFormatter(lambda x, _: f"{100*x:.0f}%")

    ax_az.yaxis.set_major_formatter(percent_fmt)
    ax_el.yaxis.set_major_formatter(percent_fmt)
    for ax in (ax_az, ax_el):
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_ha("left")
            label.set_va("center")
    plt.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format="svg", bbox_inches="tight")
    if show:
        plt.show()

def plot_hemisphere_summary_polar(
    az_deg,
    el_deg,
    weights=None,
    az_has_transverse=None,
    el_has_transverse=None,
    weights_transverse=None,
    az_bins=36,
    el_bins=18,
    figsize=(14, 5),
    save_path=None,
    title="Hemisphere Peak Directions",
    cmap="viridis",
    dpi=300,
    colorbar_fraction=False,
    show=False,
):
    """
    Hemisphere summary using standard spherical convention:
        Azimuth  : 0° – 360°
        Elevation: 0° –  90°

    Designed to accept the direct output of compute_az_el_fast():

        az, el, weights, az_has_transverse, el_has_transverse, weights_transverse \\
            = compute_az_el_fast(..., degrees=True)

        plot_hemisphere_summary_polar(
            az_deg             = az,
            el_deg             = el,
            weights            = weights,
            az_has_transverse  = az_has_transverse,
            el_has_transverse  = el_has_transverse,
            weights_transverse = weights_transverse,
        )

    Panels:
        1. Azimuth polar bar     – az_has_transverse (pole-excluded)
        2. Elevation polar bar   – el_deg / weights (all valid, including pole)
        3. 2-D hemispherical map – az_has_transverse x el_has_transverse (pole-excluded)

    If az_has_transverse is not provided, az_deg is used for all three panels.

    Parameters
    ----------
    az_deg             : all valid azimuths [0, 360°]
    el_deg             : all valid elevations [0, 90°]
    weights            : weights for az_deg / el_deg
    az_has_transverse  : pole-excluded azimuths — panels 1 & 3
    el_has_transverse  : pole-excluded elevations — panels 1 & 3
    weights_transverse : weights for the pole-excluded subset
    az_bins            : number of azimuth bins
    el_bins            : number of elevation bins
    figsize            : (w, h). Portrait -> vertical stack; landscape -> side-by-side
    save_path          : SVG output path
    title              : figure suptitle
    cmap               : colormap name
    dpi                : save resolution
    colorbar_fraction  : normalise 2-D panel by fraction
    show               : call plt.show()

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)

    # Panels 1 & 3: use pole-excluded subset if provided, else fall back to full
    if az_has_transverse is not None:
        az_p13   = np.asarray(az_has_transverse, dtype=float)
        el_p13   = np.asarray(el_has_transverse, dtype=float)
        w_p13    = weights_transverse
        n_total    = len(az_deg)
        n_excluded = n_total - len(az_p13)
        pole_str   = (
            f"Near-pole excluded: {n_excluded} / {n_total} peaks "
            f"({n_excluded / n_total * 100:.1f}%)"
        )
    else:
        az_p13   = az_deg
        el_p13   = el_deg
        w_p13    = weights
        pole_str = None

    # ------------------------------------------------------------------ #
    #  Histograms                                                          #
    # ------------------------------------------------------------------ #
    az_edges = np.linspace(0,  360, az_bins + 1)
    el_edges = np.linspace(0,   90, el_bins + 1)

    # Panel 1: azimuth — pole-excluded
    az_counts, _ = np.histogram(az_p13, bins=az_edges, weights=w_p13)
    # Panel 2: elevation — all valid (pole peaks have meaningful el = 90°)
    el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)

    az_total = az_counts.sum()
    el_total = el_counts.sum()
    az_frac  = az_counts / az_total if az_total > 0 else az_counts
    el_frac  = el_counts / el_total if el_total > 0 else el_counts

    bin_centers_az = np.deg2rad(0.5 * (az_edges[:-1] + az_edges[1:]))
    bin_width_az   = np.deg2rad(np.diff(az_edges))

    bin_centers_el = np.deg2rad(0.5 * (el_edges[:-1] + el_edges[1:]))
    bin_width_el   = np.deg2rad(np.diff(el_edges))

    count_label = "Weighted count" if weights is not None else "Count"

    # ------------------------------------------------------------------ #
    #  2-D density — pole-excluded                                       #
    #  Elevation inverted: 90° pole -> centre (r=0), 0° equator -> edge  #
    # ------------------------------------------------------------------ #
    el_inv  = 90.0 - el_p13
    r_edges = np.linspace(0, 90, el_bins + 1)

    H2, _, _ = np.histogram2d(
        az_p13, el_inv, bins=[az_edges, r_edges], weights= w_p13 if w_p13 is not None else None
    )
    if colorbar_fraction:
        H2_plot       = H2 / H2.sum() if H2.sum() > 0 else H2
        density_label = "Fraction"
    else:
        H2_plot       = H2
        density_label = count_label

    # ------------------------------------------------------------------ #
    #  Figure layout                                                       #
    # ------------------------------------------------------------------ #
    w, h     = figsize
    vertical = h >= w

    fig = plt.figure(figsize=figsize)
    if vertical:
        ax_az = fig.add_subplot(311, projection="polar")
        ax_el = fig.add_subplot(312, projection="polar")
        ax_2d = fig.add_subplot(313, projection="polar")
    else:
        ax_az = fig.add_subplot(131, projection="polar")
        ax_el = fig.add_subplot(132, projection="polar")
        ax_2d = fig.add_subplot(133, projection="polar")

    cmap_obj = colormaps[cmap]

    # ------------------------------------------------------------------ #
    #  Panel 1 – Azimuth (pole-excluded)                                  #
    # ------------------------------------------------------------------ #
    norm_az   = Normalize(vmin=0, vmax=az_frac.max() if az_frac.max() > 0 else 1)
    colors_az = cmap_obj(norm_az(az_frac))

    ax_az.bar(
        bin_centers_az,
        az_frac,
        width=bin_width_az,
        bottom=0,
        color=colors_az,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )

    ax_az.axvline(0, color="crimson", linestyle="--", linewidth=1.0, label="0° (North)")

    ax_az.set_theta_zero_location("E")
    ax_az.set_theta_direction(1)
    ax_az.set_rlabel_position(45)
    ax_az.yaxis.set_tick_params(labelsize=6)
    az_title = "Azimuth distribution\n(pole-excluded)" if az_has_transverse is not None \
               else "Azimuth distribution"
    ax_az.set_title(az_title, va="bottom", pad=18)
    ax_az.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7)

    # ------------------------------------------------------------------ #
    #  Panel 2 – Elevation (all valid peaks, including pole)              #
    # ------------------------------------------------------------------ #
    norm_el   = Normalize(vmin=0, vmax=el_frac.max() if el_frac.max() > 0 else 1)
    colors_el = cmap_obj(norm_el(el_frac))

    ax_el.bar(
        bin_centers_el,
        el_frac,
        width=bin_width_el,
        bottom=0,
        color=colors_el,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )

    ax_el.axvline(0,              color="crimson",   linestyle="--",
                  linewidth=1.0, label="0° (equator)")
    ax_el.axvline(np.deg2rad(90), color="steelblue", linestyle="--",
                  linewidth=1.0, label="90° (pole)")

    ax_el.set_thetamin(0)
    ax_el.set_thetamax(90)
    ax_el.set_theta_zero_location("W")
    ax_el.set_theta_direction(-1)
    ax_el.set_rlabel_position(90)
    ax_el.yaxis.set_tick_params(labelsize=6)
    el_title = "Elevation distribution\n(all valid peaks)" if az_has_transverse is not None \
               else "Elevation distribution"
    ax_el.set_title(el_title, va="bottom", pad=18)
    ax_el.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=7)

    # ------------------------------------------------------------------ #
    #  Panel 3 – 2-D hemispherical density (pole-excluded)               #
    # ------------------------------------------------------------------ #
    az_rad_edges = np.deg2rad(az_edges)
    r_norm_edges = r_edges / 90.0

    Az, R = np.meshgrid(az_rad_edges, r_norm_edges)
    C     = H2_plot.T

    mesh = ax_2d.pcolormesh(Az, R, C, cmap=cmap_obj, shading="auto")

    ax_2d.set_theta_zero_location("E")
    ax_2d.set_theta_direction(1)
    ax_2d.set_rlim(0, 1)
    ax_2d.set_rticks([0,0.25, 0.5, 0.75, 1.0])
    ax_2d.set_yticklabels(["90°","67.5°", "45°", "22.5°", "0°"], fontsize=6)
    ax_2d.set_title("Hemispherical density\n(centre = 90°, edge = 0°)", va="bottom", pad=18)

    cbar = fig.colorbar(mesh, ax=ax_2d, fraction=0.046, pad=0.12, shrink=0.7)
    cbar.set_label(density_label, fontsize=8)

    # ------------------------------------------------------------------ #
    #  Suptitle + pole annotation                                          #
    # ------------------------------------------------------------------ #
    sup = f"{title}\n{pole_str}" if pole_str is not None else title
    plt.suptitle(sup, fontsize=11, fontweight="bold", y = .9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(f"{save_path}.svg", dpi=dpi, format="svg", bbox_inches="tight")
    if show:
        plt.show()

    return fig

def plot_hemisphere_summary_polar_2(
    az_deg,
    el_deg,
    weights=None,
    az_has_transverse=None,
    el_has_transverse=None,
    weights_transverse=None,
    has_az=None,
    az_bins=36,
    el_bins=18,
    figsize=(14, 5),
    save_path=None,
    title="Hemisphere Peak Directions",
    cmap="viridis",
    dpi=300,
    colorbar_fraction=False,
    show=False,
):
    """
    Hemisphere summary using standard spherical convention:
        Azimuth  : 0° – 360°  (0°=Right, 90°=Dorsal, 180°=Left, 270°=Ventral)
        Elevation: 0° –  90°  (0°=transverse plane, 90°=superior pole)

    Designed to accept the direct output of compute_az_el_fast():

        az, el, weights, az_has_transverse, el_has_transverse,
        weights_transverse, has_az = compute_az_el_fast(..., degrees=True)

        plot_hemisphere_summary_polar(
            az_deg             = az,
            el_deg             = el,
            weights            = weights,
            az_has_transverse  = az_has_transverse,
            el_has_transverse  = el_has_transverse,
            weights_transverse = weights_transverse,
            has_az             = has_az,
        )

    Panels:
        1. Azimuth polar bar     – az_has_transverse (pole-excluded)
        2. Elevation polar bar   – el_deg / weights (all valid, including pole)
        3. 2-D hemispherical map – az_has_transverse x el_has_transverse (pole-excluded)
                                   + pole peaks (no transverse component) placed at centre

    Parameters
    ----------
    az_deg             : all valid azimuths [0, 360°]
    el_deg             : all valid elevations [0, 90°]
    weights            : weights for az_deg / el_deg
    az_has_transverse  : pole-excluded azimuths — panels 1 & 3
    el_has_transverse  : pole-excluded elevations — panels 1 & 3
    weights_transverse : weights for the pole-excluded subset
    has_az             : (M,) bool mask over full valid set — used to extract
                         pole peaks (az_deg[~has_az], el_deg[~has_az]) for
                         placing them at the centre of the 2-D density panel
    az_bins            : number of azimuth bins
    el_bins            : number of elevation bins
    figsize            : (w, h). Portrait -> vertical stack; landscape -> side-by-side
    save_path          : SVG output path
    title              : figure suptitle
    cmap               : colormap name
    dpi                : save resolution
    colorbar_fraction  : normalise 2-D panel by fraction
    show               : call plt.show()

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)

    # Panels 1 & 3: use pole-excluded subset if provided, else fall back to full
    if az_has_transverse is not None:
        az_p13 = np.asarray(az_has_transverse, dtype=float)
        el_p13 = np.asarray(el_has_transverse, dtype=float)
        w_p13  = weights_transverse
        n_total    = len(az_deg)
        n_excluded = n_total - len(az_p13)
        pole_str   = (
            f"Near-pole excluded from azimuth: {n_excluded} / {n_total} peaks "
            f"({n_excluded / n_total * 100:.1f}%)"
        )
    else:
        az_p13   = az_deg
        el_p13   = el_deg
        w_p13    = weights
        pole_str = None

    # Extract pole peaks from full set using has_az mask
    # These have el ~ 90° and undefined azimuth — placed at centre of 2-D panel
    if has_az is not None:
        has_az    = np.asarray(has_az, dtype=bool)
        pole_w    = weights[~has_az] if weights is not None else None
        n_pole    = (~has_az).sum()
    else:
        pole_w = None
        n_pole = 0

    # ------------------------------------------------------------------ #
    #  Histograms                                                          #
    # ------------------------------------------------------------------ #
    az_edges = np.linspace(0,  360, az_bins + 1)
    el_edges = np.linspace(0,   90, el_bins + 1)

    # Panel 1: azimuth — pole-excluded
    az_counts, _ = np.histogram(az_p13, bins=az_edges, weights=w_p13)
    # Panel 2: elevation — all valid (pole peaks have meaningful el = 90°)
    el_counts, _ = np.histogram(el_deg, bins=el_edges, weights=weights)

    az_total = az_counts.sum()
    el_total = el_counts.sum()
    az_frac  = az_counts / az_total if az_total > 0 else az_counts
    el_frac  = el_counts / el_total if el_total > 0 else el_counts

    bin_centers_az = np.deg2rad(0.5 * (az_edges[:-1] + az_edges[1:]))
    bin_width_az   = np.deg2rad(np.diff(az_edges))

    bin_centers_el = np.deg2rad(0.5 * (el_edges[:-1] + el_edges[1:]))
    bin_width_el   = np.deg2rad(np.diff(el_edges))

    count_label = "Weighted count" if weights is not None else "Count"

    # ------------------------------------------------------------------ #
    #  2-D density                                                         #
    #  Elevation inverted: 90° pole -> centre (r=0), 0° equator -> edge   #
    #  Pole peaks (no transverse) are added uniformly into the innermost  #
    #  radius bin, distributed equally across all azimuth bins            #
    # ------------------------------------------------------------------ #
    el_inv  = 90.0 - el_p13
    r_edges = np.linspace(0, 90, el_bins + 1)

    # Main histogram — pole-excluded
    H2_stat, _, _ = np.histogram2d(
        az_p13, el_inv, bins=[az_edges, r_edges], weights=w_p13
    )

    # Optionally account for pole peaks *as a single bin mass*
    if n_pole > 0:
        pole_total = pole_w.sum() if pole_w is not None else float(n_pole)
        H2_stat[0, 0] += pole_total   # or keep separately if you prefer

    if colorbar_fraction:
        total = H2_stat.sum()
        H2_norm = H2_stat / total if total > 0 else H2_stat
        density_label = "Fraction"
    else:
        H2_norm = H2_stat
        density_label = count_label

    H2_plot = H2_norm.copy()

    if n_pole > 0:
        print(f"Adding pole mass: {pole_total} ({pole_str})")
        # distribute pole mass uniformly across azimuth bins
        pole_mass = (
            (pole_total / H2_stat.sum()) if colorbar_fraction else pole_total
        )
        H2_plot[:, 0] += pole_mass/az_bins


    # ------------------------------------------------------------------ #
    #  Figure layout                                                       #
    # ------------------------------------------------------------------ #
    w, h     = figsize
    vertical = h >= w

    fig = plt.figure(figsize=figsize)
    if vertical:
        ax_az = fig.add_subplot(311, projection="polar")
        ax_el = fig.add_subplot(312, projection="polar")
        ax_2d = fig.add_subplot(313, projection="polar")
    else:
        ax_az = fig.add_subplot(131, projection="polar")
        ax_el = fig.add_subplot(132, projection="polar")
        ax_2d = fig.add_subplot(133, projection="polar")

    cmap_obj = colormaps[cmap]

    # ------------------------------------------------------------------ #
    #  Panel 1 – Azimuth (pole-excluded)                                  #
    # ------------------------------------------------------------------ #
    norm_az   = Normalize(vmin=0, vmax=az_frac.max() if az_frac.max() > 0 else 1)
    colors_az = cmap_obj(norm_az(az_frac))

    ax_az.bar(
        bin_centers_az,
        az_frac,
        width=bin_width_az,
        bottom=0,
        color=colors_az,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )

    ax_az.axvline(np.deg2rad(90), color="crimson", linestyle="--",
                  linewidth=1.0, label="90° (Dorsal)")

    ax_az.set_theta_zero_location("E")   # 0° (Right) on the right
    ax_az.set_theta_direction(1)         # CCW: Right(0°)->Dorsal(90°)->Left(180°)->Ventral(270°)
    ax_az.set_rlabel_position(45)
    ax_az.yaxis.set_tick_params(labelsize=6)
    az_title = "Azimuth distribution\n(pole-excluded)" if az_has_transverse is not None \
               else "Azimuth distribution"
    ax_az.set_title(az_title, va="bottom", pad=18)
    ax_az.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7)

    # ------------------------------------------------------------------ #
    #  Panel 2 – Elevation (all valid peaks, including pole)              #
    # ------------------------------------------------------------------ #
    norm_el   = Normalize(vmin=0, vmax=el_frac.max() if el_frac.max() > 0 else 1)
    colors_el = cmap_obj(norm_el(el_frac))

    ax_el.bar(
        bin_centers_el,
        el_frac,
        width=bin_width_el,
        bottom=0,
        color=colors_el,
        edgecolor="white",
        linewidth=0.4,
        align="center",
    )

    ax_el.axvline(0,              color="crimson",   linestyle="--",
                  linewidth=1.0, label="0° (equator)")
    ax_el.axvline(np.deg2rad(90), color="steelblue", linestyle="--",
                  linewidth=1.0, label="90° (pole)")

    ax_el.set_thetamin(0)
    ax_el.set_thetamax(90)
    ax_el.set_theta_zero_location("W")
    ax_el.set_theta_direction(-1)
    ax_el.set_rlabel_position(90)
    ax_el.yaxis.set_tick_params(labelsize=6)
    el_title = "Elevation distribution\n(all valid peaks)" if az_has_transverse is not None \
               else "Elevation distribution"
    ax_el.set_title(el_title, va="bottom", pad=18)
    ax_el.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=7)

    # ------------------------------------------------------------------ #
    #  Panel 3 – 2-D hemispherical density                                #
    #  Centre = pole (90°), edge = equator (0°)                           #
    #  Pole peaks distributed uniformly in the innermost ring             #
    # ------------------------------------------------------------------ #
    az_rad_edges = np.deg2rad(az_edges)
    r_norm_edges = r_edges / 90.0

    Az, R = np.meshgrid(az_rad_edges, r_norm_edges)
    C     = H2_plot.T

    mesh = ax_2d.pcolormesh(Az, R, C, cmap=cmap_obj, shading="auto")

    ax_2d.set_theta_zero_location("E")
    ax_2d.set_theta_direction(1)
    ax_2d.set_rlim(0, 1)
    ax_2d.set_rticks([0,0.25, 0.5, 0.75, 1.0])
    ax_2d.set_yticklabels(["90°","67.5°", "45°", "22.5°", "0°"], fontsize=8)
    ax_2d.set_title(
        "Hemispherical density\n(centre = pole 90°, edge = equator 0°)",
        va="bottom", pad=18
    )

    cbar = fig.colorbar(mesh, ax=ax_2d, fraction=0.046, pad=0.12, shrink=0.7)
    cbar.set_label(density_label, fontsize=8)

    # ------------------------------------------------------------------ #
    #  Suptitle + pole annotation                                          #
    # ------------------------------------------------------------------ #
    sup = f"{title}\n{pole_str}" if pole_str is not None else title
    plt.suptitle(sup, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(f"{save_path}.svg", dpi=dpi, format="svg", bbox_inches="tight")
    if show:
        plt.show()

    return fig