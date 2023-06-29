import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from biopandas.pdb import PandasPdb
from mrcfile import open as mrc_open
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from itertools import filterfalse
from scipy.stats import norm
from collections import Counter
from typing import List, Dict, Tuple, Optional
from numba import njit
import random


def read_map(root: str) -> Tuple[np.ndarray]:
    """
    Read the density map from `map` file

    Params
    ----------
    root: str
        The root of `map` data 

    Return
    ----------
    data: np.ndarray
        The density in each coordinate from `map` data
    grid_size: np.ndarray
        The grid size of `map` data
    origin: np.ndarray
        The origin coordinate of `map` data
    """

    with mrc_open(root, permissive=True) as mrc:
        header = mrc.header
        data = mrc.data
    # Data need to transform since the shape of `map` file does not fit to `pdb` file
    data = np.einsum('zyx->xyz', data)
    data = (data > 0) * data
    origin = np.array(header["origin"].item())
    grid_size = np.array(header["cella"].item()) / data.shape
    return data, grid_size, origin

def read_pdb(root: str, atomic: str = None) -> pd.DataFrame:
    """
    Read atomic model from `pdb` file

    Params
    ----------
    root: str
        The root of `map` data 
    atomic: str
        The chosen atomic that we only consider

    Return
    ----------
    pd.DataFrame
        The dataframe contained data that we only consider
    """

    ppdb = PandasPdb().read_pdb(root)
    df = ppdb.df['ATOM']
    if atomic is not None:
        df_chosen = df[df["atom_name"] == atomic] if atomic is not None else df
        df_well = df_chosen[df_chosen["occupancy"] == 1]
        df_defected = df_chosen.drop(df_well.index)
        gb_defected = df_defected.groupby(["chain_id", "residue_name", "residue_number"])
        new_index = [np.mean(value) for value in gb_defected.groups.values()]
        new_df = gb_defected.mean().reset_index()
        new_df.index = new_index
        df_processed = pd.concat([df_well, new_df])
        return df_processed
    else:
        return df
    
def generate_points_on_sphere(
        radius: float = 1, 
        num_points: int = 8, 
        center_x: float = 0, 
        center_y: float = 0, 
        center_z: float = 0
    ) -> np.ndarray:
    """
    Construct the grid points, whose denses are equal on shpere for the specific radius from the center

    Params
    ----------
    radius: float
        The radius for the specific radius from the center
    num_points: int
        The number of points that we want to make on the sphere
    center_x: float
        The number of center in x coordinate
    center_y: float
        The number of center in y coordinate
    center_z: float
        The number of center in z coordinate

    Return
    ----------
    points: np.ndarray
        The grid points that we construct
    """
    n = np.arange(1, num_points+1)
    phi = (np.sqrt(5) - 1) / 2
    z = (2*n - 1) / num_points - 1
    x = np.sqrt(1 - z**2) * np.cos(2 * np.pi * n * phi)
    y = np.sqrt(1 - z**2) * np.sin(2 * np.pi * n * phi)
    points = np.column_stack((center_x + radius*x, center_y + radius*y, center_z + radius*z))
    return points 


def generate_grid_points(
        atom_points,
        residue_names,
        start_rad: float = 0.01,
        max_rad: float = 1.5,
        gap: float= 0.01,
        max_points: int = 8,
        base_num_points: int = 4,
        max_iter: int = 30
) -> Tuple[Dict[str, List[np.ndarray]]]:
    """
    Generate grid points from each center that we chose from atomic model

    Params
    ----------
    df_processed: pd.DataFrame
        The processed dataframe 
    start_rad: float (default=0.01)
            Minimum radius for generating grid points. 
    max_rad: float (default=0.8)
        Maximum radius for generating grid points. 
    gap: float (default=0.01)
        Gap between radii for generating grid points. 
    max_points: int (default=8)
        Maximum number of points to generate at each radius. 
    base_num_points: int (default=4)
        Number of points to generate at the minimum radius. 
    max_iter: int (default=30)
        The maximal iteration that we want to find the grid points

    Return
    ----------
    grid_points_chosen: Dict[str, List[np.ndarray]]
        The generated grid points
    distances_to_center: Dict[str, List[np.ndarray]]
        The distances(radius) of grid points to center respectively
    """
    rads = np.round(np.arange(start_rad, max_rad, gap), 2)
    unique_residue = np.unique(residue_names)
    grid_points_chosen = {name: [] for name in unique_residue}
    distances_to_center = {name: [] for name in unique_residue}
    Xs_tilde = {name: [] for name in unique_residue}
    
    for atomic_index in tqdm(range(residue_names.shape[0])):
    # for atomic_index in tqdm(range(2000)):
        grid_points = {}
        name = residue_names[atomic_index]
        all_distances_to_center = []
        all_grid_points = []
        all_Xs_tilde = []
        for rad in rads:
            # The number of grid points we want to search depends on the dense of the ball
            num_points = int((rad**2 / start_rad**2) * base_num_points)
            num_candidate = num_points if num_points < max_points else max_points
            atom_point = atom_points[atomic_index].reshape(-1)
            # Search grid points for `MAX_ITER` otherwise give up

            grid_points = generate_points_on_sphere(rad, num_candidate, *atom_point)
            all_distances_to_center.extend([rad] * len(grid_points))
            all_grid_points.extend(grid_points.tolist())
            all_Xs_tilde.extend([[1, - 1 / 2 * rad ** 2]] * len(grid_points))
        grid_points_chosen[name].append(np.array(all_grid_points))
        distances_to_center[name].append(np.array(all_distances_to_center))
        Xs_tilde[name].append(np.array(all_Xs_tilde))
    
    return grid_points_chosen, distances_to_center, Xs_tilde


def interpolator(
        data: np.ndarray, 
        grid_size: np.ndarray, 
        origin: np.ndarray
) -> RegularGridInterpolator:
    """
    Interpolates 3D data on a regular grid.

    Params
    ----------
    data: np.ndarray
        3D array of data to be interpolated.
    grid_size: np.ndarray
        Array of size 3 representing the grid size in each dimension.
    origin: np.ndarray
        Array of size 3 representing the origin point of the grid.

    Return
    ----------
    RegularGridInterpolator: Interpolation function.

    """
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
    x = np.linspace(0, nx - 1, nx) * grid_size[0] + origin[0]
    y = np.linspace(0, ny - 1, ny) * grid_size[1] + origin[1]
    z = np.linspace(0, nz - 1, nz) * grid_size[2] + origin[2]
    interp_func = RegularGridInterpolator((x, y, z), data)
    return interp_func


def density_mean(
    data_list: List[Dict[str, np.ndarray]],
    distances_to_center: Dict[str, List[np.ndarray]]
) -> Tuple[Dict[str, List[List[float]]]]:
    """
    Calculates the mean densities for each radius and returns the results.

    Params
    ----------
    data_list: List[Dict[str, np.ndarray]]
        List of dictionaries containing the data.
    distances_to_center: Dict[str, List[np.ndarray]]
        Dictionary containing the distances to center.

    Return
    ----------
    Tuple[Dict[str, List[List[float]]]]
        Tuple containing a dictionary of mean densities for each radius.

    """
    radius_densities = []
    for data in data_list:
        radius_density = {key: [] for key in distances_to_center.keys()}

        for key, value in distances_to_center.items():

            for i in range(len(value)):
                mean_densities = []
                counter = Counter(distances_to_center[key][i])
                densities = data[key][i]
                start = 0

                for distance, num in counter.items():
                    densities_chosen = densities[start: start + num]

                    # Remove NaN values from densities if present
                    if np.isnan(densities_chosen).any():
                        densities_chosen = densities_chosen[~np.isnan(densities_chosen)]

                    mean_densities.append(densities_chosen.mean())
                    start += num
                radius_density[key].append(np.array(mean_densities))
            radius_density[key] = np.array(radius_density[key])
        radius_densities.append(radius_density)

    return radius_densities

def plot_density(
    density_map: Dict,
    estimated_density_maps: List,
    max_radius: float,
    gap: float,
    labels: List,
    colors: List,
    subplots_num: int = None,
    separated: bool = False,
    save: bool = False
) -> None:
    """
    Plot the density maps and the corresponding estimated density maps.

    Params
    ----------
    density_map: Dict
        Dictionary containing the density maps.
    estimated_density_maps: List
        List of dictionaries containing the estimated density maps.
    max_radius: float
        Maximum radius value.
    gap: float
        Gap between radius values.
    labels: List
        List of labels for the estimated density maps.
    colors: List
        List of colors for the estimated density maps.
    subplots_num: int = None
        Number of subplots to create (default: None).
    separated: bool = False
        Flag indicating whether to plot the density maps separately (default: False).
    save: bool = False
        Flag indicating whether to save the figure (default: False).

    Returns
    -------
        None
    """

    # Set font size and weight
    font = {'weight' : 'bold', 'size': 25}
    matplotlib.rc('font', **font)

    # Determine the number of subplots
    subplots_num = len(density_map) if not subplots_num else subplots_num
    nums = int(subplots_num / 5) + 1 if subplots_num % 5 != 0 else int(subplots_num / 5)

    # Create the subplots
    fig, axes = plt.subplots(nums, 5, figsize=(25, nums * 4), sharex=True, sharey=True, squeeze=False)
    x = np.arange(0, max_radius - gap, gap)
    length = len(x)

    # Plot the density maps and estimated density maps separately
    if separated:
        curr_times = 0

        for name in density_map: 
            chosen_estimated_density_maps = list(map(lambda x: x[name], estimated_density_maps))
            for times, elements in enumerate(zip(density_map[name], chosen_estimated_density_maps[0])):
                density, estimated_density = elements
                i = (times + curr_times) // 5
                j = (times + curr_times) % 5
                axes[i][j].plot(x, density[:length], linewidth=1, alpha=0.5, c="orange", label="map")
                axes[i][j].plot(x, estimated_density[:length], label=labels[0], linestyle="--", c=colors[0], linewidth=1.5)
                axes[i][j].text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=axes[i][j].transAxes)
            curr_times += (times + 1)
    else:
        # Plot all density maps and corresponding estimated density maps together
        for times, name in enumerate(density_map): 
            i = times // 5
            j = times % 5
            chosen_estimated_density_maps = list(map(lambda x: x[name], estimated_density_maps))
            for density in density_map[name]:
                axes[i][j].plot(x, density[:length], linewidth=1, alpha=0.5, c="orange", label="map")
                for idx, estimated_density in enumerate(chosen_estimated_density_maps):
                    axes[i][j].plot(x, estimated_density[:length], label=labels[idx], linestyle="--", c=colors[idx], linewidth=1.5)
            axes[i][j].text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=axes[i][j].transAxes)
    
    labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}

    # Create legend
    fig.legend(
    labels_handles.values(),
    labels_handles.keys(),
    loc = "upper center",
    bbox_to_anchor = (0.25, 0.1),
    bbox_transform = plt.gcf().transFigure,
    ncol=len(labels) + 1
    )

    # Set x-axis and y-axis labels
    fig.supxlabel('Radius')
    fig.supylabel('Voxel Value')

    # Adjust plot layout
    plt.tight_layout()
    fig.tight_layout(rect=(0.025, 0.03, 1, 1))

    # Save the figure if specified
    if save:
        fig.savefig("../figures/densities_compared.png")

    return