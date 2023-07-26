import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.patches import Ellipse
from biopandas.pdb import PandasPdb
from mrcfile import open as mrc_open
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from collections import Counter
from typing import List, Dict, Tuple


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
    data = (data >= 0) * data
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
    new_df = df[df.groupby(["residue_number", "atom_name"])["alt_loc"].transform(min) == df["alt_loc"]]
    df_chosen = new_df[new_df["atom_name"] == atomic] if atomic is not None else new_df
    return df_chosen
    
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

    Return
    ----------
    grid_points_chosen: Dict[str, List[np.ndarray]]
        The generated grid points
    distances_to_center: Dict[str, List[np.ndarray]]
        The distances(radius) of grid points to center respectively
    """
    rads = np.round(np.arange(start_rad, max_rad + gap, gap), 2)
    unique_residue = np.unique(residue_names)
    grid_points_chosen = {name: [] for name in unique_residue}
    distances_to_center = {name: [] for name in unique_residue}
    Xs_tilde = {name: [] for name in unique_residue}
    
    for atomic_index in tqdm(range(residue_names.shape[0])):
        grid_points = {}
        name = residue_names[atomic_index]
        all_distances_to_center = []
        all_grid_points = []
        all_Xs_tilde = []
        for rad in rads:
            # The number of grid points we want to search depends on the dense of the ball
            num_points = int((rad**2 / rads[1]**2) * base_num_points)
            num_candidate = 1 if num_points == 0 else num_points if num_points < max_points else max_points
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
    x = np.linspace(0, nx - 1, nx) * grid_size[0] + origin[0] #- grid_size[0] / 2
    y = np.linspace(0, ny - 1, ny) * grid_size[1] + origin[1] #- grid_size[1] / 2
    z = np.linspace(0, nz - 1, nz) * grid_size[2] + origin[2] #- grid_size[2] / 2
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
    start_radius: float,
    max_radius: float,
    gap: float,
    labels: List,
    colors: List,
    subplots_num: int = None,
    separated: bool = False,
    root: str = None
) -> None:
    """
    Plot the density maps and the corresponding estimated density maps.

    Params
    ----------
    density_map: Dict
        Dictionary containing the density maps.
    estimated_density_maps: List
        List of dictionaries containing the estimated density maps.
    start_radius: float
        Minimum radius value.
    max_radius: float
        Maximum radius value.
    gap: float
        Gap between radius values.
    labels: List
        List of labels for the estimated density maps.
    colors: List
        List of colors for the estimated density maps.
    subplots_num: int = None
        Number of subplots to create.
    separated: bool = False
        Flag indicating whether to plot the density maps separately.
    root: str = None
        The root where you want to save the figure.

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
    x = np.arange(start_radius, max_radius + gap, gap)
    length = len(x)

    # Plot the density maps and estimated density maps separately
    if separated:
        curr_times = 0

        for name in density_map: 
            for times, elements in enumerate(zip(density_map[name], estimated_density_maps[name])):
                density, estimated_density = elements
                i = (times + curr_times) // 5
                j = (times + curr_times) % 5
                axes[i][j].plot(x, density[:length], linewidth=3, alpha=1, c="orange", label="map")
                axes[i][j].plot(x, estimated_density[:length], label=labels[0], linestyle="--", c=colors[0], linewidth=3)
                axes[i][j].text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=axes[i][j].transAxes)
            curr_times += (times + 1)
    else:
        # Plot all density maps and corresponding estimated density maps together
        for times, name in enumerate(density_map): 
            i = times // 5
            j = times % 5
            chosen_estimated_density_maps = list(map(lambda x: x[name], estimated_density_maps))
            for density in density_map[name]:
                axes[i][j].plot(x, density[:length], linewidth=3, alpha=1, c="orange", label="map")
                for idx, estimated_density in enumerate(chosen_estimated_density_maps):
                    axes[i][j].plot(x, estimated_density[:length], label=labels[idx], linestyle="--", c=colors[idx], linewidth=3)
            axes[i][j].text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=axes[i][j].transAxes)
    
    labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}

    # Create legend
    fig.legend(
    labels_handles.values(),
    labels_handles.keys(),
    loc = "upper center",
    bbox_to_anchor = (0.5, 0.1),
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
    if root:
        fig.savefig(root)

    return


def simulation_plot(data, zlabel, title, min_num=0, max_num=0.5):
    font = {'weight' : 'bold', 'size': 10}
    matplotlib.rc('font', **font)
    z = data
    nrows, ncols = z.shape
    x = np.linspace(min_num, max_num, ncols)
    y = np.linspace(min_num, max_num, nrows)
    x, y = np.meshgrid(x, y)

    mappable = plt.cm.ScalarMappable()
    mappable.set_array(z)

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='overlay')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=1, antialiased=False, shade=False,
                        cmap=mappable.cmap, norm=mappable.norm)    
    ax.set_xlabel("Contamination ratio of points")
    ax.set_ylabel("Contamination ratio of betas")
    ax.set_zlabel(zlabel, rotation=90)
    ax.set_box_aspect(aspect=None, zoom=0.98)
    ax.view_init(35, 260)
    fig.tight_layout()
    ax.set_title(title, y=1.0)
    fig.colorbar(mappable)
    plt.savefig("./figures/" + title)
    plt.show()


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals * nstd)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    # ellip = Ellipse(xy=pos, width=sigma0[name], height=sigma1[name], angle=0, **kwargs)
    ax.add_artist(ellip)
    
    return pos, width, height