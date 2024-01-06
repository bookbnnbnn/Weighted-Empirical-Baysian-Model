import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from biopandas.pdb import PandasPdb
from mrcfile import open as mrc_open
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from collections import Counter
from typing import Callable, Dict, Any, Optional, Tuple, List


def read_map(root: str) -> Tuple[np.ndarray]:
    """
    Read the density map from `map` file

    Parameters
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

def read_pdb(root: str, atoms: List = None) -> pd.DataFrame:
    """
    Read atom model from `pdb` file

    Parameters
    ----------
    root: str
        The root of `map` data 
    atom: str
        The chosen atom that we only consider

    Return
    ----------
    pd.DataFrame
        The dataframe contained data that we only consider
    """

    ppdb = PandasPdb().read_pdb(root)
    df = ppdb.df['ATOM']
    new_df = df[df.groupby(["residue_number", "atom_name"])["alt_loc"].transform(min) == df["alt_loc"]]
    df_chosen = new_df[new_df["atom_name"].isin(atoms)] if atoms is not None else new_df
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

    Parameters
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

    Parameters
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
    atom_coordinates = {name: [] for name in unique_residue}
    
    for atomic_index in tqdm(range(residue_names.shape[0])):
        grid_points = {}
        name = residue_names[atomic_index]
        all_distances_to_center = []
        all_grid_points = []
        all_Xs_tilde = []
        for rad in rads:
            # The number of grid points we want to search depends on the dense of the ball
            num_points = int((rad**2 / rads[1]**2) * base_num_points)
            num_candidate = base_num_points if num_points == 0 else num_points if num_points < max_points else max_points
            atom_point = atom_points[atomic_index].reshape(-1)
            # Search grid points for `MAX_ITER` otherwise give up

            grid_points = generate_points_on_sphere(rad, num_candidate, *atom_point)
            all_distances_to_center.extend([rad] * len(grid_points))
            all_grid_points.extend(grid_points.tolist())
            all_Xs_tilde.extend([[1, - 1 / 2 * rad ** 2]] * len(grid_points))
        grid_points_chosen[name].append(np.array(all_grid_points))
        distances_to_center[name].append(np.array(all_distances_to_center))
        Xs_tilde[name].append(np.array(all_Xs_tilde))
        atom_coordinates[name].append(atom_point)
    
    for name in unique_residue:
        grid_points_chosen[name] = np.array(grid_points_chosen[name])
        distances_to_center[name] = np.array(distances_to_center[name])
        Xs_tilde[name] = np.array(Xs_tilde[name])
        atom_coordinates[name] = np.array(atom_coordinates[name])
    
    return grid_points_chosen, distances_to_center, Xs_tilde, atom_coordinates


def interpolator(
        data: np.ndarray, 
        grid_size: np.ndarray, 
        origin: np.ndarray
) -> RegularGridInterpolator:
    """
    Interpolates 3D data on a regular grid.

    Parameters
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

    Parameters
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

def simulation_plot(data: np.ndarray, zlabel: str, title: str, min_num: float = 0, max_num: float = 0.4, save: bool = False) -> None:
    """
    Plot a 3D surface plot for simulation data.

    Parameters
    ----------
    data : np.ndarray
        2D array containing the simulation data.
    zlabel : str
        Label for the z-axis.
    title : str
        Title for the plot.
    min_num : float, optional
        Minimum value for the x and y axis, by default 0.
    max_num : float, optional
        Maximum value for the x and y axis, by default 0.5.

    Returns
    -------
    None
    """

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
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
    #                     linewidth=1, antialiased=False, shade=False,
    #                     cmap=mappable.cmap, norm=mappable.norm)    
    
    surf = ax.plot_surface(x, y, z, cmap="plasma", linewidth=0, antialiased=False, shade=False, alpha=0.5)    
    ax.set_xlabel("Contamination ratio of points")
    ax.set_ylabel("Contamination ratio of betas")
    ax.set_zlabel(zlabel, rotation=90)
    ax.set_box_aspect(aspect=None, zoom=0.95)
    ax.view_init(35, 260)
    fig.tight_layout()
    ax.set_title(title, y=1.0)
    fig.colorbar(surf, shrink=0.5, aspect=20, pad=-0.06)
    plt.xticks(np.arange(min_num, max_num + 0.1, 0.1))
    if save:
        plt.savefig("./figures/" + title, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_cov_ellipse(cov: np.ndarray, pos: np.ndarray, nstd: float = 2, ax: plt.Axes = None, **kwargs) -> Tuple[np.ndarray, float, float]:
    """
    Plot an ellipse based on the specified covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        The 2x2 covariance matrix to base the ellipse on.
    pos : np.ndarray
        The location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
    nstd : float, optional
        The radius of the ellipse in numbers of standard deviations, by default 2.
    ax : plt.Axes, optional
        The axis that the ellipse will be plotted on, by default None.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        A tuple containing the position, width, and height of the ellipse.
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
    ax.add_artist(ellip)
    
    return pos, width, height


def sub_plots(
        plot_func: Callable[[plt.Axes, str, Dict[str, Any]], None],
        data: Dict[str, Any],
        x_label: str,
        y_label: str,
        fontsize: int = 15,
        plot_dim: Tuple[int, int] = (2, 2),
        figsize: Tuple[float, float] = (16, 12),
        sharex: bool = False,
        sharey: bool = False,
        legend: Optional[Dict[str, Any]] = None,
        root: Optional[str] = None,
        **kwargs
) -> None:
    """
    Create subplots with custom plotting function.

    Parameters
    ----------
    plot_func : Callable[[plt.Axes, str, Dict[str, Any]], None]
        The plotting function to be applied on each subplot.
        It should take three arguments: the subplot axis, the name of the data,
        and the data itself.
    data : Dict[str, Any]
        Dictionary containing the data to be plotted. The keys represent the name
        of the data, and the values represent the actual data.
    x_label : str
        The label for the x-axis of each subplot.
    y_label : str
        The label for the y-axis of each subplot.
    fontsize : int, optional
        The fontsize for the plot labels, by default 15.
    plot_dim : Tuple[int, int], optional
        The dimensions of the subplot grid, by default (2, 2).
    figsize : Tuple[float, float], optional
        The size of the entire figure in inches, by default (16, 12).
    sharex : bool, optional
        Whether to share the x-axis among subplots, by default False.
    sharey : bool, optional
        Whether to share the y-axis among subplots, by default False.
    legend : Dict[str, Any], optional
        A dictionary containing optional legend settings, by default None.
    root : str, optional
        The root path where the figure will be saved, by default None.
    **kwargs
        Additional keyword arguments to be passed to the custom_plot function.

    Returns
    -------
    None
    """

    font = {'weight' : 'bold', 'size':fontsize}
    matplotlib.rc('font', **font)
    times = 0
    fig, axes = plt.subplots(*plot_dim, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
    for name in data:
        i = times // plot_dim[1]
        j = times % plot_dim[1]
        plot_func(axes[i][j], name, data, **kwargs)
        times += 1
    fig.tight_layout(rect=[0.04, 0.09, 1, 0.95])
    fig.supxlabel(x_label, fontsize=35)
    fig.supylabel(y_label, fontsize=35)
    if legend is not None:
        if "handles" not in legend and "labels" not in legend:
            labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}
            fig.legend(handles=labels_handles.values(), labels=labels_handles.keys(), **legend)
        else:
            fig.legend(**legend)
    if root is not None:
        fig.savefig(root)


def distance_hist(
    ax: plt.Axes,
    name: str,
    outliers: Dict[str, np.ndarray],
    statistic_distances: Dict[str, np.ndarray],
    margin: float
) -> None:
    """
    Plot the histogram of distances with respect to normal and outlier data points.

    Parameters
    ----------
    ax : plt.Axes
        The axis on which the histogram will be plotted.
    name : str
        The name of the data.
    outliers : Dict[str, np.ndarray]
        Dictionary containing outlier indices for each data.
    statistic_distances : Dict[str, np.ndarray]
        Dictionary containing the distances for each data.
    margin : float
        The critical value for identifying outliers.

    Returns
    -------
    None
    """

    normal_index = ~np.isin(np.arange(0, len(statistic_distances[name])), outliers[name])
    log_distance = np.log(np.array(statistic_distances[name]))
    sns.histplot(log_distance[normal_index], bins=np.arange(min(log_distance) - 0.3, max(log_distance) + 0.3, 0.2), label=f"Normal ({sum(normal_index)})", ax=ax)
    sns.histplot(log_distance[outliers[name]], bins=np.arange(min(log_distance) - 0.3, max(log_distance) + 0.3, 0.2), label=f"Outliers ({len(outliers[name])})", ax=ax, color="#ff7f0e")
    ax.axvline(x=np.log(margin), color = 'red', label = 'critical value', linestyle = '--')
    ax.text(0.9, 0.6, name, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize = 25)
    ax.set(ylabel='')
    ax.legend(loc="upper right")


def confidence_region_plot(
    ax: plt.Axes,
    name: str,
    outliers: Dict[str, np.ndarray],
    statistic_distances: Dict[str, np.ndarray],
    betas_WEB: Dict[str, np.ndarray],
    sigmas: Dict[str, np.ndarray],
    mus_mle: Dict[str, np.ndarray],
    margin: float
) -> None:
    """
    Plot the confidence region for weighted empirical Bayesian analysis.

    Parameters
    ----------
    ax : plt.Axes
        The axis on which the confidence region will be plotted.
    name : str
        The name of the data.
    outliers : Dict[str, np.ndarray]
        Dictionary containing outlier indices for each data.
    statistic_distances : Dict[str, np.ndarray]
        Dictionary containing the distances for each data.
    betas_WEB : Dict[str, np.ndarray]
        Dictionary containing the betas for each data.
    sigmas : Dict[str, np.ndarray]
        Dictionary containing the covariance matrices for each data.
    mus_mle : Dict[str, np.ndarray]
        Dictionary containing the means for each data.
    margin : float
        The critical value for identifying outliers.

    Returns
    -------
    None
    """

    normal_index = ~np.isin(np.arange(0, len(statistic_distances[name])), outliers[name])
    ax.scatter(betas_WEB[name][normal_index][:, 0], betas_WEB[name][normal_index][:, 1], color='blue')
    ax.scatter(betas_WEB[name][outliers[name]][:, 0], betas_WEB[name][outliers[name]][:, 1], color='#ff7f0e')
    pos, width, height = plot_cov_ellipse(sigmas[name], mus_mle[name], nstd=margin, ax=ax, alpha=0.5, color='green')
    ax.text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize = 25)


def outliers_density_plot(
    ax: plt.Axes,
    name: str,
    densities_data: Dict[str, List[np.ndarray]],
    densities_mle: Dict[str, np.ndarray],
    densities_outliers: Dict[str, List[np.ndarray]],
    start_radius: float,
    max_radius: float,
    gap: float
) -> None:
    """
    Plot the density maps for weighted empirical Bayesian analysis.

    Parameters
    ----------
    ax : plt.Axes
        The axis on which the density maps will be plotted.
    name : str
        The name of the data.
    densities_data : Dict[str, List[np.ndarray]]
        Dictionary containing the density maps for each data.
    densities_mle : Dict[str, np.ndarray]
        Dictionary containing the maximum likelihood estimated density maps for each data.
    densities_outliers : Dict[str, List[np.ndarray]]
        Dictionary containing the density maps of outliers for each data.
    start_radius : float
        The minimum radius value for the plot.
    max_radius : float
        The maximum radius value for the plot.
    gap : float
        The gap between radius values for the plot.

    Returns
    -------
    None
    """

    x = np.arange(start_radius, max_radius + gap, gap)
    for density in densities_data[name]:
        ax.plot(x, density[:len(x)], alpha=0.3, c="orange", label="map")
    if name in densities_outliers:
        for density_outliers in densities_outliers[name]:
            ax.plot(x, density_outliers[:len(x)], alpha=1, c="green", label="Outliers")
    ax.plot(x, densities_mle[name][:len(x)], label="WEB MLE-E", linestyle="--", c="blue", linewidth=3)
    ax.text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)


def density_plot(
    ax: plt.Axes,
    name: str,
    densities_data: Dict[str, List[np.ndarray]],
    start_radius: float,
    max_radius: float,
    gap: float,
    q_scores: Dict[str, np.ndarray] = None
) -> None:
    """
    Plot the density maps for weighted empirical Bayesian analysis.

    Parameters
    ----------
    ax : plt.Axes
        The axis on which the density maps will be plotted.
    name : str
        The name of the data.
    densities_data : Dict[str, List[np.ndarray]]
        Dictionary containing the density maps for each data.
    start_radius : float
        The minimum radius value for the plot.
    max_radius : float
        The maximum radius value for the plot.
    gap : float
        The gap between radius values for the plot.

    Returns
    -------
    None
    """

    x = np.arange(start_radius, max_radius + gap, gap)
    for density in densities_data[name]:
        ax.plot(x, density[:len(x)], linewidth=3, alpha=0.3, label="map", c="orange")
    ax.text(0.8, 0.8, f"{name} \n ({len(densities_data[name])})", horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    ax.axvline(x=1, color = 'red', label = 'unit radius', linestyle = '--')
    if q_scores is not None:
        densities = densities_data[name]
        q_score = q_scores[name]
        density_poorest = densities[np.where(q_score == np.min(q_score))].reshape(-1)
        density_best = densities[np.where(q_score == np.max(q_score))].reshape(-1)
        ax.plot(x, density_poorest[:len(x)], linewidth=3, label="map", c="green", alpha=1)
        ax.plot(x, density_best[:len(x)], linewidth=3, label="map", c="blue", alpha=1)
        ax.text(x[int(len(x) / 4)], density_poorest[int(len(x) / 2)], f"Q = {round(np.min(q_score), 4)}", horizontalalignment='left', verticalalignment='top', color="green", fontsize=20)
        ax.text(x[int(len(x) / 4)], density_best[int(len(x) / 4)], f"Q = {round(np.max(q_score), 4)}", horizontalalignment='left', verticalalignment='bottom', color="blue", fontsize=20)
            

    
def fitted_density_plot(
    ax: plt.Axes,
    name: str,
    densities_data: Dict[str, List[np.ndarray]],
    estimated_densities: List[Dict[str, np.ndarray]],
    start_radius: float,
    max_radius: float,
    gap: float, 
    labels: List[str],
    colors: List[str],
    separated: Optional[bool] = False
):
    """
    Plot the fitted density distributions along with the original density.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object for plotting.
    name : str
        Name of the data.
    densities_data : Dict[str, List[np.ndarray]]
        A dictionary containing density data for different names.
    estimated_densities : List[Dict[str, np.ndarray]]
        List of dictionaries containing estimated densities for different estimators.
    start_radius : float
        The starting radius for the plot.
    max_radius : float
        The maximum radius for the plot.
    gap : float
        The gap between data points in the plot.
    labels : List[str]
        List of labels for each estimator.
    colors : List[str]
        List of colors for each estimator.
    separated : bool, optional
        Whether the data densities are separated, by default False.

    Returns
    -------
    None
    """

    x = np.arange(start_radius, max_radius + gap, gap)
    chosen_estimated_densities = list(map(lambda x: x[name], estimated_densities))
    for density in densities_data[name]:
        ax.plot(x, density[:len(x)], linewidth=3, alpha=1, c="orange", label="map")
        for idx, estimated_density in enumerate(chosen_estimated_densities):
            ax.plot(x, estimated_density[:len(x)], label=labels[idx], linestyle="--", c=colors[idx], linewidth=3)
    ax.text(0.9, 0.5, name, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)