import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
from mrcfile import open as mrc_open
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from itertools import filterfalse
from scipy.stats import norm
from collections import Counter
from typing import List, Dict, Tuple, Optional


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
        df_processed: pd.DataFrame,
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
    atom_points = np.column_stack(
        (df_processed.x_coord, df_processed.y_coord, df_processed.z_coord))
    rads = np.round(np.arange(start_rad, max_rad, gap), 2)
    grid_points_chosen = {name: [] for name in df_processed["residue_name"].unique()}
    distances_to_center = {name: [] for name in df_processed["residue_name"].unique()}
    # for atomic_index in tqdm(range(len(df_processed))):
    for atomic_index in tqdm(range(100)):
        grid_points = {}
        name = df_processed.iloc[atomic_index, :]["residue_name"]
        all_distances_to_center = []
        all_grid_points = []
        for rad in rads:
            # The number of grid points we want to search depends on the dense of the ball
            num_points = int((rad**2 / start_rad**2) * base_num_points)
            num_points = num_points if num_points <= max_points else max_points
            num_candidate = num_points
            grid_points_candidate = [None] * num_points
            grid_points = []
            atom_point = atom_points[atomic_index].reshape(-1)
            # Search grid points for `MAX_ITER` otherwise give up
            for k in range(max_iter):
                num_chosen = 0
                candidate_points = generate_points_on_sphere(
                    rad, num_candidate, *atom_point)
                # Check the cloest atomic of the grid point is the atomic we want
                for candidate_point in candidate_points:
                    distances = ((candidate_point - atom_points)
                                 ** 2).sum(axis=1)
                    if (distances[atomic_index] == distances.min()) and (num_chosen < num_points):
                        grid_points_candidate[num_chosen] = candidate_point.tolist(
                        )
                        num_chosen += 1
                if isinstance(grid_points_candidate[-1], type(None)):
                    # If the number of `None` is less than before than save it
                    if grid_points.count(None) > grid_points_candidate.count(None):
                        grid_points = grid_points_candidate
                    # Reset the list to find the grid points again
                    grid_points_candidate = [None] * num_points
                    # Increase the number of candidate points
                    num_candidate += 2
                else:
                    grid_points = grid_points_candidate
                    break
                if k == max_iter - 1:
                    grid_points = list(filterfalse(
                        lambda item: not item, grid_points))
                    print("There are some grid points that could not be found.")
            all_distances_to_center.extend([rad] * len(grid_points))
            all_grid_points.extend(grid_points)
        grid_points_chosen[name].append(np.array(all_grid_points))
        distances_to_center[name].append(np.array(all_distances_to_center))
    # grid_points_chosen = {name: np.array(points) for name, points in grid_points_chosen.items()}
    return grid_points_chosen, distances_to_center

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

def estimate_density(
        grid_points_chosen: Dict[str, List[np.ndarray]], 
        distances_to_center: Dict[str, List[np.ndarray]], 
        interp_func: callable, 
        bayes_beta: Dict[str, float], 
        estimated_A_ij_tilde: Dict[str, np.ndarray],
) -> Tuple[
        Dict[str, List[List[float]]], 
        Dict[str, List[List[float]]], 
        Dict[str, List[List[float]]]
]:
    """
    Given the interpolation function from the density map, construct the 
    densities on the atomic model corresponding to their coordinates. 
    
    Params
    ----------
    grid_points_chosen: 
        The dictionary of grid points chosen for each radius.
    distances_to_center: 
        The dictionary of distances to the center for each radius.
    interp_func: 
        The callable interpolation function.
    bayes_beta: 
        The dictionary of Bayesian beta values for each radius.
    estimated_A_ij_tilde: 
        The dictionary of estimated values of A_ij_tilde for each radius.
    estimated_means_of_error: 
        The dictionary of estimated means of error for each radius.
    
    Return
    ----------
    radius_density: 
        The dictionary of mean densities for each grid point radius.
    estimated_radius_density: 
        The dictionary of estimated mean densities for each grid point radius.
    qscore_radius_density: 
        The dictionary of qscore mean densities for each grid point radius.
    """
    radius_density = {key: [] for key in grid_points_chosen.keys()}
    estimated_radius_density = {key: [] for key in grid_points_chosen.keys()}
    qscore_radius_density = {key: [] for key in grid_points_chosen.keys()}
    for key, value in grid_points_chosen.items():
        for i in range(len(value)):
            counter = Counter(distances_to_center[key][i])
            grid_points = grid_points_chosen[key][i]
            A_ij_tilde = estimated_A_ij_tilde[key][i]
            mean_densities = []
            estimated_mean_densities = []
            qscore_mean_densities = []
            start = 0
            densities = interp_func(grid_points)
            m = densities.mean()
            std = densities.std()
            A = m + 10 * std
            B = m - std
            for distance, num in counter.items():
                mean_densities.append(densities[start: start + num].mean())
                X_k = - 1 / 2 * distance**2
                estimated_mean_densities.append(np.exp(A_ij_tilde + X_k * bayes_beta[key]))
                qscore_mean_densities.append((2*np.pi*0.6**2)**(-3/2)*np.exp(-1/(2*0.6**2)*distance**2) * A + B)
                start += num
            radius_density[key].append(mean_densities)
            estimated_radius_density[key].append(estimated_mean_densities)
            qscore_radius_density[key].append(qscore_mean_densities)
    return radius_density, estimated_radius_density, qscore_radius_density

def plot_density(
    radius_density: Dict[str, np.ndarray], 
    estimated_radius_density: Dict[str, np.ndarray],
    qscore_radius_density: Dict[str, np.ndarray],
    A_B: Tuple[float, float],
    amino_acid: Optional[str] = None, 
    indexes: Optional[List[int]] = None, 
    start_rad: float = 0.01,
    max_rad: float = 0.8,
    gap: float = 0.01,
    compared: bool = False,
    estimated: bool = True
) -> None:
    """
    Plots cryo-EM map density, estimated Gaussian in QEB, Gaussian in qscore. 

    Parameters:
    -----------
    radius_density: dict[str, np.ndarray]
        The dictionary of radius densities.
    estimated_radius_density: dict[str, np.ndarray]
        The dictionary of estimated radius densities in QEB.
    qscore_radius_density: dict[str, np.ndarray]
        The dictionary of qscore radius densities in qscore.
    A_B: tuple[float, float]
        The tuple of coefficients.
    amino_acid: Optional[str] (default=None)
        An amino acid to plot. If None, all amino acids are plotted.
    indexes: Optional[List[int]] (default=None)
        Indexes of radius densities to plot.
    start_rad: float (default=0.01)
            Minimum radius for generating grid points. 
    max_rad: float (default=0.8)
        Maximum radius for generating grid points. 
    gap: float (default=0.01)
        Gap between radii for generating grid points. 
    compared: bool (default=False)
        Whether to compare the estimated Gaussian with the Gaussian in qscore.
    estimated: bool (default=True)
        Whether to plot the estimated Gaussian.

    Return
    ----------
    None
    """
    x_axis = np.arange(start_rad, max_rad, gap)
    # If not choose specific amino acid, then plot all types
    radius_density = {amino_acid: radius_density[amino_acid]} if amino_acid is not None else radius_density
    estimated_radius_density = {amino_acid: estimated_radius_density[amino_acid]} if amino_acid is not None else estimated_radius_density
    qscore_radius_density = {amino_acid: qscore_radius_density[amino_acid]} if amino_acid is not None else qscore_radius_density
    for key, mean_densities in radius_density.items():
        mean_densities = np.array(mean_densities)[indexes] if indexes is not None else mean_densities
        estimated_mean_densities = np.array(estimated_radius_density[key]) if indexes is not None else estimated_radius_density[key]
        qscore_mean_densities = np.array(qscore_radius_density[key]) if indexes is not None else qscore_radius_density[key]
        indexes = indexes if indexes is not None else list(range(len(mean_densities)))
        for index, mean_density in zip(indexes, mean_densities):
            plt.plot(x_axis, mean_density, label="cryo-EM map density")
            if estimated:
                plt.plot(x_axis, estimated_mean_densities[index], "--", label="estimaed gaussian")
                plt.plot(x_axis, (2*np.pi*0.6**2)**(-3/2)*np.exp(-1/(2*0.6**2)*x_axis**2)*A_B[0] + A_B[1], label="gaussian in qscore?")
                plt.plot(x_axis, qscore_mean_densities[index], label="gaussian in qscore?")
            plt.xlabel("Distance to the Center (Anstrom)")
            plt.ylabel("Density")
            plt.title(key)
            if not compared:
                plt.legend()
                plt.show()
        indexes = None
        plt.show()