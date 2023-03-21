import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
from mrcfile import open as mrc_open
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from itertools import filterfalse
from typing import List, Dict, Tuple, Union


def read_map(root: str) -> Tuple[np.ndarray]:

    with mrc_open(root, permissive=True) as mrc:
        header = mrc.header
        data = mrc.data
    # Data need to transform since the shape of `map` file does not fit
    data = np.einsum('zyx->xyz', data)
    data = (data > 0) * data
    origin = np.array(header["origin"].item())
    grid_size = np.array(header["cella"].item()) / data.shape
    return data, grid_size, origin

def read_pdb(
        root: str, 
        atomic: str = None
        ) -> pd.DataFrame:
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

# my method
def generate_points_on_sphere(
        radius: Union[int, float] = 1, 
        num_points: int = 8, 
        center_x: Union[int, float] = 0, 
        center_y: Union[int, float] = 0, 
        center_z: Union[int, float] = 0
        ) -> np.ndarray:
    n = np.arange(1, num_points+1)
    phi = (np.sqrt(5) - 1) / 2
    z = (2*n - 1) / num_points - 1
    x = np.sqrt(1 - z**2) * np.cos(2 * np.pi * n * phi)
    y = np.sqrt(1 - z**2) * np.sin(2 * np.pi * n * phi)
    points = np.column_stack((center_x + radius*x, center_y + radius*y, center_z + radius*z))
    return points 

def generate_grid_points(
        df_processed,
        start_rad=0.01,
        max_rad=1.5,
        gap=0.01,
        max_points=8,
        base_num_points=4,
        max_iter=30
):
    atom_points = np.column_stack(
        (df_processed.x_coord, df_processed.y_coord, df_processed.z_coord))
    rads = np.round(np.arange(start_rad, max_rad, gap), 2)
    all_grid_points = {}
    distances_to_center = {}
    # for atomic_index in tqdm(range(len(df_processed))):
    for atomic_index in tqdm(range(10)):
        grid_points = {}
        name = tuple(df_processed.iloc[atomic_index, :][[
                     "residue_name", "chain_id", "residue_number"]])
        distances_to_center[name] = []
        all_grid_points[name] = []
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
            distances_to_center[name].extend([rad] * len(grid_points))
            all_grid_points[name].extend(grid_points)
    return all_grid_points, distances_to_center

def interpolator(
        data: np.ndarray, 
        grid_size: np.ndarray, 
        origin: np.ndarray
        ) -> RegularGridInterpolator:
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
    x = np.linspace(0, nx - 1, nx) * grid_size[0] + origin[0]
    y = np.linspace(0, ny - 1, ny) * grid_size[1] + origin[1]
    z = np.linspace(0, nz - 1, nz) * grid_size[2] + origin[2]
    interp_func = RegularGridInterpolator((x, y, z), data)
    return interp_func





    