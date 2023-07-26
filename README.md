# Weighted Empirical Bayse

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The WEB (Weighted Empirical Bayes) package is a comprehensive Python library specifically designed for the analysis of density maps and atomic models in the context of amino acids. It offers a wide range of tools and functions to conduct weighted empirical Bayesian analysis, enabling researchers to study the fitted results of proteins for all amino acids and effectively handle outlier-related issues within and between each amino acid.

## Features

- Density Map and Atomic Model Analysis: The package allows users to analyze density maps and atomic models obtained from molecular dynamics simulations or experimental data. It provides functions to read and process these data for further analysis.

- Weighted Empirical Bayesian Analysis: The core functionality of the package revolves around the weighted empirical Bayesian analysis. It employs a statistical approach to assign appropriate weights to data points, taking into account the uncertainties and variances in the data. This ensures a more accurate and robust estimation of protein properties for all amino acids.

- Fitted Result of Protein: Users can use the package to obtain the fitted results of the protein for all amino acids. This includes detailed information about the distribution of density values and variations at each amino acid's location.

- Outlier Detection and Handling: The package offers mechanisms to detect and deal with outliers within and between each amino acid. Outliers can significantly impact the accuracy of the analysis, and the package ensures their proper treatment to obtain reliable results.

- Visualizations: To aid in data exploration and presentation, the package provides visualization functions to plot density maps and the corresponding fitted results. These visualizations help researchers gain valuable insights into the protein's behavior and its interactions with amino acids.

## Installation

To use the WEB (Weighted Empirical Bayes) package, we recommend creating a virtual environment and installing the necessary dependencies. Follow the steps below to set up the package:

1. Create a virtual environment named "WEB" using `conda` with Python version 3.10:

```bash
conda create -n WEB python=3.10
```

2. Activate the virtual environment:

```bash
conda activate WEB
```

3. Install the required dependencies by using `pip` and the `requirements.txt` file provided with the package:

```bash
pip install -r requirements.txt
```

## Usage

```python
from WEB import *

# Load data
web = WEB(start_radius=0, max_radius=1, gap=0.2)
data = web.read_data(
    root_map="path/to/map_file.mrc", 
    root_pdb="path/to/pdb_file.pdb",
    atomic="atom/to/survey",
    )

# Fit the model
_ = web.paramters_initial()
betas_WEB, histories = web.WEB_iter()

# Plot fitted result
web.plot_data()

# Find outliers
outliers, statistic_distances = web.find_outliers()

# Visualize outliers
web.distances_hist()
web.confidence_regions_plot()
web.outliers_density_plot()
```

## Examples

The package includes two Jupyter Notebook examples demonstrating how to use the WEB model with real data and simulation data:

`Instance_real_data.ipynb`: Example using real data from map and pdb files.


`Instance_simulation.ipynb`: Example using synthetic data generated from the model.

## License

State the license under which your package is released. For example, if you are using the MIT License:

[MIT License](LICENSE)

---
