# TCFDgepWPS

## Overview

TCFDgepWPS is a symbolic regression framework based on Gene
Expression Programming (GEP) for wall-pressure spectrum (WPS) modeling
of turbulent boundary layers.

For users who intend to directly utilize the trained model,
please refer to results/compile_Goody-T-HGEP.py, which provides 
the implementation of the trained model.

c

The model is developed based on the open-source geppy framework with
customized modifications for GEP-based physical modeling.

------------------------------------------------------------------------

# Features

-   Gene Expression Programming (GEP)-based symbolic regression
-   Hierarchical GEP (HGEP) model construction
-   Goody-type wall-pressure spectrum prediction
-   Explicit analytical expression generation
-   Multi-sub-model weighted combination strategy
-   MPI-based parallel evolutionary optimization
-   Automatic model saving and loading


------------------------------------------------------------------------

# Project Structure

    TCFDgepWPS
    │
    ├── config
    │   └── gepConfig.py
    │       Configuration of GEP evolutionary parameters
    │
    ├── data
    │   ├── readData.py
    │   │   Data loading and preprocessing
    │   └── data_delta_star.pkl
    │       Dataset used for model training/evaluation
    │
    ├── gep
    │   Partial third-party dependencies required by this project
    │   │
    │   ├── geppy
    │   │   Modified version of the open-source geppy package
    │   ├── deap
    │   ├── mpi4py
    │   ├── scipy
    │   └── ...
    │
    ├── supFunctions
    │   ├── basicOperations.py
    │   │   Basic mathematical operators used in symbolic expressions
    │   ├── fitness.py
    │   │   Fitness evaluation functions for GEP evolution
    │   ├── jointCor.py
    │   │   Correlation functions for HGEP sub-model combination
    │   └── model_io.py
    │       Model storage and loading utilities
    │
    ├── results
    │   ├── bestIndividual.py
    │   │   Extraction and analysis of optimal GEP individuals
    │   └── compile_Goody-T-HGEP.py
    │       Compilation of symbolic GEP expressions into executable models
    │
    └── Goody-T-HGEP_main.py
        Main program for HGEP model evolution

------------------------------------------------------------------------

# Third-party Dependencies

Partial third-party dependencies are included in the `gep` directory to
facilitate code reproducibility and reduce additional installation
requirements.

The included dependencies provide essential functions for:

-   evolutionary computation (`deap`)
-   symbolic regression (`geppy`)
-   scientific computing (`scipy`)
-   parallel computation (`mpi4py`)

## Modified geppy package

The `geppy` package included in this repository is **a modified version
of the original open-source geppy library**.

The modifications were introduced to support the hierarchical GEP (HGEP)
framework developed in this work, including customized symbolic
expression handling and model construction procedures.

The included `geppy` package is **not identical to the official release
version**. Users should use the provided version to ensure compatibility
with this framework.

------------------------------------------------------------------------

# Requirements

Recommended environment:

    Python >= 3.10

Required packages:

    numpy
    scipy
    deap
    mpi4py

Since partial dependencies are included in the `gep` directory,
additional package installation may not be required depending on the
user's Python environment.

# Usage

## Single-process execution

``` bash
python Goody-T-HGEP_main.py
```

## MPI parallel execution

``` bash
mpirun -np 8 python Goody-T-HGEP_main.py
```

where `-np 8` specifies the number of parallel processes.

------------------------------------------------------------------------

# Input Data

The model requires boundary-layer parameters and frequency information:

    omega
    RT
    beta_star
    G
    H
    Delta_ZS
    cf


# Citation

If you use this code for academic research, please cite:

    [Please insert the corresponding paper citation]

------------------------------------------------------------------------

# License

This project is intended for academic research purposes.

The included third-party libraries follow their respective licenses.
