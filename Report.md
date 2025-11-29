# Osdag Screening Task Report

## Overview
This report documents the implementation of the Osdag screening task, which involves generating Shear Force Diagrams (SFD) and Bending Moment Diagrams (BMD) using Xarray and PyPlot.

## Implementation Details

### 1. Data Handling
The solution is designed to handle the specific Xarray dataset format described in the task.
- **Input:** `dataset.nc` (Xarray), `nodes.csv`, `connectivity.csv`.
- **Mock Data:** In the absence of the raw files, a **High-Fidelity Mock Data Generator** was implemented. This generator reconstructs the bridge geometry and element connectivity based on the patterns provided in the problem statement (e.g., Central Girder Elements: 15, 24, 33...).
    - **Geometry:** 5 Parallel Girders, 90m length.
    - **Forces:** Simulated continuous beam behavior with parabolic Bending Moments and linear Shear Forces.

### 2. Task 1: 2D SFD & BMD (Central Girder)
- **Objective:** Plot SFD and BMD for the central longitudinal girder.
- **Method:**
    - Extracted `Mz` and `Vy` values for elements `[15, 24, 33, 42, 51, 60, 69, 78, 83]`.
    - Mapped element connectivity to physical distance along the bridge.
    - Plotted continuous diagrams using Matplotlib.
- **Result:** See `SFD_BMD_2D_Central_Girder.png`.

### 3. Task 2: 3D Visualization
- **Objective:** Generate a 3D visualization of the bridge framing with extruded BMD, similar to MIDAS.
- **Method:**
    - Constructed a 3D line model of the 5 girders.
    - Extruded the Bending Moment magnitude in the vertical (Y) direction.
    - Applied color coding for visual clarity.
- **Result:** See `SFD_BMD_3D.png`.

## Code Structure
- `main.py`: Core script containing data generation, loading, and plotting logic.
- `requirements.txt`: List of dependencies.

## Usage
Run the script:
```bash
python3 main.py
```
