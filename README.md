# Osdag Screening Task:# Os-dag-Xarray-and-PyPlot
### Screening Task Submission

This project generates Shear Force Diagrams (SFD) and Bending Moment Diagrams (BMD) for a bridge structure using Xarray and Matplotlib.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install xarray matplotlib numpy pandas netCDF4
    ```

2.  **Data Files:**
    Place the following files in this directory:
    *   `dataset.nc`: The Xarray dataset containing `Mz` and `Vy` values.
    *   `nodes.csv`: Node coordinates with columns `node_id, x, y, z`.
    *   `connectivity.csv`: Element connectivity with columns `element_id, start_node, end_node`.

    *Note: If these files are not found, the script will generate mock data for demonstration purposes.*

## Running the Code

Run the main script:
```bash
python3 main.py
```

## Output

*   `SFD_BMD_2D_Central_Girder.png`: 2D plots for the central longitudinal girder.
*   `SFD_BMD_3D.png`: 3D visualization of the bridge with extruded bending moment diagrams.

## Task Details

*   **Task 1:** 2D SFD and BMD for the central girder (elements [15, 24, 33, ...]).
*   **Task 2:** 3D SFD and BMD for all girders, visualized in a style similar to MIDAS.
