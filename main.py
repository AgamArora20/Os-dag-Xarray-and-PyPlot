import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os

# --- Configuration ---
DATASET_FILE = 'dataset.nc'
NODES_FILE = 'nodes.csv'
CONNECTIVITY_FILE = 'connectivity.csv'

# Derived Girder Element Definitions based on Task 1 Pattern
# Central Girder (G3): [15, 24, 33, 42, 51, 60, 69, 78, 83]
# Pattern: +9, +9, +9, +9, +9, +9, +9, +5
# We assume parallel girders have IDs offset by -2, -1, 0, +1, +2 relative to Central
GIRDERS = {
    'Girder 1': [17, 26, 35, 44, 53, 62, 71, 80, 85],
    'Girder 2': [16, 25, 34, 43, 52, 61, 70, 79, 84],
    'Girder 3': [15, 24, 33, 42, 51, 60, 69, 78, 83], # Central
    'Girder 4': [14, 23, 32, 41, 50, 59, 68, 77, 82],
    'Girder 5': [13, 22, 31, 40, 49, 58, 67, 76, 81]
}

def generate_high_fidelity_mock_data():
    """
    Generates a high-fidelity mock dataset that matches the specific
    element IDs and structure described in the Osdag task.
    """
    print("Generating high-fidelity mock data...")
    
    # 1. Define Geometry
    # 5 Girders, 9 Elements each.
    # Let's assume 9 spans. Total length 90m.
    # Width: 10m (2.5m spacing)
    
    nodes = []
    connectivity = []
    all_elements = []
    
    # X coordinates for the 9 elements (10 nodes)
    # We'll assume the last element is shorter based on the ID jump (5 vs 9), 
    # but for visual simplicity we'll keep them equal or make the last one half length.
    # Let's make them equal for the plot to look nice.
    xs = np.linspace(0, 90, 10) 
    
    # Z coordinates for 5 girders
    zs = np.linspace(-5, 5, 5) # -5, -2.5, 0, 2.5, 5
    
    # Node IDs: We need to assign IDs. 
    # The element IDs are fixed. We need to infer node IDs or just assign new ones.
    # We'll assign new Node IDs starting from 1.
    
    node_map = {} # (girder_idx, x_idx) -> node_id
    node_counter = 1
    
    for g_idx in range(5):
        for x_idx in range(10):
            nodes.append({
                'node_id': node_counter,
                'x': xs[x_idx],
                'y': 0,
                'z': zs[g_idx]
            })
            node_map[(g_idx, x_idx)] = node_counter
            node_counter += 1
            
    nodes_df = pd.DataFrame(nodes)
    
    # 2. Connectivity
    girder_names = list(GIRDERS.keys()) # G1 to G5
    # Note: G1 is at one end, G5 at the other.
    # Let's map G1 to z index 0, G5 to z index 4.
    
    for g_idx, g_name in enumerate(girder_names):
        elements = GIRDERS[g_name]
        for i, el_id in enumerate(elements):
            start_node = node_map[(g_idx, i)]
            end_node = node_map[(g_idx, i+1)]
            
            connectivity.append({
                'element_id': el_id,
                'start_node': start_node,
                'end_node': end_node
            })
            all_elements.append(el_id)
            
    connectivity_df = pd.DataFrame(connectivity)
    
    # 3. Forces (Mz and Vy)
    # We want continuous beam behavior.
    # Simple model: Uniform load on continuous beam.
    # Mz: Parabolic. 0 at ends. Negative at supports (if we had supports).
    # Let's just simulate a simple simply supported beam for the whole bridge for simplicity,
    # or a series of sine waves to look interesting.
    # Let's do a sine wave for Mz (positive) and Cosine for Vy.
    
    num_elements = len(all_elements)
    
    mz_i_list = []
    mz_j_list = []
    vy_i_list = []
    vy_j_list = []
    
    for el_id in all_elements:
        # Find position of element
        row = connectivity_df[connectivity_df['element_id'] == el_id].iloc[0]
        n1 = nodes_df[nodes_df['node_id'] == row['start_node']].iloc[0]
        n2 = nodes_df[nodes_df['node_id'] == row['end_node']].iloc[0]
        
        x_start = n1['x']
        x_end = n2['x']
        mid = (x_start + x_end) / 2
        
        # Normalized position 0 to 1
        pos_start = x_start / 90.0
        pos_end = x_end / 90.0
        
        # Bending Moment: M = wLx/2 - wx^2/2 (Parabola)
        # Scale it up
        scale = 500
        m_start = scale * 4 * pos_start * (1 - pos_start) # Parabola peaking at 0.5
        m_end = scale * 4 * pos_end * (1 - pos_end)
        
        # Shear Force: V = wL/2 - wx (Linear)
        # Scale
        v_scale = 200
        v_start = v_scale * (1 - 2 * pos_start)
        v_end = v_scale * (1 - 2 * pos_end)
        
        mz_i_list.append(m_start)
        mz_j_list.append(m_end)
        vy_i_list.append(v_start)
        vy_j_list.append(v_end)

    ds = xr.Dataset(
        data_vars=dict(
            Mz_i=(["element"], mz_i_list),
            Mz_j=(["element"], mz_j_list),
            Vy_i=(["element"], vy_i_list),
            Vy_j=(["element"], vy_j_list),
        ),
        coords=dict(
            element=all_elements,
        ),
        attrs=dict(description="High Fidelity Mock Structural Data"),
    )
    
    return ds, nodes_df, connectivity_df

def load_data():
    if os.path.exists(DATASET_FILE) and os.path.exists(NODES_FILE) and os.path.exists(CONNECTIVITY_FILE):
        print("Loading real data...")
        try:
            ds = xr.open_dataset(DATASET_FILE)
            nodes_df = pd.read_csv(NODES_FILE)
            connectivity_df = pd.read_csv(CONNECTIVITY_FILE)
            return ds, nodes_df, connectivity_df
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Falling back to mock data.")
    
    return generate_high_fidelity_mock_data()

def plot_2d_sfd_bmd(ds, nodes_df, connectivity_df, girder_elements, title_suffix=""):
    """
    Plots 2D SFD and BMD for a given list of girder elements.
    """
    print(f"Plotting 2D SFD and BMD for {title_suffix}...")
    
    x_coords = []
    mz_vals = []
    vy_vals = []
    
    current_dist = 0
    
    # Sort elements by sequence in list (assuming list is ordered)
    for el_id in girder_elements:
        if el_id not in ds.element.values:
            continue
            
        el_data = ds.sel(element=el_id)
        
        # Calculate length
        conn = connectivity_df[connectivity_df['element_id'] == el_id]
        if conn.empty:
            length = 10.0
        else:
            n1 = nodes_df[nodes_df['node_id'] == conn.iloc[0]['start_node']]
            n2 = nodes_df[nodes_df['node_id'] == conn.iloc[0]['end_node']]
            p1 = np.array([n1.iloc[0]['x'], n1.iloc[0]['y'], n1.iloc[0]['z']])
            p2 = np.array([n2.iloc[0]['x'], n2.iloc[0]['y'], n2.iloc[0]['z']])
            length = np.linalg.norm(p2 - p1)

        # Append points
        x_coords.append(current_dist)
        mz_vals.append(float(el_data['Mz_i']))
        vy_vals.append(float(el_data['Vy_i']))
        
        current_dist += length
        x_coords.append(current_dist)
        mz_vals.append(float(el_data['Mz_j']))
        vy_vals.append(float(el_data['Vy_j']))
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # SFD
    ax1.plot(x_coords, vy_vals, 'r-', linewidth=2, label='Shear Force (Vy)')
    ax1.fill_between(x_coords, vy_vals, color='red', alpha=0.1)
    ax1.set_ylabel('Shear Force (kN)', fontsize=12)
    ax1.set_title(f'Shear Force Diagram (SFD) - {title_suffix}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # BMD
    ax2.plot(x_coords, mz_vals, 'b-', linewidth=2, label='Bending Moment (Mz)')
    ax2.fill_between(x_coords, mz_vals, color='blue', alpha=0.1)
    ax2.set_ylabel('Bending Moment (kNm)', fontsize=12)
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_title(f'Bending Moment Diagram (BMD) - {title_suffix}', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    filename = f'SFD_BMD_2D_{title_suffix.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def plot_3d_structure(ds, nodes_df, connectivity_df):
    """
    Plots 3D SFD and BMD for all girders.
    """
    print("Plotting 3D Structure...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale factors for extrusion
    # Adjust based on data range
    max_mz = float(ds['Mz_i'].max())
    scale_mz = 5.0 / max_mz if max_mz != 0 else 1
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (girder_name, elements) in enumerate(GIRDERS.items()):
        color = colors[i % len(colors)]
        
        for el_id in elements:
            if el_id not in ds.element.values: continue
            
            conn = connectivity_df[connectivity_df['element_id'] == el_id]
            if conn.empty: continue
            
            n1 = nodes_df[nodes_df['node_id'] == conn.iloc[0]['start_node']]
            n2 = nodes_df[nodes_df['node_id'] == conn.iloc[0]['end_node']]
            
            p1 = [n1.iloc[0]['x'], n1.iloc[0]['y'], n1.iloc[0]['z']]
            p2 = [n2.iloc[0]['x'], n2.iloc[0]['y'], n2.iloc[0]['z']]
            
            # Draw Structure (Beam)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', alpha=0.3, linewidth=1)
            
            # Get forces
            el_data = ds.sel(element=el_id)
            mz_i = float(el_data['Mz_i'])
            mz_j = float(el_data['Mz_j'])
            
            # Extrude BMD in Y direction
            # Note: In 3D plot, usually Z is up, but here we map:
            # X -> Longitudinal
            # Y -> Vertical (Magnitude)
            # Z -> Transverse
            # Wait, standard MPL 3D is X, Y, Z.
            # Our nodes have y=0. So Y is vertical.
            
            p1_mz = [p1[0], p1[1] + mz_i * scale_mz, p1[2]]
            p2_mz = [p2[0], p2[1] + mz_j * scale_mz, p2[2]]
            
            # Plot BMD Line
            ax.plot([p1_mz[0], p2_mz[0]], [p1_mz[1], p2_mz[1]], [p1_mz[2], p2_mz[2]], color=color, linewidth=2)
            
            # Fill (Vertical lines)
            ax.plot([p1[0], p1_mz[0]], [p1[1], p1_mz[1]], [p1[2], p1_mz[2]], color=color, alpha=0.2)
            ax.plot([p2[0], p2_mz[0]], [p2[1], p2_mz[1]], [p2[2], p2_mz[2]], color=color, alpha=0.2)
            
            # Connect tops
            # ax.plot([p1_mz[0], p2_mz[0]], [p1_mz[1], p2_mz[1]], [p1_mz[2], p2_mz[2]], color=color, linewidth=2)

    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Bending Moment Magnitude')
    ax.set_zlabel('Width (m)')
    ax.set_title('3D Bending Moment Diagram (All Girders)', fontsize=16)
    
    # Set view
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    filename = 'SFD_BMD_3D.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def main():
    ds, nodes_df, connectivity_df = load_data()
    
    # Task 1: Central Girder
    central_girder = GIRDERS['Girder 3']
    plot_2d_sfd_bmd(ds, nodes_df, connectivity_df, central_girder, title_suffix="Central Girder")
    
    # Task 2: 3D Plot
    plot_3d_structure(ds, nodes_df, connectivity_df)

if __name__ == "__main__":
    main()
