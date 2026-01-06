
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# Add current directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CollisionBoundaryPatternFormation, StateAnalysis, hexCmap

def load_data(file_path):
    """Load data from .h5 file."""
    # We can use StateAnalysis to load data, but it requires a model instance or we can just read the h5 directly
    # Since we want to use StateAnalysis features later, it might be good to reconstruct the model
    # But for now, let's just read the h5 file directly to get the data
    
    print(f"Loading data from {file_path}...")
    try:
        with pd.HDFStore(file_path, mode='r') as store:
            positionX_df = store.select("positionX")
            phaseTheta_df = store.select("phaseTheta")
            
        return positionX_df, phaseTheta_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def parse_filename_params(filename):
    """Parse parameters from filename to reconstruct the model."""
    # Example filename: CollisionBoundaryPatternFormation(K=25.000,D0=7.000,A0=1.257,L=7.0,v=3.0,dist=uniform,wMin=0.000,dw=6.000,N=1000,dt=0.005,snap=10,seed=9).h5
    params = {}
    name = os.path.basename(filename)
    if "(" in name and ")" in name:
        param_str = name.split("(")[1].split(")")[0]
        items = param_str.split(",")
        for item in items:
            if "=" in item:
                key, value = item.split("=")
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return params

def cluster_particles(position, phase, dist_threshold=0.5, phase_threshold=0.1, boundary_length=7.0):
    """
    Cluster particles based on spatial proximity and phase synchronization.
    
    Args:
        position: (N, 2) array of positions
        phase: (N,) array of phases
        dist_threshold: maximum spatial distance for connection
        phase_threshold: maximum phase difference for connection
        boundary_length: for periodic boundary conditions (if applicable, though CollisionBoundary might not be periodic)
        
    Returns:
        labels: (N,) array of cluster labels
        n_clusters: number of clusters
    """
    N = position.shape[0]
    
    # Calculate spatial distance matrix
    # Using simple Euclidean distance for now. If periodic, need to adjust.
    # The filename suggests "CollisionBoundary", which implies hard boundaries, not periodic.
    dist_mat = squareform(pdist(position))
    
    # Calculate phase difference matrix
    phase_diff_mat = np.abs(phase[:, np.newaxis] - phase[np.newaxis, :])
    # Handle phase periodicity (min difference)
    phase_diff_mat = np.minimum(phase_diff_mat, 2*np.pi - phase_diff_mat)
    
    # Adjacency matrix: connected if close in space AND close in phase
    adj_mat = (dist_mat < dist_threshold) & (phase_diff_mat < phase_threshold)
    
    # Use NetworkX to find connected components
    G = nx.from_numpy_array(adj_mat)
    components = list(nx.connected_components(G))
    
    labels = np.full(N, -1, dtype=int)
    for i, comp in enumerate(components):
        for node_idx in comp:
            labels[node_idx] = i
            
    return labels, len(components)

def plot_clusters(position, phase, labels, n_clusters, boundary_length=7.0, title="Clusters"):
    """Plot particles colored by cluster."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate distinct colors for clusters
    # Use a colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_pos = position[mask]
        cluster_phase = phase[mask]
        
        # Calculate centroid
        centroid = np.mean(cluster_pos, axis=0)
        
        ax.scatter(cluster_pos[:, 0], cluster_pos[:, 1], 
                   color=colors[i], label=f"Cluster {label}", s=20, alpha=0.8)
        
        # Draw quiver for phase
        ax.quiver(cluster_pos[:, 0], cluster_pos[:, 1],
                  np.cos(cluster_phase), np.sin(cluster_phase),
                  color='k', alpha=0.3, width=0.002, scale=20)
        
        # Add label text at centroid
        ax.text(centroid[0], centroid[1], str(label), 
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    ax.set_xlim(0, boundary_length)
    ax.set_ylim(0, boundary_length)
    ax.set_aspect('equal')
    ax.set_title(title)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def main():
    # 1. File Path
    # Using one of the files found
    filename = "CollisionBoundaryPatternFormation(K=18.750,D0=7.000,A0=2.513,L=7.0,v=3.0,dist=uniform,wMin=0.000,dw=0.000,N=2000,dt=0.005,snap=10,seed=9).h5"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", filename)
    
    # 2. Load Data
    pos_df, theta_df = load_data(file_path)
    if pos_df is None:
        return

    # Parse params
    params = parse_filename_params(filename)
    N = int(params.get('N', 2000))
    L = float(params.get('L', 7.0))
    
    # Get last frame
    # Reshape
    n_frames = len(pos_df) // N
    last_pos = pos_df.values[-N:]
    last_theta = theta_df.values[-N:].flatten()
    
    print(f"Loaded {n_frames} frames. Analyzing last frame.")
    
    # 3. Cluster
    # Parameters for clustering - these might need tuning
    
    labels, n_clusters = cluster_particles(last_pos, last_theta, dist_threshold=1.5, phase_threshold=0.1, boundary_length=L)
    print(f"Found {n_clusters} clusters.")
    
    # 4. Plot original clusters
    plot_clusters(last_pos, last_theta, labels, n_clusters, boundary_length=L, title="Identified Clusters")
    
    # 5. Select Cluster to Modify
    if n_clusters == 0:
        print("No clusters found to modify.")
        return

    # For automation, we select the largest cluster
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_id = np.argmax(counts)
    
    print(f"Total clusters found: {n_clusters}")
    print(f"Largest cluster is ID {largest_cluster_id} with {counts[largest_cluster_id]} particles.")
    
    # Allow user to manually select cluster
    while True:
        try:
            user_in = input(f"Enter cluster ID to modify (0-{n_clusters-1}) [default: {largest_cluster_id}]: ")
            if user_in.strip() == "":
                selected_cluster_id = largest_cluster_id
                break
            selected_cluster_id = int(user_in)
            if 0 <= selected_cluster_id < n_clusters:
                break
            else:
                print(f"ID out of range. Please enter 0-{n_clusters-1}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    print(f"Modifying Cluster {selected_cluster_id}...")
    
    # 6. Modify Data
    # Let's rotate the phase of this cluster by pi
    mask = labels == selected_cluster_id
    modified_theta = last_theta.copy()
    modified_theta[mask] = np.mod(modified_theta[mask] + 0.5*np.pi, 2*np.pi)
    
    modified_pos = last_pos.copy()
    # Optionally move them? Let's just change phase for now as requested "modify particle data"
    
    # 7. Initialize New Model
    print("Initializing new simulation with modified state...")
    
    # Reconstruct model args
    # params from filename: K, D0, A0, L, v, dist, wMin, dw, N, dt, snap, seed
    
    new_save_path = os.path.join(script_dir, "data_modified")
    os.makedirs(new_save_path, exist_ok=True)
    
    model = CollisionBoundaryPatternFormation(
        strengthK=params.get('K', 18.75),
        distanceD0=params.get('D0', 7),
        phaseLagA0=params.get('A0', 0.8 * np.pi),
        boundaryLength=params.get('L', 7),
        speedV=params.get('v', 3),
        freqDist=params.get('dist', 'uniform'),
        initPhaseTheta=modified_theta, # Pass modified phases
        omegaMin=params.get('wMin', 0),
        deltaOmega=params.get('dw', 0),
        agentsNum=int(params.get('N', 2000)),
        dt=params.get('dt', 0.005),
        tqdm=True,
        savePath=new_save_path, # New save path
        shotsnaps=int(params.get('snap', 10)),
        randomSeed=int(params.get('seed', 9)),
        overWrite=True
    )
    
    # Overwrite position
    model.positionX = modified_pos
    
    # 8. Run Simulation
    run_steps = 2000
    print(f"Running simulation for {run_steps} steps...")
    model.run(run_steps)
    
    # 9. Plot Result
    print("Simulation finished. Plotting result...")
    model.plot()
    plt.title("After Modification and Evolution")
    plt.savefig("After_Modification_and_Evolution.png")
    plt.show()

if __name__ == "__main__":
    main()
