
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

class PerturbedCollisionBoundaryPatternFormation(CollisionBoundaryPatternFormation):
    """
    Subclass that allows for continuous phase perturbation (random or periodic)
    on specific subsets of particles.
    """
    def __init__(self, perturbation_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbation_list = perturbation_list if perturbation_list else []
        # perturbation_list format:
        # [
        #    {
        #       'indices': np.array([...]),
        #       'type': 'random' or 'periodic',
        #       'strength': float,
        #       'frequency': float (optional)
        #    },
        #    ...
        # ]
        self.time_elapsed = 0.0

    def update(self):
        # 1. Run standard update (dynamics + interactions + boundary)
        super().update()
        
        # 2. Update time
        self.time_elapsed += self.dt
        
        # 3. Apply continuous perturbations
        if not self.perturbation_list:
            return

        additional_phase_delta = np.zeros(self.agentsNum)
        
        for p in self.perturbation_list:
            indices = p['indices']
            p_type = p['type']
            strength = p.get('strength', 0.0)
            
            if p_type == 'random':
                # Langevin noise: strength * N(0,1) * sqrt(dt)
                # strength acts as diffusion coefficient related parameter
                noise = strength * np.random.normal(0, 1, size=len(indices)) * np.sqrt(self.dt)
                additional_phase_delta[indices] += noise
                
            elif p_type == 'periodic':
                freq = p.get('frequency', 1.0)
                # Force: F(t) = A * sin(wt)
                # Phase change: dTheta = F(t) * dt
                delta = strength * np.sin(freq * self.time_elapsed) * self.dt
                additional_phase_delta[indices] += delta
        
        # Apply delta and wrap to [0, 2pi]
        if np.any(additional_phase_delta != 0):
            self.phaseTheta = np.mod(self.phaseTheta + additional_phase_delta, 2 * np.pi)

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

def cluster_particles(position, phase, dist_threshold=1.5, phase_threshold=0.1, boundary_length=7.0):
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

    # Calculate largest cluster for default
    if n_clusters > 0:
        counts = np.bincount(labels[labels>=0])
        largest_cluster_id = np.argmax(counts)
    else:
        largest_cluster_id = 0

    print("-" * 50)
    print(f"Enter cluster modifications.")
    print(f"Format: 'id:type:param1:param2:ratio'")
    print(f"Types:")
    print(f"  - shift (or s): One-time phase shift. param1=shift(in PI).")
    print(f"  - random (or r): Continuous random noise. param1=strength.")
    print(f"  - periodic (or p): Continuous periodic forcing. param1=strength, param2=freq.")
    print(f"Examples:")
    print(f"  '0:s:1.0' -> Cluster 0, shift 1.0*pi")
    print(f"  '0:r:5.0' -> Cluster 0, random noise strength 5.0")
    print(f"  '0:p:5.0:2.0' -> Cluster 0, strength 5.0, freq 2.0")
    print(f"  '0:p:5.0:2.0:0.5' -> ... applied to 50% of particles")
    print(f"Press Enter to modify the largest cluster ({largest_cluster_id}) by PI (shift).")
    print("-" * 50)
    
    parsed_mods = []
    
    while True:
        user_in = input(f"Enter modifications: ")
        if user_in.strip() == "":
            # Default: shift largest cluster by pi
            parsed_mods.append({
                'id': largest_cluster_id,
                'type': 'shift',
                'params': [1.0],
                'ratio': 1.0
            })
            break
        
        try:
            parts = user_in.split(',')
            valid_input = True
            temp_mods = []
            
            for part in parts:
                part = part.strip()
                if not part: continue
                
                subparts = [s.strip() for s in part.split(':')]
                
                if len(subparts) == 0: continue
                
                cid = int(subparts[0])
                if not (0 <= cid < n_clusters):
                    print(f"Error: Cluster ID {cid} out of range.")
                    valid_input = False
                    break
                
                # Defaults
                mod_type = 'shift'
                mod_args = []
                ratio = 1.0
                
                # Heuristic parsing
                if len(subparts) == 1:
                    # "0" -> shift 1.0 pi
                    mod_args = [1.0]
                elif len(subparts) >= 2:
                    # Check if second arg is type or number
                    p2 = subparts[1].lower()
                    if p2 in ['shift', 's']:
                        mod_type = 'shift'
                        # param1 (shift val)
                        if len(subparts) > 2: mod_args = [float(subparts[2])]
                        else: mod_args = [1.0]
                        # ratio
                        if len(subparts) > 3: ratio = float(subparts[3])
                    elif p2 in ['random', 'r']:
                        mod_type = 'random'
                        if len(subparts) > 2: mod_args = [float(subparts[2])]
                        else: mod_args = [1.0] # default strength
                        if len(subparts) > 3: ratio = float(subparts[3])
                    elif p2 in ['periodic', 'p']:
                        mod_type = 'periodic'
                        if len(subparts) > 2: mod_args.append(float(subparts[2])) # strength
                        else: mod_args.append(1.0)
                        if len(subparts) > 3: mod_args.append(float(subparts[3])) # freq
                        else: mod_args.append(1.0)
                        if len(subparts) > 4: ratio = float(subparts[4])
                    else:
                        # Assume old format: id:shift:ratio
                        # Or id:shift
                        try:
                            val = float(p2)
                            mod_type = 'shift'
                            mod_args = [val]
                            if len(subparts) > 2: ratio = float(subparts[2])
                        except ValueError:
                             print(f"Unknown type or value: {p2}")
                             valid_input = False
                             break
                
                temp_mods.append({
                    'id': cid,
                    'type': mod_type,
                    'params': mod_args,
                    'ratio': ratio
                })
            
            if valid_input and temp_mods:
                parsed_mods = temp_mods
                break
            elif not valid_input:
                continue
            else:
                print("No valid modifications parsed.")
                
        except Exception as e:
            print(f"Invalid input: {e}")

    print(f"Applying modifications: {parsed_mods}")

    # Create figure for result
    fig, ax = plt.subplots(figsize=(10, 10))
    
    new_save_path = os.path.join(script_dir, "data_modified")
    os.makedirs(new_save_path, exist_ok=True)
    
    # Modify Data
    modified_theta = last_theta.copy()
    continuous_perturbations = []
    
    title_parts = []

    for mod in parsed_mods:
        cid = mod['id']
        m_type = mod['type']
        mod_params = mod['params']
        ratio = mod['ratio']
        
        # Select particles
        cluster_indices = np.where(labels == cid)[0]
        n_cluster_particles = len(cluster_indices)
        n_modify = int(n_cluster_particles * ratio)
        
        if ratio >= 1.0:
            selected_indices = cluster_indices
        else:
            selected_indices = np.random.choice(cluster_indices, n_modify, replace=False)
            
        print(f"  - Cluster {cid} ({m_type}): targeting {n_modify}/{n_cluster_particles} particles.")
        
        if m_type == 'shift':
            shift_val = mod_params[0] * np.pi
            modified_theta[selected_indices] = np.mod(modified_theta[selected_indices] + shift_val, 2*np.pi)
            title_parts.append(f"C{cid}:Shift {mod_params[0]:.1f}$\pi$")
            
        elif m_type == 'random':
            strength = mod_params[0]
            continuous_perturbations.append({
                'indices': selected_indices,
                'type': 'random',
                'strength': strength
            })
            title_parts.append(f"C{cid}:Rnd({strength})")
            
        elif m_type == 'periodic':
            strength = mod_params[0]
            freq = mod_params[1] if len(mod_params) > 1 else 1.0
            continuous_perturbations.append({
                'indices': selected_indices,
                'type': 'periodic',
                'strength': strength,
                'frequency': freq
            })
            title_parts.append(f"C{cid}:Per({strength}, {freq})")
    
    modified_pos = last_pos.copy()
    
    # Initialize New Model
    model = PerturbedCollisionBoundaryPatternFormation(
        perturbation_list=continuous_perturbations,
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
        savePath=new_save_path,
        shotsnaps=int(params.get('snap', 10)),
        randomSeed=int(params.get('seed', 9)),
        overWrite=True
    )
    
    # Overwrite position
    model.positionX = modified_pos
    
    # Run Simulation
    run_steps = 50000
    model.run(run_steps)
    
    # Plot Result
    model.plot(ax=ax)
    ax.set_aspect('equal')
    
    title_str = ", ".join(title_parts) if title_parts else "No Modifications"
    ax.set_title(f"Modified: {title_str}")
    
    plt.tight_layout()
    plt.savefig("Modified_Simulation.png")
    plt.show()

if __name__ == "__main__":
    main()
