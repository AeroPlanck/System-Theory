"""
Particle Evolution Trajectory Analysis Script
Analyzes particle evolution trajectories from initial random distribution to final lattice solution in collective dynamics simulations

Main Functions:
1. Determine spatial position distribution of each particle cluster after system reaches steady state
2. Calculate average rotation center coordinates for all particles within each cluster
3. Record particle IDs within each cluster at final state using cluster rotation centers as reference
4. Color-code particles based on their cluster IDs and assign different geometric markers for enhanced visualization
5. Map final state particle IDs back to initial spatial distribution
6. Establish correspondence diagram between initial and final particle spatial distributions
7. Create enhanced visualizations with different colors and shapes for each cluster to improve observability
"""

import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import json
import os
import shutil
import pickle
import sys
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import seaborn as sns

# Add path to import main module
sys.path.append("..")
from main import PhaseLagPatternFormation, StateAnalysis

# Set matplotlib parameters
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['figure.figsize'] = (12, 8)

# Set seaborn theme
sns.set_theme(
    style="ticks", 
    font_scale=1.1, 
    rc={
        'figure.figsize': (12, 8),
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': '#dddddd',
        'grid.linewidth': 0.5,
        "lines.linewidth": 1.5,
        'text.color': '#000000',
        'figure.titleweight': "bold",
        'xtick.color': '#000000',
        'ytick.color': '#000000'
    }
)

# Color mapping
colors = ["#403990", "#3A76D6", "#FFC001", "#F46F43", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
cluster_cmap = mcolors.LinearSegmentedColormap.from_list("cluster_cmap", colors)

class ParticleEvolutionAnalyzer:
    """Particle Evolution Trajectory Analyzer"""
    
    def __init__(self, model_params=None, save_path="./analysis_results"):
        """
        Initialize analyzer
        
        Args:
            model_params: Model parameter dictionary
            save_path: Results save path
        """
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Default model parameters
        if model_params is None:
            model_params = {
                'strengthK': 20,
                'distanceD0': 1,
                'phaseLagA0': 0.6 * np.pi,
                'omegaMin': 0,
                'deltaOmega': 0,
                'agentsNum': 1000,
                'dt': 0.001,
                'boundaryLength': 7,
                'speedV': 3,
                'randomSeed': 9,
                'run_steps': 80000  # Add run steps parameter
            }
        
        self.model_params = model_params
        self.model = None
        self.sa = None
        self.clusters = None
        self.cluster_centers = None
        self.particle_trajectories = None
        
    def load_or_run_simulation(self, force_rerun=True):
        """Load or run simulation"""
        print("Initializing model...")
        
        # Create model instance
        self.model = PhaseLagPatternFormation(
            strengthK=self.model_params['strengthK'],
            distanceD0=self.model_params['distanceD0'],
            phaseLagA0=self.model_params['phaseLagA0'],
            omegaMin=self.model_params['omegaMin'],
            deltaOmega=self.model_params['deltaOmega'],
            agentsNum=self.model_params['agentsNum'],
            dt=self.model_params['dt'],
            boundaryLength=self.model_params['boundaryLength'],
            speedV=self.model_params['speedV'],
            tqdm=True,
            savePath="d:/PythonProject/System Theory/Frustration Induced Lattice/temp_data",
            shotsnaps=1,
            randomSeed=self.model_params['randomSeed'],
            overWrite=force_rerun
        )
        
        # Run simulation
        print(f"Running simulation, steps: {self.model_params['run_steps']}...")
        self.model.run(self.model_params['run_steps'])
        print("Simulation completed")
        
        # Create state analyzer
        print("Loading simulation data...")
        self.sa = StateAnalysis(self.model)
        print(f"Simulation data loaded, {self.sa.TNum} time steps in total")
        
    def detect_clusters(self, time_index=-1, eps=0.5, min_samples=3):
        """
        Detect particle clusters
        
        Args:
            time_index: Time index, -1 means last moment
            eps: Neighborhood radius for DBSCAN clustering
            min_samples: Minimum sample number for DBSCAN clustering
        """
        print("Detecting particle clusters...")
        
        # Get position data at specified time
        positions = self.sa.totalPositionX[time_index]
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        
        # Count clustering results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Detected {n_clusters} clusters, {n_noise} noise points")
        
        # Save clustering results
        self.clusters = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'positions': positions,
            'time_index': time_index
        }
        
        return labels, n_clusters
    
    def calculate_cluster_centers(self):
        """Calculate rotation center coordinates for each cluster"""
        if self.clusters is None:
            raise ValueError("Please run detect_clusters method first")
            
        print("Calculating cluster rotation centers...")
        
        labels = self.clusters['labels']
        positions = self.clusters['positions']
        phases = self.sa.totalPhaseTheta[self.clusters['time_index']]
        
        centers = {}
        
        for cluster_id in range(self.clusters['n_clusters']):
            # Get particles belonging to current cluster
            mask = labels == cluster_id
            cluster_positions = positions[mask]
            cluster_phases = phases[mask]
            
            if len(cluster_positions) == 0:
                continue
                
            # Calculate geometric center as initial estimate
            geometric_center = np.mean(cluster_positions, axis=0)
            
            # Optimize rotation center: minimize weighted distance to all particles
            def objective(center):
                distances = np.linalg.norm(cluster_positions - center, axis=1)
                # Use phase coherence as weight
                phase_coherence = np.abs(np.mean(np.exp(1j * cluster_phases)))
                return np.sum(distances) / (phase_coherence + 1e-6)
            
            result = minimize(objective, geometric_center, method='BFGS')
            rotation_center = result.x
            
            centers[cluster_id] = {
                'center': rotation_center,
                'geometric_center': geometric_center,
                'particle_indices': np.where(mask)[0],
                'particle_count': len(cluster_positions),
                'phase_coherence': np.abs(np.mean(np.exp(1j * cluster_phases)))
            }
            
        self.cluster_centers = centers
        print(f"Calculation completed, {len(centers)} cluster centers in total")
        
        return centers
    
    def assign_particle_colors(self):
        """Assign particle colors and markers based on clusters"""
        if self.clusters is None or self.cluster_centers is None:
            raise ValueError("Please run detect_clusters and calculate_cluster_centers methods first")
            
        print("Assigning particle colors and markers...")
        
        labels = self.clusters['labels']
        n_particles = len(labels)
        
        # Create color array
        particle_colors = np.zeros((n_particles, 3))
        
        # Define marker styles for different clusters
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        
        # Create marker array
        particle_markers = np.full(n_particles, 'o', dtype=object)
        
        # Assign colors and markers for each cluster
        for cluster_id in range(self.clusters['n_clusters']):
            mask = labels == cluster_id
            color_idx = cluster_id % len(colors)
            marker_idx = cluster_id % len(marker_styles)
            
            color = mcolors.to_rgb(colors[color_idx])
            marker = marker_styles[marker_idx]
            
            particle_colors[mask] = color
            particle_markers[mask] = marker
            
        # Use gray color and circle marker for noise points
        noise_mask = labels == -1
        particle_colors[noise_mask] = [0.5, 0.5, 0.5]
        particle_markers[noise_mask] = 'o'
        
        self.particle_colors = particle_colors
        self.particle_markers = particle_markers
        print("Color and marker assignment completed")
        
        return particle_colors, particle_markers
    
    def calculate_periodic_displacement(self, pos1, pos2, boundary_length):
        """Calculate displacement considering periodic boundary conditions"""
        displacement = pos2 - pos1
        
        # Apply periodic boundary correction
        # If displacement is larger than half the boundary, use the shorter path
        for i in range(displacement.shape[1]):  # For each dimension (x, y)
            mask_positive = displacement[:, i] > boundary_length / 2
            mask_negative = displacement[:, i] < -boundary_length / 2
            
            displacement[mask_positive, i] -= boundary_length
            displacement[mask_negative, i] += boundary_length
            
        return displacement
    
    def plot_particles_with_markers(self, ax, positions, colors, markers, s=50, alpha=0.7):
        """Plot particles with different colors and markers for each cluster"""
        if self.clusters is None:
            # If no clusters, plot all particles with same marker
            ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=s, alpha=alpha)
            return
            
        labels = self.clusters['labels']
        
        # Plot each cluster with its specific marker
        for cluster_id in range(self.clusters['n_clusters']):
            mask = labels == cluster_id
            if np.any(mask):
                cluster_positions = positions[mask]
                cluster_colors = colors[mask]
                cluster_marker = markers[mask][0]  # All particles in same cluster have same marker
                
                ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                          c=cluster_colors, marker=cluster_marker, s=s, alpha=alpha,
                          edgecolors='black', linewidths=0.5)
        
        # Plot noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_positions = positions[noise_mask]
            noise_colors = colors[noise_mask]
            ax.scatter(noise_positions[:, 0], noise_positions[:, 1], 
                      c=noise_colors, marker='o', s=s, alpha=alpha,
                      edgecolors='black', linewidths=0.5)
    
    def add_cluster_legend(self, ax, max_inline_clusters=6):
        """Add legend showing cluster colors and markers"""
        if self.clusters is None or self.cluster_centers is None:
            return None
            
        # Create legend elements
        legend_elements = []
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        
        for cluster_id in range(self.clusters['n_clusters']):
            color_idx = cluster_id % len(colors)
            marker_idx = cluster_id % len(marker_styles)
            
            color = colors[color_idx]
            marker = marker_styles[marker_idx]
            
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=color, markersize=8,
                                            markeredgecolor='black', markeredgewidth=0.5,
                                            label=f'Cluster {cluster_id}', linestyle='None'))
        
        # Add noise points to legend if they exist
        if -1 in self.clusters['labels']:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='gray', markersize=8,
                                            markeredgecolor='black', markeredgewidth=0.5,
                                            label='Noise Points', linestyle='None'))
        
        # If too many clusters, return legend elements for separate subplot
        if len(legend_elements) > max_inline_clusters:
            return legend_elements
        else:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            return None
    
    def create_legend_subplot(self, ax, legend_elements):
        """Create a separate subplot for legend when there are too many clusters"""
        ax.axis('off')
        
        # Calculate number of columns for legend layout
        n_items = len(legend_elements)
        n_cols = min(3, n_items)  # Maximum 3 columns
        
        ax.legend(handles=legend_elements, loc='center', ncol=n_cols, 
                 frameon=True, fancybox=True, shadow=True)
        ax.set_title('Cluster Legend', fontsize=12, fontweight='bold')
    
    def trace_particle_trajectories(self):
        """Trace particle trajectories and establish correspondence between initial and final states"""
        print("Tracing particle trajectories...")
        
        if self.sa is None:
            raise ValueError("Please load simulation data first")
            
        # 获取初始和最终位置
        initial_positions = self.sa.totalPositionX[0]
        final_positions = self.sa.totalPositionX[-1]
        
        # 计算轨迹
        trajectories = {
            'initial_positions': initial_positions,
            'final_positions': final_positions,
            'all_positions': self.sa.totalPositionX,
            'all_phases': self.sa.totalPhaseTheta,
            'time_steps': np.arange(self.sa.TNum)
        }
        
        # 计算每个粒子的位移（考虑周期性边界条件）
        # 使用模型的边界长度参数
        boundary_length = self.model.boundaryLength
        displacements = self.calculate_periodic_displacement(initial_positions, final_positions, boundary_length)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        trajectories['displacements'] = displacements
        trajectories['displacement_magnitudes'] = displacement_magnitudes
        
        self.particle_trajectories = trajectories
        print("Trajectory tracing completed")
        
        return trajectories
    
    def plot_cluster_analysis(self, save_fig=True):
        """Plot cluster analysis results"""
        if self.clusters is None or self.cluster_centers is None:
            raise ValueError("Please run cluster detection and center calculation first")
            
        # Check if we need a separate legend subplot
        n_clusters = self.clusters['n_clusters']
        noise_exists = -1 in self.clusters['labels']
        total_legend_items = n_clusters + (1 if noise_exists else 0)
        
        if total_legend_items > 6:
            # Create 2x3 layout with separate legend subplot
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            legend_ax = axes[0, 2]  # Use top-right subplot for legend
        else:
            # Use original 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            legend_ax = None
        
        positions = self.clusters['positions']
        labels = self.clusters['labels']
        phases = self.sa.totalPhaseTheta[self.clusters['time_index']]
        
        # 1. Cluster spatial distribution with different markers
        ax1 = axes[0, 0]
        # Get particle colors and markers for enhanced visualization
        particle_colors, particle_markers = self.assign_particle_colors()
        self.plot_particles_with_markers(ax1, positions, particle_colors, particle_markers)
        
        # Draw cluster centers
        for cluster_id, center_info in self.cluster_centers.items():
            center = center_info['center']
            ax1.plot(center[0], center[1], 'r*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1)
            ax1.annotate(f'C{cluster_id}', (center[0], center[1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_title('Steady-state Cluster Spatial Distribution')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.grid(True, alpha=0.3)
        
        # Handle legend display
        legend_elements = self.add_cluster_legend(ax1)
        if legend_elements is not None and legend_ax is not None:
            self.create_legend_subplot(legend_ax, legend_elements)
        
        # 2. Phase distribution
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], 
                              c=phases, cmap='hsv', s=50, alpha=0.7)
        plt.colorbar(scatter2, ax=ax2, label='Phase')
        ax2.set_title('Particle Phase Distribution')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cluster statistics
        if legend_ax is not None:
            # 2x3 layout
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
            # Hide the bottom-right subplot in 2x3 layout
            axes[1, 2].axis('off')
        else:
            # 2x2 layout
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
            
        cluster_sizes = []
        cluster_coherences = []
        
        for cluster_id in range(self.clusters['n_clusters']):
            size = self.cluster_centers[cluster_id]['particle_count']
            coherence = self.cluster_centers[cluster_id]['phase_coherence']
            cluster_sizes.append(size)
            cluster_coherences.append(coherence)
        
        bars = ax3.bar(range(len(cluster_sizes)), cluster_sizes, 
                      color=[colors[i % len(colors)] for i in range(len(cluster_sizes))])
        ax3.set_title('Particle Count per Cluster')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Particles')
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase coherence
        bars2 = ax4.bar(range(len(cluster_coherences)), cluster_coherences,
                       color=[colors[i % len(colors)] for i in range(len(cluster_coherences))])
        ax4.set_title('Phase Coherence per Cluster')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Phase Coherence')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.save_path, 'cluster_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Cluster analysis plot saved to {self.save_path}/cluster_analysis.png")
        
        plt.show()
    
    def plot_evolution_trajectory(self, save_fig=True):
        """Plot particle evolution trajectory"""
        if self.particle_trajectories is None:
            raise ValueError("Please run trace_particle_trajectories method first")
            
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        initial_pos = self.particle_trajectories['initial_positions']
        final_pos = self.particle_trajectories['final_positions']
        displacements = self.particle_trajectories['displacements']
        displacement_mags = self.particle_trajectories['displacement_magnitudes']
        
        # Get particle colors and markers
        particle_colors, particle_markers = self.assign_particle_colors()
        
        # 1. Initial state
        ax1 = axes[0, 0]
        self.plot_particles_with_markers(ax1, initial_pos, particle_colors, particle_markers)
        ax1.set_title('Initial Random Distribution')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.grid(True, alpha=0.3)
        # Add legend for cluster identification
        self.add_cluster_legend(ax1)
        
        # 2. Final state
        ax2 = axes[0, 1]
        self.plot_particles_with_markers(ax2, final_pos, particle_colors, particle_markers)
        
        # Draw cluster centers
        for cluster_id, center_info in self.cluster_centers.items():
            center = center_info['center']
            ax2.plot(center[0], center[1], 'r*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1)
        
        ax2.set_title('Final Lattice Solution')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.grid(True, alpha=0.3)
        # Add legend for cluster identification
        self.add_cluster_legend(ax2)
        
        # 3. Displacement vectors
        ax3 = axes[0, 2]
        ax3.quiver(initial_pos[:, 0], initial_pos[:, 1], 
                  displacements[:, 0], displacements[:, 1],
                  displacement_mags, cmap='viridis', alpha=0.7)
        ax3.set_title('Particle Displacement Vectors')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.grid(True, alpha=0.3)
        
        # 4. Displacement magnitude distribution
        ax4 = axes[1, 0]
        ax4.hist(displacement_mags, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('Displacement Magnitude Distribution')
        ax4.set_xlabel('Displacement Magnitude')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Trajectory overlay plot (considering periodic boundary)
        ax5 = axes[1, 1]
        boundary_length = self.model.boundaryLength
        
        for i in range(len(initial_pos)):
            # Calculate the corrected final position for visualization
            displacement = self.calculate_periodic_displacement(
                initial_pos[i:i+1], final_pos[i:i+1], boundary_length)[0]
            corrected_final = initial_pos[i] + displacement
            
            ax5.plot([initial_pos[i, 0], corrected_final[0]], 
                    [initial_pos[i, 1], corrected_final[1]], 
                    color=particle_colors[i], alpha=0.5, linewidth=1)
        
        ax5.scatter(initial_pos[:, 0], initial_pos[:, 1], 
                   c='blue', s=30, alpha=0.7, label='Initial Position')
        ax5.scatter(final_pos[:, 0], final_pos[:, 1], 
                   c='red', s=30, alpha=0.7, label='Final Position')
        ax5.set_title('Initial-Final State Correspondence')
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Cluster formation process
        ax6 = axes[1, 2]
        # Calculate cluster count at different time points
        time_points = np.linspace(0, self.sa.TNum-1, 20, dtype=int)
        cluster_counts = []
        
        for t in time_points:
            pos_t = self.sa.totalPositionX[t]
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(pos_t)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            cluster_counts.append(n_clusters)
        
        ax6.plot(time_points, cluster_counts, 'o-', linewidth=2, markersize=6)
        ax6.set_title('Cluster Formation Process')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Number of Clusters')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.save_path, 'evolution_trajectory.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Evolution trajectory plot saved to {self.save_path}/evolution_trajectory.png")
        
        plt.show()
    

    
    def save_analysis_results(self):
        """Save analysis results"""
        print("Saving analysis results...")
        
        results = {
            'model_params': self.model_params,
            'clusters': self.clusters,
            'cluster_centers': self.cluster_centers,
            'particle_trajectories': self.particle_trajectories
        }
        
        # 保存为pickle文件
        with open(os.path.join(self.save_path, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # 保存为JSON文件（部分数据）
        json_results = {
            'model_params': self.model_params,
            'n_clusters': self.clusters['n_clusters'] if self.clusters else None,
            'n_particles': self.model_params['agentsNum'],
            'cluster_summary': {}
        }
        
        if self.cluster_centers:
            for cluster_id, center_info in self.cluster_centers.items():
                json_results['cluster_summary'][str(cluster_id)] = {
                    'center': center_info['center'].tolist(),
                    'particle_count': center_info['particle_count'],
                    'phase_coherence': float(center_info['phase_coherence'])
                }
        
        with open(os.path.join(self.save_path, 'analysis_summary.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Analysis results saved to {self.save_path}")
    
    def run_complete_analysis(self, eps=0.5, min_samples=3):
        """Run complete analysis workflow"""
        print("Starting complete particle evolution trajectory analysis...")
        
        # 1. 加载或运行仿真
        self.load_or_run_simulation()
        
        # 2. 检测团簇
        self.detect_clusters(eps=eps, min_samples=min_samples)
        
        # 3. 计算团簇中心
        self.calculate_cluster_centers()
        
        # 4. 追踪粒子轨迹
        self.trace_particle_trajectories()
        
        # 5. 绘制分析结果
        self.plot_cluster_analysis()
        self.plot_evolution_trajectory()
        
        # 6. 保存结果
        self.save_analysis_results()
        
        print("Analysis completed!")
        
        # Print summary
        print("\n=== Analysis Results Summary ===")
        print(f"Total particles: {self.model_params['agentsNum']}")
        print(f"Detected clusters: {self.clusters['n_clusters']}")
        print(f"Noise points: {self.clusters['n_noise']}")
        
        if self.cluster_centers:
            print("\nCluster information:")
            for cluster_id, center_info in self.cluster_centers.items():
                print(f"  Cluster {cluster_id}: {center_info['particle_count']} particles, "
                      f"phase coherence: {center_info['phase_coherence']:.3f}")


def main():
    """Main function"""
    # Create analyzer instance
    analyzer = ParticleEvolutionAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis(
        eps=0.5,           # DBSCAN clustering parameter
        min_samples=3      # DBSCAN clustering parameter
    )


if __name__ == "__main__":
    main()