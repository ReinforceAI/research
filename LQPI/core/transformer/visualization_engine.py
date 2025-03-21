import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

class VisualizationEngine:
    """
    Creates visualizations for quantum field-like properties of activations.
    
    This class is responsible for:
    1. Creating visualizations of activation spaces
    2. Plotting transition trajectories
    3. Visualizing dimensional hierarchies
    4. Creating animations of state transitions
    5. Visualizing topological features
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any] = None, 
                logger: Optional[logging.Logger] = None):
        """
        Initialize the visualization engine.
        
        Args:
            output_dir: Directory to save visualization files
            config: Configuration for visualizations
            logger: Logger instance for detailed logging
        """
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logger or logging.getLogger('visualization_engine')
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom color maps for specific visualizations
        self.cmap_personality = plt.cm.viridis
        self.cmap_transitions = plt.cm.plasma
        self.cmap_heatmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        # Custom field color map (blues)
        self.cmap_field = LinearSegmentedColormap.from_list('field_blues', [
            '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
            '#4292c6', '#2171b5', '#08519c', '#08306b'
        ])
        
        # Initialize plot counter for unique filenames
        self.plot_counter = 0
        
        self.logger.info(f"VisualizationEngine initialized with output_dir: {output_dir}")
    
    def plot_activation_space(self, embeddings: np.ndarray, labels: List[str], 
                            title: str = "Activation Space", dim: int = 2) -> str:
        """
        Plot activation space embeddings.
        
        Args:
            embeddings: Embedded activation vectors [samples, 2 or 3]
            labels: Labels for each point
            title: Plot title
            dim: Dimensionality of the plot (2 or 3)
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Plotting {dim}D activation space with {len(labels)} points")
        
        try:
            # Convert labels to numeric for coloring
            unique_labels = list(set(labels))
            label_indices = [unique_labels.index(label) for label in labels]
            
            fig = plt.figure(figsize=(12, 10))
            
            if dim == 3 and embeddings.shape[1] >= 3:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(
                    embeddings[:, 0], 
                    embeddings[:, 1], 
                    embeddings[:, 2],
                    c=label_indices, 
                    cmap=self.cmap_personality,
                    s=100, 
                    alpha=0.8
                )
                ax.set_zlabel('Dimension 3')
            else:
                ax = fig.add_subplot(111)
                scatter = ax.scatter(
                    embeddings[:, 0], 
                    embeddings[:, 1], 
                    c=label_indices, 
                    cmap=self.cmap_personality,
                    s=100, 
                    alpha=0.8
                )
            
            # Add labels
            if len(unique_labels) <= 10:  # Only add legend for a reasonable number of labels
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=self.cmap_personality(unique_labels.index(label) / max(1, len(unique_labels) - 1)), 
                                            markersize=10, label=label) 
                                for label in unique_labels]
                ax.legend(handles=legend_elements, loc='best')
            
            # For better identification, add text labels to points
            if len(labels) <= 20:  # Only add text labels for a small number of points
                for i, (x, y) in enumerate(embeddings[:, :2]):
                    if dim == 3:
                        ax.text(x, y, embeddings[i, 2], labels[i], fontsize=9)
                    else:
                        ax.text(x, y, labels[i], fontsize=9)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"activation_space_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Activation space plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to plot activation space")
            self.logger.error(traceback.format_exc())
            raise
    
    def plot_transition_trajectory(self, trajectory: np.ndarray, labels: Optional[List[str]] = None,
                                 title: str = "Transition Trajectory") -> str:
        """
        Plot a trajectory through activation space.
        
        Args:
            trajectory: Sequence of points in activation space [steps, 2 or 3]
            labels: Optional labels for each point
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Plotting transition trajectory with {trajectory.shape[0]} points")
        
        try:
            dim = trajectory.shape[1]
            if dim < 2:
                self.logger.error(f"Trajectory has insufficient dimensions: {dim}")
                raise ValueError("Trajectory must have at least 2 dimensions")
            
            fig = plt.figure(figsize=(12, 10))
            
            if dim >= 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.6)
                scatter = ax.scatter(
                    trajectory[:, 0], 
                    trajectory[:, 1], 
                    trajectory[:, 2],
                    c=range(len(trajectory)), 
                    cmap=self.cmap_transitions,
                    s=100, 
                    alpha=0.8
                )
                ax.set_zlabel('Dimension 3')
            else:
                ax = fig.add_subplot(111)
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6)
                scatter = ax.scatter(
                    trajectory[:, 0], 
                    trajectory[:, 1],
                    c=range(len(trajectory)), 
                    cmap=self.cmap_transitions,
                    s=100, 
                    alpha=0.8
                )
            
            # Add colorbar to show progression
            cbar = plt.colorbar(scatter)
            cbar.set_label('Transition Step')
            
            # Add labels
            if labels is not None and len(labels) <= 20:
                for i, (x, y) in enumerate(trajectory[:, :2]):
                    if dim >= 3:
                        ax.text(x, y, trajectory[i, 2], labels[i], fontsize=9)
                    else:
                        ax.text(x, y, labels[i], fontsize=9)
            
            # Mark start and end points
            if dim >= 3:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=200, alpha=0.8, marker='*')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=200, alpha=0.8, marker='*')
            else:
                ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=200, alpha=0.8, marker='*')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=200, alpha=0.8, marker='*')
            
            # Add markers for detected jumps
            if 'jumps' in self.config:
                jump_indices = self.config['jumps']
                for idx in jump_indices:
                    if 0 <= idx < len(trajectory):
                        if dim >= 3:
                            ax.scatter(trajectory[idx, 0], trajectory[idx, 1], trajectory[idx, 2], 
                                     color='yellow', s=150, alpha=0.8, marker='o', edgecolors='black')
                        else:
                            ax.scatter(trajectory[idx, 0], trajectory[idx, 1], 
                                     color='yellow', s=150, alpha=0.8, marker='o', edgecolors='black')
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"transition_trajectory_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Transition trajectory plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to plot transition trajectory")
            self.logger.error(traceback.format_exc())
            raise
    
    def create_attention_heatmap(self, attention_maps: Dict[str, np.ndarray], 
                                   layer_name: str = None) -> str:
        """
        Create a heatmap visualization of attention patterns.
        
        Args:
            attention_maps: Dictionary of attention maps by layer
            layer_name: Specific layer to visualize (if None, select the first available)
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating attention heatmap visualization")
        
        try:
            # Select layer to visualize
            if layer_name is not None and layer_name in attention_maps:
                attention = attention_maps[layer_name]
            elif attention_maps:
                layer_name = list(attention_maps.keys())[0]
                attention = attention_maps[layer_name]
            else:
                self.logger.error("No attention maps provided for visualization")
                raise ValueError("No attention maps provided")
            
            # Process attention tensor
            # Attention shape could be [batch, heads, seq_from, seq_to]
            # or [batch, seq_from, seq_to] depending on model
            if len(attention.shape) == 4:
                # Average over heads
                attention = np.mean(attention, axis=1)
            
            # If batch dimension, use first batch
            if len(attention.shape) == 3 and attention.shape[0] > 1:
                attention = attention[0]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot heatmap
            im = ax.imshow(attention, cmap=self.cmap_heatmap)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
            
            # Set labels
            ax.set_xlabel("Token Position (Target)")
            ax.set_ylabel("Token Position (Source)")
            
            # Set title
            ax.set_title(f"Attention Pattern: {layer_name}")
            
            # Add grid
            ax.grid(False)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"attention_heatmap_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Attention heatmap saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create attention heatmap")
            self.logger.error(traceback.format_exc())
            raise
    
    def plot_stability_curves(self, stability_data: Dict[str, Any],
                                    title: str = "Stability Analysis") -> str:
        """
        Create  visualization of stability curves with transition detection.
        
        Args:
            stability_data: Dictionary containing stability analysis results
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating  stability curve visualization")
        
        try:
            # Extract relevant data
            levels = stability_data.get("levels", [])
            stability_scores = stability_data.get("stability_scores", [])
            critical_level = stability_data.get("critical_level")
            distance_clusters = stability_data.get("distance_clusters", {})
            sharp_transitions = stability_data.get("sharp_transitions", [])
            
            # Create figure with 2 subplots
            fig, axs = plt.subplots(2, 1, figsize=(12, 14))
            
            # Plot 1: Stability curve with critical level and sharp transitions
            axs[0].plot(levels, stability_scores, 'o-', color='blue', alpha=0.7, linewidth=2)
            axs[0].set_title("Stability Under Perturbation")
            axs[0].set_xlabel("Perturbation Level")
            axs[0].set_ylabel("Stability Score")
            axs[0].grid(True, linestyle='--', alpha=0.7)
            
            # Mark critical level
            if critical_level is not None and critical_level in levels:
                idx = levels.index(critical_level)
                if idx < len(stability_scores):
                    axs[0].axvline(x=critical_level, color='red', linestyle='--', 
                                label=f"Critical Level: {critical_level:.2f}")
                    axs[0].plot(critical_level, stability_scores[idx], 'ro', ms=10)
            
            # Mark sharp transitions
            for idx in sharp_transitions:
                if 0 <= idx < len(levels):
                    axs[0].plot(levels[idx], stability_scores[idx], 'D', color='orange', 
                            markersize=10, label=f"Sharp Transition" if idx == sharp_transitions[0] else "")
            
            # Add bands for quantum vs. classical behavior
            axs[0].axhspan(0.7, 1.0, alpha=0.2, color='green', label='Quantum-like Protection')
            axs[0].axhspan(0.3, 0.7, alpha=0.2, color='yellow', label='Intermediate')
            axs[0].axhspan(0.0, 0.3, alpha=0.2, color='red', label='Classical-like')
            
            axs[0].legend()
            
            # Plot 2: Rate of change (derivative) to highlight transitions
            if len(levels) > 1 and len(stability_scores) > 1:
                derivatives = np.diff(stability_scores) / np.diff(levels)
                mid_levels = [(levels[i] + levels[i+1])/2 for i in range(len(levels)-1)]
                
                axs[1].plot(mid_levels, derivatives, 'o-', color='purple', alpha=0.8)
                axs[1].set_title("Rate of Stability Change")
                axs[1].set_xlabel("Perturbation Level")
                axs[1].set_ylabel("dStability/dPerturbation")
                axs[1].grid(True, linestyle='--', alpha=0.7)
                
                # Mark critical derivative threshold
                axs[1].axhline(y=-0.5, color='red', linestyle='--', 
                            label="Quantum Transition Threshold")
                
                # Highlight sharp transitions
                sharp_indices = [i for i, d in enumerate(derivatives) if d < -0.5]
                if sharp_indices:
                    axs[1].scatter([mid_levels[i] for i in sharp_indices], 
                                [derivatives[i] for i in sharp_indices],
                                color='red', s=100, zorder=5, 
                                label="Quantum-like Transitions")
                
                axs[1].legend()
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"stability_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"stability visualization saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create  stability visualization")
            self.logger.error(traceback.format_exc())
            raise
    
    def visualize_dimensional_hierarchy(self, hierarchy_data: Dict[str, Any],
                                        title: str = "Dimensional Hierarchy") -> str:
        """
        Visualize dimensional hierarchy and compression with  golden ratio detection.
        
        Args:
            hierarchy_data: Dictionary containing dimensional analysis results
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Visualizing dimensional hierarchy with golden ratio analysis")
        
        try:
            # Extract relevant data
            eigenvalues = hierarchy_data.get('eigenvalues', [])
            cumulative_variance = hierarchy_data.get('cumulative_variance', [])
            dim_thresholds = hierarchy_data.get('dimension_thresholds', {})
            compression_cascade = hierarchy_data.get('compression_cascade', [])
            golden_patterns = hierarchy_data.get('golden_ratio_patterns', [])
            
            # Create figure with 3 subplots - add an additional plot for golden ratio analysis
            fig, axs = plt.subplots(4, 1, figsize=(12, 20))
            
            # Plot 1: Eigenvalue spectrum
            if eigenvalues:
                # Plot first 100 eigenvalues or fewer if less available
                n_eigs = min(100, len(eigenvalues))
                axs[0].plot(range(1, n_eigs+1), eigenvalues[:n_eigs], 'o-', color='blue', alpha=0.7)
                axs[0].set_title("Eigenvalue Spectrum")
                axs[0].set_xlabel("Principal Component Index")
                axs[0].set_ylabel("Eigenvalue")
                axs[0].set_yscale('log')  # Log scale often better for eigenvalues
                axs[0].grid(True, linestyle='--', alpha=0.7)
                
                # Mark golden ratio patterns if available
                if golden_patterns:
                    # Extract indices from patterns
                    for pattern in golden_patterns:
                        indices = pattern.get('indices', '').split(':')
                        if len(indices) == 2 and indices[0].isdigit() and indices[1].isdigit():
                            i, j = int(indices[0]), int(indices[1])
                            if i < n_eigs and j < n_eigs:
                                # Plot the pair of eigenvalues
                                axs[0].plot([i+1, j+1], [eigenvalues[i], eigenvalues[j]], 'ro-', 
                                            linewidth=2, alpha=0.7)
                                
                                # Add annotation
                                ratio_type = pattern.get("type", "")
                                ratio_value = eigenvalues[i] / eigenvalues[j]
                                axs[0].annotate(f"{ratio_type}: {ratio_value:.3f}", 
                                            xy=((i+j+2)/2, np.sqrt(eigenvalues[i]*eigenvalues[j])),
                                            xytext=(0, 10), textcoords='offset points',
                                            ha='center', va='bottom',
                                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
                
            # Plot 2: Cumulative variance
            if cumulative_variance:
                axs[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-', color='green', alpha=0.7)
                axs[1].set_title("Cumulative Explained Variance")
                axs[1].set_xlabel("Number of Components")
                axs[1].set_ylabel("Cumulative Explained Variance")
                axs[1].grid(True, linestyle='--', alpha=0.7)
                
                # Mark dimension thresholds
                for name, dim in dim_thresholds.items():
                    if dim <= len(cumulative_variance):
                        axs[1].axvline(x=dim, color='red', linestyle='--', 
                                    label=f"{name} Threshold: {dim} dimensions")
                
                axs[1].legend()
                
            # Plot 3: Golden Ratio Analysis - NEW
            if eigenvalues and len(eigenvalues) > 1:
                # Calculate eigenvalue ratios
                ratios = np.array(eigenvalues[:-1]) / np.array(eigenvalues[1:])
                
                # Plot ratio distribution
                axs[2].plot(range(1, len(ratios)+1), ratios, 'o-', color='purple', alpha=0.7)
                axs[2].set_title("Eigenvalue Ratios (Adjacent Components)")
                axs[2].set_xlabel("Component Pair Index")
                axs[2].set_ylabel("Ratio λᵢ/λᵢ₊₁")
                axs[2].grid(True, linestyle='--', alpha=0.7)
                
                # Mark golden ratio values
                golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
                golden_ratio_inverse = 1 / golden_ratio  # ≈ 0.618...
                
                axs[2].axhline(y=golden_ratio, color='red', linestyle='--', 
                            label=f"φ = {golden_ratio:.4f}")
                axs[2].axhline(y=golden_ratio_inverse, color='orange', linestyle='--', 
                            label=f"1/φ = {golden_ratio_inverse:.4f}")
                axs[2].axhline(y=golden_ratio**2, color='green', linestyle='--', 
                            label=f"φ² = {golden_ratio**2:.4f}")
                axs[2].axhline(y=1/golden_ratio**2, color='purple', linestyle='--', 
                            label=f"1/φ² = {1/golden_ratio**2:.4f}")
                
                axs[2].legend()
                
            # Plot 4: Compression cascade
            if compression_cascade:
                dims = [item.get('from_dim', 0) for item in compression_cascade]
                dims.append(compression_cascade[-1].get('to_dim', 0))
                
                info_preservation = [1.0]  # Start with 100% preservation
                for item in compression_cascade:
                    info_preservation.append(item.get('info_preservation', 0))
                
                axs[3].plot(dims, info_preservation, 'o-', color='blue', alpha=0.7)
                axs[3].set_title("Information Preservation Across Dimensions")
                axs[3].set_xlabel("Dimensions")
                axs[3].set_ylabel("Information Preservation Ratio")
                axs[3].grid(True, linestyle='--', alpha=0.7)
                
                # Set x-axis to show the exact dimensions
                axs[3].set_xticks(dims)
                
                # Annotate compression ratios and golden ratio alignment
                for i, item in enumerate(compression_cascade):
                    ratio = item.get('compression_ratio', 0)
                    quantum_efficiency = item.get('quantum_efficiency', 0)
                    x1, x2 = dims[i], dims[i+1]
                    y = (info_preservation[i] + info_preservation[i+1]) / 2
                    
                    # Create annotation text with quantum efficiency if available
                    if 'quantum_efficiency' in item:
                        annotation = f"Ratio: {ratio:.2f}\nEfficiency: {quantum_efficiency:.2f}"
                    else:
                        annotation = f"Ratio: {ratio:.2f}"
                    
                    axs[3].annotate(annotation, xy=((x1+x2)/2, y), 
                                xytext=(0, 10), textcoords='offset points',
                                ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
                    
                    # Add golden ratio alignment indicator if available
                    if 'closest_gr_name' in item and 'gr_deviation' in item:
                        gr_name = item.get('closest_gr_name', '')
                        gr_deviation = item.get('gr_deviation', 1.0)
                        
                        if gr_deviation < 0.2:  # Show only for reasonably close alignment
                            alignment_text = f"≈ {gr_name} ({gr_deviation:.1%} dev)"
                            axs[3].annotate(alignment_text, xy=((x1+x2)/2, y), 
                                        xytext=(0, -20), textcoords='offset points',
                                        ha='center', va='top',
                                        bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.4))
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"dimensional_hierarchy_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Dimensional hierarchy visualization saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to visualize dimensional hierarchy")
            self.logger.error(traceback.format_exc())
            raise

    def visualize_nonlinear_trajectory(self, topo_data: Dict[str, Any],
                                    title: str = "Non-Linear Trajectory Analysis") -> str:
        """
        Visualize non-linear trajectory patterns in activation space, adapting to their natural structure.
        
        Args:
            topo_data: Dictionary containing trajectory data and non-linearity metrics
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating visualization based on natural trajectory properties")
        
        try:
            # Log data structure to understand natural properties
            self.logger.info(f"Examining natural data structure:")
            if "trajectory" in topo_data:
                traj_meta = topo_data["trajectory"]
                self.logger.info(f"  - Trajectory steps: {len(traj_meta)}")
                if traj_meta:
                    sample = traj_meta[0]
                    self.logger.info(f"  - Sample vector properties: {sample}")
            
            self.logger.info(f"  - Available keys: {list(topo_data.keys())}")
            
            # Extract core metrics
            angle_changes = topo_data.get("angle_changes", [])
            nonlinearity_score = topo_data.get("nonlinearity_score", 0)
            
            # Create adaptable visualization based on available data
            fig = plt.figure(figsize=(12, 14))
            
            # Determine layout based on available data
            plot_trajectory = False
            
            # Check if we have PCA projection for visualization
            if "pca_projection" in topo_data and len(topo_data["pca_projection"]) > 1:
                self.logger.info("Using PCA projection for trajectory visualization")
                plot_trajectory = True
                projection = topo_data["pca_projection"]
                projection_type = "PCA"
                explained_variance = topo_data.get("pca_explained_variance", [0, 0])
            # Check for UMAP projection if available
            elif "umap_projection" in topo_data and len(topo_data["umap_projection"]) > 1:
                self.logger.info("Using UMAP projection for trajectory visualization")
                plot_trajectory = True
                projection = topo_data["umap_projection"]
                projection_type = "UMAP"
                explained_variance = None
            # Check if raw trajectory vectors are available with at least 2D
            elif "trajectory" in topo_data and topo_data["trajectory"]:
                # Try to create a simple 2D visualization if possible
                self.logger.info("Checking if trajectory can be naturally visualized in 2D")
                
                # This is an adaptive approach - we'll extract what we can
                try:
                    # See if we have raw vectors to work with
                    if isinstance(topo_data["trajectory"], list) and all(isinstance(v, np.ndarray) for v in topo_data["trajectory"]):
                        vectors = topo_data["trajectory"]
                        if vectors and vectors[0].shape[0] >= 2:
                            self.logger.info(f"Using first two dimensions of {vectors[0].shape[0]}D vectors")
                            plot_trajectory = True
                            projection = np.array([[v[0], v[1]] for v in vectors])
                            projection_type = "First2D"
                            explained_variance = None
                    # Check for trajectory metadata format
                    elif isinstance(topo_data["trajectory"], list) and all(isinstance(v, dict) for v in topo_data["trajectory"]):
                        # We have metadata dictionary format
                        vectors = topo_data["trajectory"]
                        if any(v.get("flattened_shape", (0,))[0] >= 2 for v in vectors):
                            self.logger.info("Computing ad-hoc 2D projection from high-dimensional data")
                            # Create simple projection from dimensionality reduction
                            from sklearn.decomposition import PCA
                            try:
                                # Build vectors from metadata if actual vectors aren't stored
                                # This is just for visualization purposes
                                positions = np.zeros((len(vectors), 2))
                                for i in range(1, len(vectors)):
                                    # Cumulative position based on step number
                                    positions[i, 0] = positions[i-1, 0] + 0.1 * np.cos(vectors[i].get("step", i) * 0.5)
                                    positions[i, 1] = positions[i-1, 1] + 0.1 * np.sin(vectors[i].get("step", i) * 0.5)
                                    
                                    # Scale by vector magnitude for visual effect
                                    magnitude = vectors[i].get("magnitude", 1.0)
                                    positions[i, 0] *= magnitude / 10
                                    positions[i, 1] *= magnitude / 10
                                
                                plot_trajectory = True
                                projection = positions
                                projection_type = "Symbolic"
                                explained_variance = None
                            except Exception as e:
                                self.logger.warning(f"Failed to create projection: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Could not extract trajectory for visualization: {str(e)}")
            
            # Set up subplot layout
            if plot_trajectory:
                gs = plt.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)
                ax_traj = fig.add_subplot(gs[0])
                ax_angles = fig.add_subplot(gs[1])
            else:
                # Just show angle changes
                gs = plt.GridSpec(1, 1, figure=fig)
                ax_angles = fig.add_subplot(gs[0])
            
            # Plot trajectory if we have data
            if plot_trajectory:
                x = projection[:, 0]
                y = projection[:, 1]
                
                # Plot trajectory path
                ax_traj.plot(x, y, 'o-', color='blue', alpha=0.7, linewidth=2)
                
                # Add arrows to show direction
                for i in range(len(x)-1):
                    ax_traj.arrow(x[i], y[i], (x[i+1]-x[i])*0.8, (y[i+1]-y[i])*0.8, 
                            head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
                
                # Mark start and end points
                ax_traj.plot(x[0], y[0], 'go', markersize=10, label='Start')
                ax_traj.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
                
                # Add appropriate title based on projection type
                if projection_type == "PCA" and explained_variance is not None:
                    title_text = f"Trajectory Visualization (PCA Projection, {explained_variance[0]:.1%} + {explained_variance[1]:.1%} variance)"
                else:
                    title_text = f"Trajectory Visualization ({projection_type})"
                    
                ax_traj.set_title(title_text)
                ax_traj.set_xlabel("Dimension 1")
                ax_traj.set_ylabel("Dimension 2")
                ax_traj.legend()
                ax_traj.grid(True, linestyle='--', alpha=0.7)
            
            # Plot angular changes
            if angle_changes:
                # Plot angular changes
                steps = list(range(1, len(angle_changes)+1))
                ax_angles.plot(steps, angle_changes, 'o-', color='purple', alpha=0.7)
                ax_angles.set_title("Directional Change in Trajectory")
                ax_angles.set_xlabel("Sequence Step")
                ax_angles.set_ylabel("Angular Change (degrees)")
                ax_angles.grid(True, linestyle='--', alpha=0.7)
                
                # Add reference lines
                ax_angles.axhline(y=90, color='red', linestyle='--', label="90° (Strong Non-linearity)")
                ax_angles.axhline(y=45, color='orange', linestyle='--', label="45° (Moderate Non-linearity)")
                ax_angles.axhline(y=10, color='green', linestyle='--', label="10° (Near Linear)")
                
                # Add non-linearity score
                ax_angles.text(0.02, 0.95, f"Non-linearity Score: {nonlinearity_score:.3f}", 
                        transform=ax_angles.transAxes, fontsize=12,
                        bbox=dict(facecolor='yellow', alpha=0.2))
                
                ax_angles.legend()
            else:
                ax_angles.text(0.5, 0.5, "Angular change data not available", 
                        ha='center', va='center', fontsize=12)
                ax_angles.axis('off')
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"nonlinear_trajectory_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Natural trajectory visualization saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create natural trajectory visualization")
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_eigenvalue_distribution(self, eigenvalues: np.ndarray,
                                    title: str = "Eigenvalue Distribution") -> str:
        """
        Plot eigenvalue distribution and associated metrics.
        
        Args:
            eigenvalues: Array of eigenvalues
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Plotting eigenvalue distribution")
        
        try:
            # Sort eigenvalues in descending order
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Create figure with 3 subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Eigenvalue spectrum
            axs[0, 0].plot(range(1, len(sorted_eigenvalues)+1), sorted_eigenvalues, 'o-', color='blue', alpha=0.7)
            axs[0, 0].set_title("Eigenvalue Spectrum")
            axs[0, 0].set_xlabel("Index")
            axs[0, 0].set_ylabel("Eigenvalue")
            axs[0, 0].set_yscale('log')  # Log scale often better for eigenvalues
            axs[0, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Eigenvalue ratios
            if len(sorted_eigenvalues) > 1:
                ratios = sorted_eigenvalues[:-1] / sorted_eigenvalues[1:]
                axs[0, 1].plot(range(1, len(ratios)+1), ratios, 'o-', color='green', alpha=0.7)
                axs[0, 1].set_title("Eigenvalue Ratios")
                axs[0, 1].set_xlabel("Index")
                axs[0, 1].set_ylabel("Ratio λᵢ/λᵢ₊₁")
                axs[0, 1].grid(True, linestyle='--', alpha=0.7)
                
                # Mark golden ratio
                golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
                axs[0, 1].axhline(y=golden_ratio, color='red', linestyle='--', 
                                label=f"Golden Ratio: {golden_ratio:.3f}")
                axs[0, 1].axhline(y=1/golden_ratio, color='orange', linestyle='--', 
                                label=f"Inverse Golden Ratio: {1/golden_ratio:.3f}")
                axs[0, 1].legend()
            
            # Plot 3: Eigenvalue spacings
            if len(sorted_eigenvalues) > 1:
                spacings = np.diff(sorted_eigenvalues)
                mean_spacing = np.mean(spacings)
                
                # Normalize spacings
                if mean_spacing > 0:
                    normalized_spacings = spacings / mean_spacing
                    
                    # Histogram of normalized spacings
                    hist, bin_edges = np.histogram(normalized_spacings, bins=20, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    axs[1, 0].bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
                                color='purple', alpha=0.7)
                    axs[1, 0].set_title("Normalized Eigenvalue Spacing Distribution")
                    axs[1, 0].set_xlabel("Normalized Spacing s")
                    axs[1, 0].set_ylabel("P(s)")
                    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
                    
                    # Plot theoretical distributions
                    s_values = np.linspace(0, 3, 100)
                    
                    # Wigner surmise (GOE) for quantum chaotic systems
                    wigner = (np.pi/2) * s_values * np.exp(-np.pi * s_values**2 / 4)
                    
                    # Poisson for classical systems
                    poisson = np.exp(-s_values)
                    
                    axs[1, 0].plot(s_values, wigner, 'r-', label='Wigner (Quantum)')
                    axs[1, 0].plot(s_values, poisson, 'g-', label='Poisson (Classical)')
                    axs[1, 0].legend()
            
            # Plot 4: Participation ratio
            if len(sorted_eigenvalues) > 0:
                total = np.sum(np.abs(sorted_eigenvalues))
                if total > 0:
                    normalized_eigenvalues = sorted_eigenvalues / total
                    participation_ratio = 1.0 / np.sum(normalized_eigenvalues**2)
                    
                    axs[1, 1].bar([0], [participation_ratio], width=0.5, 
                                color='orange', alpha=0.7)
                    axs[1, 1].set_title("Participation Ratio")
                    axs[1, 1].set_ylabel("Effective Dimensions")
                    axs[1, 1].set_xlim(-1, 1)
                    axs[1, 1].set_xticks([0])
                    axs[1, 1].set_xticklabels([''])
                    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
                    
                    # Add text annotation
                    axs[1, 1].text(0, participation_ratio/2, f"{participation_ratio:.2f}", 
                                ha='center', va='center', fontsize=12)
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"eigenvalue_distribution_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Eigenvalue distribution plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to plot eigenvalue distribution")
            self.logger.error(traceback.format_exc())
            raise
    
    def create_transition_animation(self, activation_sequence: np.ndarray,
                                title: str = "State Transition Animation") -> str:
        """
        Create an animation of state transitions.
        
        Args:
            activation_sequence: Sequence of activation states [timesteps, 2]
            title: Animation title
            
        Returns:
            Path to the saved animation
        """
        self.logger.info("Creating transition animation")
        
        try:
            # Ensure activation_sequence is a numpy array
            activation_sequence = np.array(activation_sequence)
            
            if activation_sequence.shape[1] != 2:
                self.logger.error(f"Animation requires 2D points, but got {activation_sequence.shape[1]}D")
                raise ValueError("Animation requires 2D points")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Compute limits with some padding
            x_min, x_max = activation_sequence[:, 0].min(), activation_sequence[:, 0].max()
            y_min, y_max = activation_sequence[:, 1].min(), activation_sequence[:, 1].max()
            
            # Add padding
            x_pad = (x_max - x_min) * 0.1
            y_pad = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            
            # Set up animation elements
            trajectory, = ax.plot([], [], 'b-', alpha=0.6)
            point, = ax.plot([], [], 'ro', markersize=10)
            
            # Add title
            ax.set_title(title)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add mix ratio information if available
            mix_ratio_text = None
            if 'mix_ratios' in self.config and len(self.config.get('mix_ratios', [])) > 0:
                mix_ratio_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
            
            # Initialization function
            def init():
                trajectory.set_data([], [])
                point.set_data([], [])
                if mix_ratio_text:
                    mix_ratio_text.set_text('')
                return [trajectory, point] + ([mix_ratio_text] if mix_ratio_text else [])
            
            # Animation function
            def animate(i):
                # Get x and y coordinates as sequences (lists)
                x_data = [activation_sequence[i, 0]]  # Wrap in list to make it a sequence
                y_data = [activation_sequence[i, 1]]  # Wrap in list to make it a sequence
                
                # Update trajectory and point
                trajectory.set_data(activation_sequence[:i+1, 0], activation_sequence[:i+1, 1])
                point.set_data(x_data, y_data)  # Now passing sequences
                
                # Update mix ratio text if available
                if mix_ratio_text and 'mix_ratios' in self.config:
                    mix_ratios = self.config['mix_ratios']
                    if i < len(mix_ratios):
                        mix_ratio_text.set_text(f"Mix Ratio: {mix_ratios[i]:.2f}")
                
                return [trajectory, point] + ([mix_ratio_text] if mix_ratio_text else [])
            
            # Create animation
            anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                        frames=len(activation_sequence),
                                        interval=200, blit=True)
            
            # Save animation
            self.plot_counter += 1
            filename = f"transition_animation_{self.plot_counter:03d}.gif"
            filepath = os.path.join(self.output_dir, filename)
            
            anim.save(filepath, writer='pillow', fps=5)
            plt.close(fig)
            
            self.logger.info(f"Transition animation saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create transition animation")
            self.logger.error(traceback.format_exc())
            raise


    def visualize_topological_protection(self, topo_data: Dict[str, Any],
                                    title: str = "Topological Protection Analysis") -> str:
        """
        Visualize topological protection metrics.
        
        Args:
            topo_data: Dictionary containing topological protection metrics
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating topological protection visualization")
        
        try:
            # Extract topological features
            spectral_gap = topo_data.get("spectral_gap", 0)
            topological_charge = topo_data.get("estimated_topological_charge", 0)
            betti_0 = topo_data.get("betti_0", 0)
            betti_1 = topo_data.get("betti_1", 0)
            clustering_coefficient = topo_data.get("clustering_coefficient", 0)
            
            # Create figure with multiple panels
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # Panel 1: Topological charge visualization
            axs[0, 0].bar(['Topological Charge'], [min(topological_charge/1000, 10)], color='blue', alpha=0.7)
            axs[0, 0].set_title(f"Estimated Topological Charge: {topological_charge:.1f}")
            axs[0, 0].set_ylabel("Charge (thousands)")
            axs[0, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Add reference lines for quantum-like thresholds
            axs[0, 0].axhline(y=1, color='green', linestyle='--', label="Quantum Threshold")
            
            # Panel 2: Spectral gap visualization
            axs[0, 1].bar(['Spectral Gap'], [spectral_gap], color='purple', alpha=0.7)
            axs[0, 1].set_title(f"Spectral Gap: {spectral_gap:.4f}")
            axs[0, 1].set_ylabel("Gap Size")
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)
            
            # Add reference lines
            axs[0, 1].axhline(y=0.5, color='green', linestyle='--', label="High Protection")
            axs[0, 1].axhline(y=0.1, color='orange', linestyle='--', label="Moderate Protection")
            axs[0, 1].legend()
            
            # Panel 3: Betti numbers
            axs[1, 0].bar(['Betti-0', 'Betti-1'], [betti_0, betti_1], color=['blue', 'orange'], alpha=0.7)
            axs[1, 0].set_title(f"Betti Numbers (β₀={betti_0}, β₁={betti_1})")
            axs[1, 0].set_ylabel("Count")
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Add annotation for golden ratio (if applicable)
            if betti_0 > 0:
                ratio = betti_1 / betti_0
                golden_ratio = (1 + np.sqrt(5)) / 2
                if abs(ratio - golden_ratio) < 0.2:
                    axs[1, 0].text(0.5, 0.9, f"Ratio ≈ φ ({ratio:.3f})", transform=axs[1, 0].transAxes,
                                ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.5))
            
            # Panel 4: Protection summary
            protection_level = "High" if spectral_gap > 0.5 and topological_charge > 1000 else \
                            "Moderate" if spectral_gap > 0.1 and topological_charge > 100 else "Low"
            
            protection_color = {'High': 'green', 'Moderate': 'orange', 'Low': 'red'}[protection_level]
            
            axs[1, 1].text(0.5, 0.6, f"Topological Protection:\n{protection_level}",
                        ha='center', va='center', fontsize=16,
                        bbox=dict(facecolor=protection_color, alpha=0.3))
            
            axs[1, 1].text(0.5, 0.3, f"Clustering: {clustering_coefficient:.3f}\nField Score: {min(topological_charge/1000, 10):.1f}",
                        ha='center', va='center', fontsize=12)
            
            axs[1, 1].axis('off')
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"topological_protection_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Topological protection visualization saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create topological protection visualization")
            self.logger.error(traceback.format_exc())
            raise


    def plot_personality_protection_comparison(self, personality_metrics: Dict[str, Dict],
                                          title: str = "Personality Protection Comparison") -> str:
        """
        Create a comparison visualization of topological protection across personalities.
        
        Args:
            personality_metrics: Dictionary of metrics by personality
            title: Plot title
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating personality protection comparison visualization")
        
        try:
            # Extract relevant metrics
            personalities = list(personality_metrics.keys())
            stability_scores = []
            critical_levels = []
            quantum_scores = []
            
            for name, metrics in personality_metrics.items():
                stability_scores.append(metrics.get("overall_stability", 0))
                critical_levels.append(metrics.get("critical_level", 0))
                quantum_scores.append(metrics.get("quantum_protection_score", 0))
            
            # Create figure with 3 subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 16))
            
            # Plot 1: Overall stability comparison
            stability_bars = axs[0].bar(personalities, stability_scores, color='blue', alpha=0.7)
            axs[0].set_title("Overall Stability by Personality")
            axs[0].set_ylabel("Stability Score")
            axs[0].set_ylim(0, 1)
            axs[0].grid(True, linestyle='--', alpha=0.7)
            
            # Add reference lines
            axs[0].axhline(y=0.7, color='green', linestyle='--', label="High Stability")
            axs[0].axhline(y=0.4, color='orange', linestyle='--', label="Moderate Stability")
            axs[0].legend()
            
            # Rotate x-labels for readability
            plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Plot 2: Critical perturbation levels
            critical_bars = axs[1].bar(personalities, critical_levels, color='purple', alpha=0.7)
            axs[1].set_title("Critical Perturbation Level by Personality")
            axs[1].set_ylabel("Perturbation Level")
            axs[1].set_ylim(0, 1)
            axs[1].grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-labels for readability
            plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Plot 3: Quantum protection scores
            quantum_bars = axs[2].bar(personalities, quantum_scores, color='green', alpha=0.7)
            axs[2].set_title("Quantum Protection Score by Personality")
            axs[2].set_ylabel("Quantum Score")
            axs[2].set_ylim(0, 1)
            axs[2].grid(True, linestyle='--', alpha=0.7)
            
            # Add reference lines
            axs[2].axhline(y=0.7, color='blue', linestyle='--', label="Strong Quantum Evidence")
            axs[2].axhline(y=0.4, color='orange', linestyle='--', label="Moderate Quantum Evidence")
            axs[2].legend()
            
            # Rotate x-labels for readability
            plt.setp(axs[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
            
            # Save figure
            self.plot_counter += 1
            filename = f"personality_protection_comparison_{self.plot_counter:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Personality protection comparison saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error("Failed to create personality protection comparison")
            self.logger.error(traceback.format_exc())
            raise
        
    def save_plot(self, fig, filename: str) -> str:
        """
        Save a matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Filename for the saved figure
            
        Returns:
            Path to the saved figure
        """
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save plot to {filepath}")
            self.logger.error(traceback.format_exc())
            raise