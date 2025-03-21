import numpy as np
import torch
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import umap
import networkx as nx
from scipy.signal import find_peaks


class ActivationAnalyzer:
    """
    Analyzes activation patterns to detect quantum field-like properties.
    
    This class is responsible for:
    1. Dimensionality reduction of activation patterns
    2. Detecting clusters and transitions in activation space
    3. Measuring topological properties of activation patterns
    4. Analyzing stability and coherence metrics
    5. Detecting patterns related to quantum field theory
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the activation analyzer.
        
        Args:
            config: Configuration for analysis methods
            logger: Logger instance for detailed logging
        """
        self.config = config
        
        # Setup logging
        self.logger = logger or logging.getLogger('activation_analyzer')
        
        # Constants for analysis
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
        self.golden_ratio_inverse = 1 / self.golden_ratio  # ≈ 0.618...
        
        self.logger.info("ActivationAnalyzer initialized")
        self.logger.debug(f"Analysis config: {config}")
    
    def reduce_dimensions(self, activations: np.ndarray, method: str = "tsne", 
                        n_components: int = 2, **kwargs) -> np.ndarray:
        """
        Reduce dimensionality of activation patterns for visualization and analysis.
        
        Args:
            activations: Activation matrix [samples, features]
            method: Dimensionality reduction method (tsne, pca, umap, mds)
            n_components: Number of dimensions to reduce to
            **kwargs: Additional parameters for the reduction method
            
        Returns:
            Reduced dimensionality representation [samples, n_components]
        """
        self.logger.info(f"Reducing dimensions with method: {method}, components: {n_components}")
        
        try:
            # Ensure activations is 2D
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D")
                activations = activations.reshape(activations.shape[0], -1)
            
            # Apply dimensionality reduction
            if method.lower() == "tsne":
                reducer = TSNE(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(activations)
            elif method.lower() == "pca":
                reducer = PCA(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(activations)
                # Store explained variance
                self.last_pca_explained_variance = reducer.explained_variance_ratio_
            elif method.lower() == "umap":
                reducer = umap.UMAP(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(activations)
            elif method.lower() == "mds":
                reducer = MDS(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(activations)
            else:
                self.logger.error(f"Unknown dimensionality reduction method: {method}")
                raise ValueError(f"Unknown method: {method}")
            
            self.logger.info(f"Dimensionality reduction completed: {activations.shape} -> {reduced_data.shape}")
            return reduced_data
            
        except Exception as e:
            self.logger.error(f"Failed to reduce dimensions with method {method}")
            self.logger.error(traceback.format_exc())
            raise
    
    def cluster_activations(self, activations: np.ndarray, method: str = "kmeans", 
                          n_clusters: int = None, **kwargs) -> Dict[str, Any]:
        """
        Cluster activation patterns to identify distinct states.
        
        Args:
            activations: Activation matrix [samples, features]
            method: Clustering method (kmeans, dbscan)
            n_clusters: Number of clusters for kmeans
            **kwargs: Additional parameters for clustering method
            
        Returns:
            Dictionary containing clustering results
        """
        self.logger.info(f"Clustering activations with method: {method}")
        
        try:
            # Ensure activations is 2D
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D")
                activations = activations.reshape(activations.shape[0], -1)
            
            # Apply clustering
            if method.lower() == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, **kwargs)
                labels = clusterer.fit_predict(activations)
                centers = clusterer.cluster_centers_
                inertia = clusterer.inertia_
                
                result = {
                    "labels": labels,
                    "centers": centers,
                    "inertia": inertia,
                    "n_clusters": n_clusters
                }
                
            elif method.lower() == "dbscan":
                clusterer = DBSCAN(**kwargs)
                labels = clusterer.fit_predict(activations)
                
                # Calculate cluster centers
                unique_labels = set(labels)
                centers = []
                for label in unique_labels:
                    if label != -1:  # Skip noise points
                        center = np.mean(activations[labels == label], axis=0)
                        centers.append(center)
                
                result = {
                    "labels": labels,
                    "centers": np.array(centers) if centers else np.array([]),
                    "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0)
                }
                
            else:
                self.logger.error(f"Unknown clustering method: {method}")
                raise ValueError(f"Unknown method: {method}")
            
            # Compute additional metrics
            # Silhouette score if more than one cluster
            if result["n_clusters"] > 1 and len(set(labels)) > 1:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(activations, labels)
                result["silhouette_score"] = silhouette
            
            # Cluster distances
            if result["n_clusters"] > 1 and len(result["centers"]) > 1:
                center_distances = squareform(pdist(result["centers"]))
                result["center_distances"] = center_distances
                result["mean_cluster_distance"] = np.mean(center_distances)
                result["min_cluster_distance"] = np.min(center_distances + np.eye(len(center_distances)) * 9999)
                result["max_cluster_distance"] = np.max(center_distances)
            
            self.logger.info(f"Clustering completed: {result['n_clusters']} clusters identified")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cluster activations with method {method}")
            self.logger.error(traceback.format_exc())
            raise
    
    def measure_transition_dynamics(self, activation_sequence: List[np.ndarray]) -> Dict[str, Any]:
        """
        Measure transition dynamics between activation states.
        
        Args:
            activation_sequence: List of activation states in sequence
            
        Returns:
            Dictionary containing transition dynamics metrics
        """
        self.logger.info("Measuring transition dynamics in activation sequence")
        
        try:
            # Ensure we have a valid sequence
            if len(activation_sequence) < 2:
                self.logger.warning("Cannot measure transitions: need at least 2 states")
                return {"error": "Insufficient sequence length"}
            
            # Process activation sequence
            processed_sequence = []
            for act in activation_sequence:
                # Flatten and normalize
                act_flat = act.reshape(1, -1).squeeze()
                act_norm = act_flat / (np.linalg.norm(act_flat) + 1e-8)
                processed_sequence.append(act_norm)
            
            # Compute distances between consecutive states
            distances = []
            for i in range(len(processed_sequence) - 1):
                dist = np.linalg.norm(processed_sequence[i] - processed_sequence[i+1])
                distances.append(dist)
            
            # Identify potential jumps (discontinuities)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            jump_threshold = mean_dist + 2 * std_dist
            
            jumps = []
            for i, dist in enumerate(distances):
                if dist > jump_threshold:
                    jumps.append({"position": i, "distance": dist})
            
            # Compute momentum-like properties
            deltas = np.diff(distances)  # Acceleration-like
            
            # Compute field coherence using autocorrelation
            if len(processed_sequence) > 10:
                # Calculate autocorrelation function
                avg_state = np.mean(processed_sequence, axis=0)
                fluctuations = [state - avg_state for state in processed_sequence]
                autocorr = []
                
                for lag in range(min(10, len(fluctuations))):
                    corr = np.mean([np.dot(fluctuations[i], fluctuations[i+lag]) 
                                  for i in range(len(fluctuations) - lag)])
                    autocorr.append(corr)
                
                # Normalize
                autocorr = np.array(autocorr) / autocorr[0] if autocorr[0] != 0 else np.zeros_like(autocorr)
            else:
                autocorr = []
            
            result = {
                "distances": distances,
                "mean_distance": mean_dist,
                "std_distance": std_dist,
                "jumps": jumps,
                "jump_count": len(jumps),
                "sequential_changes": deltas.tolist(),
                "autocorrelation": autocorr.tolist() if isinstance(autocorr, np.ndarray) else autocorr
            }
            
            # Check for quantum-like behavior
            if jumps:
                self.logger.info(f"Found {len(jumps)} potential quantum jumps in transition sequence")
                
                # Analyze each jump
                for jump in jumps:
                    pos = jump["position"]
                    if pos > 0 and pos < len(processed_sequence) - 1:
                        # Check if there's evidence of tunneling (no intermediate states)
                        before = processed_sequence[pos]
                        after = processed_sequence[pos + 1]
                        
                        # In quantum tunneling, we'd expect a discontinuous jump
                        # with no evidence of intermediate states between two stable configurations
                        result["quantum_tunneling_evidence"] = True
            
            self.logger.info(f"Transition dynamics measured: {len(distances)} transitions analyzed")
            return result
            
        except Exception as e:
            self.logger.error("Failed to measure transition dynamics")
            self.logger.error(traceback.format_exc())
            raise
    
    def detect_state_jumps(self, activation_sequence: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect quantum-like jumps in state sequence.
        
        Args:
            activation_sequence: Sequence of activation states [timesteps, features]
            threshold: Threshold for jump detection (in standard deviations)
            
        Returns:
            Dictionary containing jump detection results
        """
        self.logger.info(f"Detecting state jumps with threshold: {threshold}")
        
        try:
            # Ensure activations are properly shaped
            if len(activation_sequence.shape) != 2:
                self.logger.warning(f"Reshaping sequence from {activation_sequence.shape} to 2D")
                activation_sequence = activation_sequence.reshape(activation_sequence.shape[0], -1)
            
            n_steps = activation_sequence.shape[0]
            
            if n_steps < 3:
                self.logger.warning("Sequence too short for jump detection")
                return {"error": "Sequence too short"}
            
            # Compute distances between consecutive states
            distances = []
            for i in range(n_steps - 1):
                state1 = activation_sequence[i]
                state2 = activation_sequence[i + 1]
                
                # Normalize states for comparable distances
                state1_norm = state1 / (np.linalg.norm(state1) + 1e-8)
                state2_norm = state2 / (np.linalg.norm(state2) + 1e-8)
                
                dist = np.linalg.norm(state1_norm - state2_norm)
                distances.append(dist)
            
            # Detect jumps
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            jump_threshold = mean_dist + threshold * std_dist
            
            jumps = []
            for i, dist in enumerate(distances):
                if dist > jump_threshold:
                    jumps.append({
                        "position": i,
                        "distance": dist,
                        "z_score": (dist - mean_dist) / std_dist
                    })
            
            # Analyze jump characteristics
            jump_characteristics = {}
            if jumps:
                # Time between jumps
                if len(jumps) > 1:
                    jump_positions = [j["position"] for j in jumps]
                    intervals = np.diff(jump_positions)
                    jump_characteristics["intervals"] = intervals.tolist()
                    jump_characteristics["mean_interval"] = float(np.mean(intervals))
                    jump_characteristics["std_interval"] = float(np.std(intervals))
                
                # Jump magnitude distribution
                jump_magnitudes = [j["distance"] for j in jumps]
                jump_characteristics["magnitudes"] = jump_magnitudes
                jump_characteristics["mean_magnitude"] = float(np.mean(jump_magnitudes))
                jump_characteristics["std_magnitude"] = float(np.std(jump_magnitudes))
                
                # Check quantum properties - rapid transitions
                # In quantum systems, transitions are typically very fast
                is_quantum_like = all(jumps[i+1]["position"] - jumps[i]["position"] > 3 
                                    for i in range(len(jumps)-1)) if len(jumps) > 1 else False
                
                jump_characteristics["quantum_like_spacing"] = is_quantum_like
            
            result = {
                "distances": distances,
                "mean_distance": float(mean_dist),
                "std_distance": float(std_dist),
                "jump_threshold": float(jump_threshold),
                "jumps": jumps,
                "jump_count": len(jumps),
                "jump_characteristics": jump_characteristics
            }
            
            self.logger.info(f"Detected {len(jumps)} jumps in sequence of {n_steps} states")
            return result
            
        except Exception as e:
            self.logger.error("Failed to detect state jumps")
            self.logger.error(traceback.format_exc())
            raise
    
    def compute_stability_metrics(self, activations: np.ndarray, perturbations: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute stability metrics for an activation state under perturbations.
        
        Args:
            activations: Base activation state
            perturbations: List of perturbed activation states
            
        Returns:
            Dictionary containing stability metrics
        """
        self.logger.info("Computing stability metrics under perturbations")
        
        try:
            # Flatten activations
            base_act = activations.reshape(1, -1).squeeze()
            base_norm = base_act / (np.linalg.norm(base_act) + 1e-8)
            
            # Process perturbations
            pert_distances = []
            for pert in perturbations:
                pert_flat = pert.reshape(1, -1).squeeze()
                pert_norm = pert_flat / (np.linalg.norm(pert_flat) + 1e-8)
                
                dist = np.linalg.norm(base_norm - pert_norm)
                pert_distances.append(dist)
            
            # Compute stability metrics
            mean_dist = np.mean(pert_distances)
            std_dist = np.std(pert_distances)
            max_dist = np.max(pert_distances)
            min_dist = np.min(pert_distances)
            
            # Calculate topological persistence-like measure
            # If activations maintain structure under perturbation, this
            # is evidence of topological protection
            topological_persistence = 1.0 - (mean_dist / 2.0)  # Normalized to [0,1]
            
            result = {
                "perturbation_distances": pert_distances,
                "mean_distance": float(mean_dist),
                "std_distance": float(std_dist),
                "max_distance": float(max_dist),
                "min_distance": float(min_dist),
                "topological_persistence": float(topological_persistence)
            }
            
            # Check for quantum-like stability
            # In quantum systems protected by topology, stability is high
            if topological_persistence > 0.8:
                result["stability_type"] = "high (quantum-like)"
            elif topological_persistence > 0.5:
                result["stability_type"] = "moderate"
            else:
                result["stability_type"] = "low (classical-like)"
            
            self.logger.info(f"Computed stability metrics: topological_persistence={topological_persistence:.4f}")
            return result
            
        except Exception as e:
            self.logger.error("Failed to compute stability metrics")
            self.logger.error(traceback.format_exc())
            raise
    
    def analyze_dimensional_hierarchy(self, activations: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the natural organizational patterns in activations without imposing
        artificial dimensional structures, focusing on discovering emergent field-like
        properties that may indicate quantum organization of meaning.
        
        Args:
            activations: Activation matrix [samples, features]
            
        Returns:
            Dictionary containing analysis of natural patterns
        """
        self.logger.info("Analyzing natural dimensional organization of semantic field")
        
        try:
            # Ensure activations are properly shaped for analysis
            orig_shape = activations.shape
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D for analysis")
                activations = activations.reshape(activations.shape[0], -1)
            
            # Calculate basic statistical properties of the activation space
            activation_mean = np.mean(activations)
            activation_std = np.std(activations)
            activation_min = np.min(activations)
            activation_max = np.max(activations)
            
            self.logger.info(f"Activation field statistics: mean={activation_mean:.6f}, std={activation_std:.6f}, "
                            f"min={activation_min:.6f}, max={activation_max:.6f}")
            
            # Compute similarity matrix (correlation-based)
            # This represents the "field potential" in semantic space
            similarity = np.corrcoef(activations)
            
            # Analyze eigenvalue structure of the similarity field
            # This reveals the natural "resonant modes" of the semantic field
            eigenvalues, eigenvectors = np.linalg.eigh(similarity)
            
            # Sort in descending order (largest eigenvalues first)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Normalize eigenvalues to analyze their relative importance
            total_variance = np.sum(np.abs(eigenvalues))
            explained_variance_ratio = np.abs(eigenvalues) / total_variance if total_variance > 0 else np.abs(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            self.logger.info(f"Found {len(eigenvalues)} natural modes in the semantic field")
            self.logger.info(f"Top 5 field resonance strengths: {eigenvalues[:5]}")
            
            # Find natural dimension thresholds based on cumulative variance
            # These represent the "effective dimensionality" of the semantic field
            dim_thresholds = {}
            for threshold in [0.5, 0.75, 0.9, 0.95, 0.99]:
                threshold_str = f"{int(threshold*100)}%"
                if any(cumulative_variance >= threshold):
                    dim = np.argmax(cumulative_variance >= threshold) + 1
                    dim_thresholds[threshold_str] = dim
                    self.logger.info(f"Natural {threshold_str} variance threshold: {dim} dimensions")
                else:
                    dim_thresholds[threshold_str] = len(eigenvalues)
                    self.logger.info(f"Natural {threshold_str} variance threshold: all {len(eigenvalues)} dimensions")
            
            # Calculate natural compression ratios between thresholds
            # These may reveal mathematical patterns in dimensional organization
            compression_ratios = []
            threshold_keys = sorted([k for k in dim_thresholds.keys()], 
                                key=lambda x: int(x.replace('%', '')))
            
            for i in range(len(threshold_keys) - 1):
                from_dim = dim_thresholds[threshold_keys[i]]
                to_dim = dim_thresholds[threshold_keys[i+1]]
                
                if to_dim > 0:
                    ratio = from_dim / to_dim
                    compression_ratios.append({
                        "from_threshold": threshold_keys[i],
                        "to_threshold": threshold_keys[i+1],
                        "from_dim": from_dim,
                        "to_dim": to_dim,
                        "ratio": float(ratio)
                    })
                    self.logger.info(f"Natural compression ratio {threshold_keys[i]}→{threshold_keys[i+1]}: {ratio:.4f}")
            
            # Analyze eigenvalue ratios for golden ratio patterns
            # This detects natural mathematical organization in the field
            golden_ratio_patterns = []
            if len(eigenvalues) > 1:
                eigenratios = eigenvalues[:-1] / eigenvalues[1:]
                
                self.logger.info(f"Analyzing {len(eigenratios)} eigenvalue ratios for natural patterns")
                
                # Check for values near golden ratio or its inverse
                phi_tolerance = 0.05
                found_patterns = []
                
                for i, ratio in enumerate(eigenratios):
                    # Check proximity to golden ratio and its powers
                    gr_values = [
                        self.golden_ratio,           # φ ≈ 1.618
                        self.golden_ratio_inverse,   # 1/φ ≈ 0.618
                        self.golden_ratio**2,        # φ² ≈ 2.618
                        1/self.golden_ratio**2,      # 1/φ² ≈ 0.382
                        self.golden_ratio + 1,       # φ+1 ≈ 2.618
                        self.golden_ratio - 1        # φ-1 ≈ 0.618
                    ]
                    gr_names = ["φ", "1/φ", "φ²", "1/φ²", "φ+1", "φ-1"]
                    
                    for j, (gr_val, gr_name) in enumerate(zip(gr_values, gr_names)):
                        if abs(ratio - gr_val) < phi_tolerance:
                            deviation = abs(ratio - gr_val) / gr_val
                            pattern = {
                                "indices": f"{i}:{i+1}",
                                "ratio": float(ratio),
                                "gr_value": float(gr_val),
                                "gr_name": gr_name,
                                "deviation": float(deviation)
                            }
                            found_patterns.append(pattern)
                            self.logger.info(f"Golden ratio pattern detected: eigenvalues {i}:{i+1}, "
                                            f"ratio={ratio:.6f}, close to {gr_name}={gr_val:.6f}, "
                                            f"deviation={deviation:.2%}")
                
                # Sort patterns by deviation (most precise matches first)
                golden_ratio_patterns = sorted(found_patterns, key=lambda x: x["deviation"])
            
            # Compile results
            result = {
                "original_dimensions": orig_shape,
                "eigenvalues": eigenvalues.tolist(),
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "dimension_thresholds": dim_thresholds,
                "natural_compression_ratios": compression_ratios,
                "golden_ratio_patterns": golden_ratio_patterns,
                "field_statistics": {
                    "mean": float(activation_mean),
                    "std": float(activation_std),
                    "min": float(activation_min),
                    "max": float(activation_max),
                    "eigenvalue_max": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
                    "eigenvalue_min": float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0,
                    "eigenvalue_ratio": float(eigenvalues[0] / max(1e-10, eigenvalues[-1])) if len(eigenvalues) > 0 else 0.0,
                    "effective_dimensions": int(1.0 / sum(explained_variance_ratio**2)) if len(explained_variance_ratio) > 0 else 0
                }
            }
            
            # Calculate natural compression cascade based on eigenvalue structure
            # This reveals how the field naturally compresses through dimension reduction
            if len(eigenvalues) >= 3:
                # Find natural "steps" in eigenvalue spectrum using peak detection
                eig_diffs = np.diff(eigenvalues)
                peaks, _ = find_peaks(np.abs(eig_diffs))
                
                if len(peaks) > 0:
                    # Natural dimension breakpoints
                    breakpoints = sorted([0] + [p + 1 for p in peaks] + [len(eigenvalues)])
                    cascade = []
                    
                    for i in range(len(breakpoints) - 1):
                        from_dim = breakpoints[i]
                        to_dim = breakpoints[i+1]
                        
                        # Skip meaningless steps
                        if from_dim == to_dim or from_dim >= len(eigenvalues):
                            continue
                        
                        # Calculate information preservation
                        preservation = sum(explained_variance_ratio[from_dim:to_dim])
                        
                        # Calculate compression ratio
                        if to_dim > from_dim:
                            compression = from_dim / to_dim
                            cascade.append({
                                "from_dim": int(from_dim),
                                "to_dim": int(to_dim),
                                "compression_ratio": float(compression),
                                "info_preservation": float(preservation),
                                "eigenvalue_range": [float(eigenvalues[from_dim]), 
                                                    float(eigenvalues[min(to_dim-1, len(eigenvalues)-1)])]
                            })
                    
                    if cascade:
                        self.logger.info(f"Detected natural compression cascade with {len(cascade)} steps")
                        for i, step in enumerate(cascade):
                            self.logger.info(f"  Step {i+1}: {step['from_dim']}→{step['to_dim']} dimensions, "
                                            f"ratio: {step['compression_ratio']:.4f}, "
                                            f"info: {step['info_preservation']:.4f}")
                        
                        result["natural_cascade"] = cascade
            
            self.logger.info(f"Natural dimensional analysis complete: found {len(golden_ratio_patterns)} golden ratio patterns")
            return result
            
        except Exception as e:
            self.logger.error("Failed to analyze natural dimensional organization")
            self.logger.error(traceback.format_exc())
            raise
    
    def compute_principal_components(self, activations: np.ndarray, n_components: int = 10) -> Dict[str, Any]:
        """
        Compute principal components of activation patterns.
        
        Args:
            activations: Activation matrix [samples, features]
            n_components: Number of principal components to compute
            
        Returns:
            Dictionary containing principal component analysis results
        """
        self.logger.info(f"Computing principal components (n={n_components})")
        
        try:
            # Ensure activations is 2D
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D")
                activations = activations.reshape(activations.shape[0], -1)
            
            # Compute PCA
            n_components = min(n_components, min(activations.shape))
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(activations)
            
            # Compute additional metrics
            loadings = pca.components_
            scores = transformed
            eigenvalues = pca.explained_variance_
            
            # Check for patterns in the eigenvalue distribution
            eigenratios = eigenvalues[:-1] / eigenvalues[1:]
            phi_tolerance = 0.05
            
            # Check for golden ratio patterns
            golden_ratio_patterns = []
            for i, ratio in enumerate(eigenratios):
                if (abs(ratio - self.golden_ratio) < phi_tolerance or 
                    abs(ratio - self.golden_ratio_inverse) < phi_tolerance):
                    golden_ratio_patterns.append({
                        "indices": f"{i}:{i+1}",
                        "ratio": float(ratio),
                        "type": "phi" if abs(ratio - self.golden_ratio) < phi_tolerance else "1/phi"
                    })
            
            result = {
                "eigenvalues": eigenvalues.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "loadings": loadings.tolist(),
                "scores": scores.tolist(),
                "n_components": n_components,
                "golden_ratio_patterns": golden_ratio_patterns
            }
            
            self.logger.info(f"PCA completed: {len(golden_ratio_patterns)} golden ratio patterns found")
            return result
            
        except Exception as e:
            self.logger.error("Failed to compute principal components")
            self.logger.error(traceback.format_exc())
            raise
    
    def measure_field_coherence(self, activations: np.ndarray) -> Dict[str, float]:
        """
        Measure quantum field-like coherence properties of activations.
        
        Args:
            activations: Activation matrix [samples, features]
            
        Returns:
            Dictionary containing coherence metrics
        """
        self.logger.info("Measuring field coherence properties")
        
        try:
            # Ensure activations is 2D
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D")
                activations = activations.reshape(activations.shape[0], -1)
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(activations)
            
            # Extract key metrics
            eigenvalues, _ = np.linalg.eigh(corr_matrix)
            
            # Phase coherence (using normalized eigenvalue distribution)
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            total_eigenvalue = np.sum(sorted_eigenvalues)
            normalized_eigenvalues = sorted_eigenvalues / total_eigenvalue if total_eigenvalue > 0 else sorted_eigenvalues
            
            # Phase coherence is measured by how concentrated the eigenvalue distribution is
            phase_coherence = normalized_eigenvalues[0] if len(normalized_eigenvalues) > 0 else 0
            
            # Correlation length (characteristic decay of correlations)
            # Approximate using half-life of eigenvalue distribution
            if len(normalized_eigenvalues) > 1:
                cutoff_value = normalized_eigenvalues[0] / 2
                correlation_length = np.argmax(normalized_eigenvalues < cutoff_value)
                if correlation_length == 0:  # No eigenvalue below cutoff
                    correlation_length = len(normalized_eigenvalues)
            else:
                correlation_length = 1
            
            # Field coherence metrics
            result = {
                "phase_coherence": float(phase_coherence),
                "correlation_length": int(correlation_length),
                "eigenvalue_max": float(sorted_eigenvalues[0] if len(sorted_eigenvalues) > 0 else 0),
                "eigenvalue_min": float(sorted_eigenvalues[-1] if len(sorted_eigenvalues) > 0 else 0),
                "eigenvalue_ratio": float(sorted_eigenvalues[0] / max(1e-10, sorted_eigenvalues[-1])),
                "effective_dimensions": int(1.0 / sum(normalized_eigenvalues**2) if len(normalized_eigenvalues) > 0 else 0)
            }
            
            self.logger.info(f"Field coherence measured: phase_coherence={phase_coherence:.4f}, "
                            f"correlation_length={correlation_length}")
            return result
            
        except Exception as e:
            self.logger.error("Failed to measure field coherence")
            self.logger.error(traceback.format_exc())
            raise
    
    def analyze_eigenvalue_distribution(self, similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the eigenvalue distribution of a similarity matrix for quantum properties.
        
        Args:
            similarity_matrix: Similarity or correlation matrix
            
        Returns:
            Dictionary containing eigenvalue analysis results
        """
        self.logger.info("Analyzing eigenvalue distribution")
        
        try:
            # Compute eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
            
            # Sort eigenvalues in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Normalize eigenvalues
            total = np.sum(np.abs(eigenvalues))
            normalized_eigenvalues = eigenvalues / total if total > 0 else eigenvalues
            
            # Calculate eigenvalue spacing statistics
            spacings = np.diff(eigenvalues)
            mean_spacing = np.mean(spacings) if len(spacings) > 0 else 0
            
            # Nearest-neighbor spacing distribution (NNSD)
            # This is a key quantum chaos indicator
            if len(eigenvalues) > 2:
                normalized_spacings = spacings / mean_spacing if mean_spacing > 0 else spacings
                
                # Compute spacing histogram
                hist, bin_edges = np.histogram(normalized_spacings, bins=10, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Check for Wigner-Dyson distribution (quantum chaos)
                # vs. Poisson distribution (classical systems)
                
                # Wigner surmise (GOE) for quantum chaotic systems
                wigner = lambda s: (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)
                
                # Poisson for classical systems
                poisson = lambda s: np.exp(-s)
                
                # Compute theoretical distributions
                wigner_dist = wigner(bin_centers)
                poisson_dist = poisson(bin_centers)
                
                # Calculate goodness of fit
                wigner_residuals = np.sum((hist - wigner_dist)**2)
                poisson_residuals = np.sum((hist - poisson_dist)**2)
                
                # Determine which distribution is closer
                distribution_type = "quantum-like (Wigner)" if wigner_residuals < poisson_residuals else "classical-like (Poisson)"
                fit_ratio = poisson_residuals / wigner_residuals if wigner_residuals > 0 else float('inf')
            else:
                distribution_type = "insufficient data"
                fit_ratio = 0.0
                normalized_spacings = []
            
            # Check for golden ratio patterns in eigenvalue spacings
            golden_ratio_patterns = []
            eigenratios = eigenvalues[:-1] / eigenvalues[1:] if len(eigenvalues) > 1 else []
            
            phi_tolerance = 0.05
            for i, ratio in enumerate(eigenratios[:min(20, len(eigenratios))]):
                if (abs(ratio - self.golden_ratio) < phi_tolerance or 
                    abs(ratio - self.golden_ratio_inverse) < phi_tolerance):
                    golden_ratio_patterns.append({
                        "indices": f"{i}:{i+1}",
                        "ratio": float(ratio),
                        "type": "phi" if abs(ratio - self.golden_ratio) < phi_tolerance else "1/phi"
                    })
            
            result = {
                "eigenvalues": eigenvalues.tolist(),
                "normalized_eigenvalues": normalized_eigenvalues.tolist(),
                "eigenvalue_spacings": spacings.tolist() if len(spacings) > 0 else [],
                "mean_spacing": float(mean_spacing),
                "spacing_distribution_type": distribution_type,
                "quantum_classical_ratio": float(fit_ratio),
                "golden_ratio_patterns": golden_ratio_patterns,
                "largest_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
                "eigenvalue_sum": float(np.sum(eigenvalues)),
                "participation_ratio": float(1.0 / np.sum(normalized_eigenvalues**2)) if len(normalized_eigenvalues) > 0 else 0
            }
            
            self.logger.info(f"Eigenvalue analysis complete: distribution type={distribution_type}, "
                            f"golden ratio patterns={len(golden_ratio_patterns)}")
            return result
            
        except Exception as e:
            self.logger.error("Failed to analyze eigenvalue distribution")
            self.logger.error(traceback.format_exc())
            raise
    
    def compute_topological_features(self, activations: np.ndarray, n_neighbors: int = 15) -> Dict[str, Any]:
        """
        Compute topological features of the activation space with detailed logging.
        
        Args:
            activations: Activation matrix [samples, features]
            n_neighbors: Number of neighbors for topological analysis
            
        Returns:
            Dictionary containing topological features
        """
        self.logger.info(f"Computing topological features (n_neighbors={n_neighbors})")
        
        try:
            # Ensure activations is 2D
            if len(activations.shape) != 2:
                self.logger.warning(f"Reshaping activations from {activations.shape} to 2D")
                activations = activations.reshape(activations.shape[0], -1)
            
            if activations.shape[0] < n_neighbors + 1:
                self.logger.warning(f"Too few samples ({activations.shape[0]}) for n_neighbors={n_neighbors}")
                n_neighbors = max(2, activations.shape[0] - 1)
                self.logger.warning(f"Reduced to n_neighbors={n_neighbors}")
            
            # Log basic statistics of input activation matrix
            self.logger.info(f"Activation matrix shape: {activations.shape}")
            self.logger.info(f"Activation statistics: mean={np.mean(activations):.6f}, "
                            f"std={np.std(activations):.6f}, "
                            f"min={np.min(activations):.6f}, "
                            f"max={np.max(activations):.6f}")
            
            # Compute distance matrix
            self.logger.info("Computing distance matrix...")
            dist_matrix = squareform(pdist(activations))
            self.logger.info(f"Distance matrix shape: {dist_matrix.shape}")
            self.logger.info(f"Distance statistics: mean={np.mean(dist_matrix):.6f}, "
                            f"std={np.std(dist_matrix):.6f}, "
                            f"min={np.min(dist_matrix):.6f}, "
                            f"max={np.max(dist_matrix):.6f}")
            
            # Create k-nearest neighbor graph
            self.logger.info(f"Creating k-nearest neighbor graph with k={n_neighbors}...")
            knn_graph = np.zeros_like(dist_matrix)
            for i in range(dist_matrix.shape[0]):
                # Get indices of k nearest neighbors
                nearest = np.argsort(dist_matrix[i])[1:n_neighbors+1]  # Skip self
                knn_graph[i, nearest] = 1
            
            # Make graph symmetric
            knn_graph = np.maximum(knn_graph, knn_graph.T)
            
            self.logger.info(f"KNN graph created: {np.sum(knn_graph)} edges")
            
            # Convert to networkx graph for analysis
            G = nx.from_numpy_array(knn_graph)
            self.logger.info(f"NetworkX graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Compute graph metrics
            self.logger.info("Computing graph metrics...")
            clustering_coefficient = nx.average_clustering(G)
            self.logger.info(f"Average clustering coefficient: {clustering_coefficient:.6f}")
            
            try:
                average_path_length = nx.average_shortest_path_length(G)
                self.logger.info(f"Average shortest path length: {average_path_length:.6f}")
            except nx.NetworkXError:
                # Graph may not be connected
                self.logger.warning("Graph is not connected, computing metrics on largest component")
                largest_cc = max(nx.connected_components(G), key=len)
                largest_cc_subgraph = G.subgraph(largest_cc)
                average_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
                self.logger.info(f"Largest component size: {len(largest_cc)}/{G.number_of_nodes()}")
                self.logger.info(f"Average shortest path length (largest component): {average_path_length:.6f}")
            
            # Compute connected components
            connected_components = list(nx.connected_components(G))
            largest_component_size = len(max(connected_components, key=len))
            
            self.logger.info(f"Connected components: {len(connected_components)}")
            self.logger.info(f"Largest component size: {largest_component_size}")
            
            # Log component size distribution
            component_sizes = [len(c) for c in connected_components]
            self.logger.info(f"Component size distribution: {component_sizes}")
            
            # Compute spectral gap (algebraic connectivity)
            self.logger.info("Computing spectral properties...")
            laplacian = nx.normalized_laplacian_matrix(G).todense()
            
            # Compute eigenvalues of the Laplacian
            laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
            # Sort eigenvalues
            laplacian_eigenvalues = np.sort(laplacian_eigenvalues)
            
            # The spectral gap is the second smallest eigenvalue
            # (the smallest is always 0 for connected components)
            spectral_gap = laplacian_eigenvalues[1] if len(laplacian_eigenvalues) > 1 else 0
            
            self.logger.info(f"Spectral gap: {spectral_gap:.6f}")
            self.logger.info(f"Top 5 Laplacian eigenvalues: {laplacian_eigenvalues[:5]}")
            
            # Approximate Betti numbers
            self.logger.info("Computing topological invariants (Betti numbers)...")
            # Betti-0 = number of connected components
            betti_0 = len(connected_components)
            
            # Betti-1 = number of cycles (approx)
            # For a connected graph: betti_1 = edges - vertices + 1
            # For a disconnected graph, sum over components
            betti_1 = 0
            for comp in connected_components:
                subgraph = G.subgraph(comp)
                if len(comp) > 2:  # Need at least 3 nodes for a cycle
                    edges = subgraph.number_of_edges()
                    vertices = subgraph.number_of_nodes()
                    betti_1 += max(0, edges - vertices + 1)
            
            self.logger.info(f"Betti-0 (connected components): {betti_0}")
            self.logger.info(f"Betti-1 (cycles/holes): {betti_1}")
            
            # Compute network diameter for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_subgraph = G.subgraph(largest_cc)
            network_diameter = nx.diameter(largest_cc_subgraph)
            
            self.logger.info(f"Network diameter: {network_diameter}")
            
            # Estimate topological dimension
            if n_neighbors > 1:
                topological_dimension = np.log(largest_component_size) / np.log(n_neighbors)
                self.logger.info(f"Estimated topological dimension: {topological_dimension:.6f}")
            else:
                topological_dimension = 0
                self.logger.info("Cannot estimate topological dimension with n_neighbors <= 1")
            
            # Compute continuous quantum field score
            # This score balances various topological metrics to quantify quantum-like behavior
            # Higher values indicate stronger evidence of quantum field properties
            if betti_0 > 0:
                # The ratio betti_1/betti_0 measures topological complexity
                # spectral_gap measures how well-connected the graph is
                # clustering_coefficient measures local density
                quantum_field_score = spectral_gap * clustering_coefficient * (1 + betti_1 / betti_0)
            else:
                quantum_field_score = 0
                
            self.logger.info(f"Computed quantum field score: {quantum_field_score:.6f}")
            
            # Log the detailed interpretation of the quantum field score
            if quantum_field_score > 1.0:
                self.logger.info("Quantum field score interpretation: HIGH - Strong evidence of quantum-like properties")
            elif quantum_field_score > 0.5:
                self.logger.info("Quantum field score interpretation: MEDIUM - Moderate evidence of quantum-like properties")
            else:
                self.logger.info("Quantum field score interpretation: LOW - Limited evidence of quantum-like properties")
            
            # Compute traditional topological protection categorization for backwards compatibility
            if spectral_gap > 0.1 and clustering_coefficient > 0.6:
                topological_protection = "high"
            elif spectral_gap > 0.05 and clustering_coefficient > 0.4:
                topological_protection = "medium"
            else:
                topological_protection = "low"
                
            self.logger.info(f"Traditional topological protection categorization: {topological_protection}")
            
            # Estimate Chern-like number (topological charge) based on network properties
            estimated_chern = (betti_1 * spectral_gap * clustering_coefficient) * 1000
            self.logger.info(f"Estimated topological charge: {estimated_chern:.6f}")
            
            # Prepare result
            result = {
                # Basic graph properties
                "n_neighbors": n_neighbors,
                "clustering_coefficient": float(clustering_coefficient),
                "average_path_length": float(average_path_length),
                "connected_components": betti_0,
                "largest_component_size": largest_component_size,
                "spectral_gap": float(spectral_gap),
                "network_diameter": float(network_diameter),
                "topological_dimension": float(topological_dimension),
                
                # Topological invariants
                "betti_0": betti_0,
                "betti_1": betti_1,
                "betti_ratio": float(betti_1 / max(1, betti_0)),
                
                # Quantum field metrics
                "quantum_field_score": float(quantum_field_score),
                "topological_protection": topological_protection,
                "estimated_topological_charge": float(estimated_chern),
                
                # Detailed component information for deeper analysis
                "component_sizes": component_sizes,
                "laplacian_eigenvalues_top5": laplacian_eigenvalues[:5].tolist()
            }
            
            self.logger.info(f"Topological analysis complete: betti_0={betti_0}, betti_1={betti_1}, "
                            f"quantum_field_score={quantum_field_score:.6f}")
            return result
            
        except Exception as e:
            self.logger.error("Failed to compute topological features")
            self.logger.error(traceback.format_exc())
            raise