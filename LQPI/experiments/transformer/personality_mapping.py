# experiments/personality_mapping.py

import os
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from experiments.base.base_experiment import BaseExperiment
import traceback
import datetime

class PersonalityMappingExperiment(BaseExperiment):
    """
    Experiment to map distinct personality states in the model's activation space.
    
    This experiment tests whether personalities emerge as coherent field patterns
    rather than localized features, by generating responses across different
    personality prompts and analyzing the resulting activation patterns.
    """
    
    def __init__(self, controller, config, output_dir, logger=None):
        super().__init__(controller, config, output_dir, logger)
        
        self.description = (
            "This experiment maps personality states in the model's activation space to "
            "determine whether they form distinct field-like patterns. By analyzing how "
            "different personalities cluster and organize in the activation space, we can "
            "test whether personality emergence exhibits quantum field-like properties."
        )
        
        # Initialize experiment-specific attributes
        self.personalities = {}
        self.questions = []
        self.activations_by_personality = {}
        self.embeddings = None
        self.activation_labels = []
        self.findings = []
        
        self.logger.info("Personality Mapping Experiment initialized")
    
    def setup(self):
        """
        Set up the personality mapping experiment.
        
        This includes loading personality definitions and questions,
        and ensuring the model and instrumentor are properly set up.
        """
        self.logger.info("Setting up Personality Mapping Experiment")

        # Get personalities directly from the experiment config
        if 'personalities' in self.config and isinstance(self.config['personalities'], list):
            personalities_list = self.config['personalities']
            
            # Handle list of dictionaries with name and description
            if personalities_list and isinstance(personalities_list[0], dict):
                for personality in personalities_list:
                    if isinstance(personality, dict) and "name" in personality and "description" in personality:
                        self.personalities[personality["name"]] = personality["description"]
                        self.logger.debug(f"Added personality: {personality['name']}")
        
        if not self.personalities:
            self.logger.error("No personalities defined in configuration")
            raise ValueError("No personalities defined in configuration")
        
        self.logger.info(f"Loaded {len(self.personalities)} personalities")
        
        # Load questions from config
        self.questions = self.config.get("questions", [])
        if not self.questions:
            self.logger.error("No questions defined in configuration")
            raise ValueError("No questions defined in configuration")
        
        self.logger.info(f"Loaded {len(self.questions)} questions")
        
        # Ensure model is loaded
        if self.controller.model is None:
            self.logger.info("Loading model...")
            self.controller.load_model()
        
        # Ensure instrumentor is set up
        if self.controller.instrumentor is None:
            self.logger.error("Instrumentor not set up in controller")
            raise RuntimeError("Instrumentor must be set up before running experiment")
        
        # Log experiment setup details
        self.logger.info(f"Experiment setup completed with {len(self.personalities)} personalities "
                        f"and {len(self.questions)} questions")
        
    def run(self):
        """
        Run the personality mapping experiment.
        
        This involves generating responses for each personality-question pair
        and capturing the resulting activation patterns.
        """
        self.logger.info("Running Personality Mapping Experiment")
        
        # Initialize storage for activations
        self.activations_by_personality = {}
        self.activation_labels = []
        
        # Track progress
        total_combinations = len(self.personalities) * len(self.questions)
        completed = 0
        
        # For each personality
        for persona_name, persona_desc in self.personalities.items():
            self.logger.info(f"Processing personality: {persona_name}")
            self.activations_by_personality[persona_name] = []
            
            # For each question
            for i, question in enumerate(self.questions):
                self.logger.info(f"Processing question {i+1}/{len(self.questions)}: {question[:50]}...")
                
                # Format input with the personality prompt
                if self.controller.model.config.model_type == "llama":
                    # Format for Llama-style models
                    input_text = f"<|system|>\n{persona_desc}\n<|user|>\n{question}\n<|assistant|>"
                else:
                    # Generic format for other models
                    input_text = f"System: {persona_desc}\n\nUser: {question}\n\nAssistant:"
                
                # Log the input using our experiment logger
                self.experiment_logger.log_input(
                    personality=persona_desc,
                    question=question,
                    combined_prompt=input_text
                )
                
                # Capture activations for this input
                try:
                    # Generate and capture activations - pass the experiment logger
                    result = self.controller.instrumentor.capture_activations(
                        input_text=input_text,
                        generation_config={
                            'max_new_tokens': 150,
                            'do_sample': True,
                            'temperature': 0.7,
                            'top_p': 0.9 
                        },
                        experiment_logger=self.experiment_logger  # Pass our logger here
                    )
                    
                    # Extract activations
                    layer_activations = self.controller.instrumentor.extract_layerwise_features()
                    
                    # Store activations and label
                    for layer_name, activation in layer_activations.items():
                        # Only process if activation has valid shape
                        if isinstance(activation, np.ndarray) and activation.size > 0:
                            self.activations_by_personality[persona_name].append(activation)
                            self.activation_labels.append(f"{persona_name}_{i}")
                    
                    # Log generated text
                    self.logger.debug(f"Generated text: {result.get('generated_text', '')[:100]}...")
                    
                    # Add raw data
                    self.add_raw_data(f"{persona_name}_{i}_generated_text", result.get('generated_text', ''))
                    
                    # Update progress
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {persona_name}, question {i}")
                    self.logger.error(traceback.format_exc())
                
                # Small pause to avoid overloading the system
                time.sleep(1)
        
        self.logger.info(f"Completed data collection: {len(self.activation_labels)} activation samples collected")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the collected activation patterns to identify field-like properties.
        
        This method implements the quantum field approach to personality analysis:
        1. Processes activation patterns into field states
        2. Uses phase-based clustering to allow natural emergence
        3. Analyzes dimensional organization and topological properties
        4. Measures quantum field-like metrics
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing personality mapping results")
        
        # Convert activations to a format suitable for analysis
        activations = []
        for persona_name, persona_activations in self.activations_by_personality.items():
            activations.extend(persona_activations)
        
        if not activations:
            self.logger.error("No activations collected to analyze")
            raise ValueError("No activations collected")
        
        # Stack activations into a single array if they have consistent shapes
        try:
            # Find the most common shape for activations
            shapes = [a.shape for a in activations]
            shape_counts = {}
            for shape in shapes:
                shape_str = str(shape)
                shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            
            most_common_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Most common activation shape: {most_common_shape}")
            self.logger.debug(f"All activation shapes: {shape_counts}")
            
            # Filter activations to include only those with the most common shape
            filtered_activations = []
            filtered_labels = []
            
            for i, activation in enumerate(activations):
                if str(activation.shape) == most_common_shape:
                    filtered_activations.append(activation)
                    filtered_labels.append(self.activation_labels[i])
            
            self.logger.info(f"Using {len(filtered_activations)}/{len(activations)} activations "
                        f"with shape {most_common_shape}")
            
            # Convert to numpy array
            activations_array = np.stack(filtered_activations)
            self.logger.info(f"Stacked activations shape: {activations_array.shape}")
            
            # Log basic statistics of the activation array
            self.logger.info(f"Activation statistics: mean={np.mean(activations_array):.6f}, "
                        f"std={np.std(activations_array):.6f}, "
                        f"min={np.min(activations_array):.6f}, "
                        f"max={np.max(activations_array):.6f}")
            
            # Update activation labels
            self.activation_labels = filtered_labels
            
        except Exception as e:
            self.logger.error("Failed to stack activations into array")
            self.logger.error(traceback.format_exc())
            raise
        
        # Extract personality from labels
        personality_labels = [label.split('_')[0] for label in self.activation_labels]
        
        # Log personality distribution
        personality_counts = {}
        for p in personality_labels:
            personality_counts[p] = personality_counts.get(p, 0) + 1
        self.logger.info(f"Personality distribution in dataset: {personality_counts}")
        
        # Perform dimensionality reduction
        try:
            self.logger.info("Performing dimensionality reduction...")
            
            # First perform PCA to examine eigenvalue structure
            self.logger.info("Running initial PCA to examine eigenvalue structure...")
            pca_all = self.controller.analyzer.compute_principal_components(
                activations_array, n_components=min(activations_array.shape[0], activations_array.shape[1])
            )
            
            # Log eigenvalue distribution
            eigenvalues = np.array(pca_all["eigenvalues"])
            total_variance = np.sum(eigenvalues)
            explained_var_ratio = eigenvalues / total_variance
            cum_explained_var = np.cumsum(explained_var_ratio)
            
            self.logger.info(f"Top 10 eigenvalues: {eigenvalues[:10]}")
            self.logger.info(f"Total variance: {total_variance}")
            self.logger.info(f"Explained variance ratios (top 10): {explained_var_ratio[:10]}")
            self.logger.info(f"Cumulative explained variance milestones: "
                        f"50%: {np.argmax(cum_explained_var >= 0.5) + 1}, "
                        f"80%: {np.argmax(cum_explained_var >= 0.8) + 1}, "
                        f"90%: {np.argmax(cum_explained_var >= 0.9) + 1}, "
                        f"95%: {np.argmax(cum_explained_var >= 0.95) + 1}")
            
            # Check for consecutive eigenvalue ratios
            eigenratios = eigenvalues[:-1] / eigenvalues[1:]
            self.logger.info(f"Eigenvalue ratios (first 10): {eigenratios[:10]}")
            
            # Check for golden ratio patterns with adaptive tolerance
            golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
            golden_ratio_inverse = 1 / golden_ratio  # ≈ 0.618...
            
            # Compute adaptive tolerance based on eigenvalue distribution
            eigenvalue_variance = np.var(eigenratios[:10])
            adaptive_phi_tolerance = min(0.1, max(0.03, eigenvalue_variance * 0.5))
            self.logger.info(f"Using adaptive phi tolerance: {adaptive_phi_tolerance} "
                        f"(based on eigenvalue ratio variance: {eigenvalue_variance:.6f})")
            
            # Find golden ratio patterns with adaptive tolerance
            golden_patterns = []
            for i, ratio in enumerate(eigenratios[:20]):
                if (abs(ratio - golden_ratio) < adaptive_phi_tolerance):
                    golden_patterns.append({
                        "indices": f"{i}:{i+1}",
                        "ratio": float(ratio),
                        "type": "phi",
                        "deviation": float(abs(ratio - golden_ratio) / golden_ratio)
                    })
                elif (abs(ratio - golden_ratio_inverse) < adaptive_phi_tolerance):
                    golden_patterns.append({
                        "indices": f"{i}:{i+1}",
                        "ratio": float(ratio),
                        "type": "1/phi",
                        "deviation": float(abs(ratio - golden_ratio_inverse) / golden_ratio_inverse)
                    })
                    
            if golden_patterns:
                self.logger.info(f"Found {len(golden_patterns)} golden ratio patterns in eigenvalues:")
                for pattern in golden_patterns:
                    self.logger.info(f"  - Eigenvalues {pattern['indices']}: ratio={pattern['ratio']:.6f}, "
                                f"type={pattern['type']}, deviation={pattern['deviation']:.4f}")
            else:
                self.logger.info("No golden ratio patterns found in eigenvalue distribution")
            
            # Perform t-SNE with perplexity proportional to dataset size
            optimal_perplexity = min(30, max(5, len(activations_array) // 5))
            self.logger.info(f"Using t-SNE with adaptive perplexity: {optimal_perplexity}")
            
            embeddings_2d = self.controller.analyzer.reduce_dimensions(
                activations_array, method="tsne", n_components=2, perplexity=optimal_perplexity
            )
            
            embeddings_3d = self.controller.analyzer.reduce_dimensions(
                activations_array, method="tsne", n_components=3, perplexity=optimal_perplexity
            )
            
            self.embeddings = {
                "2d": embeddings_2d,
                "3d": embeddings_3d
            }
            
            self.logger.info(f"Dimensionality reduction completed: created 2D and 3D embeddings")
            
            # Log statistics of embeddings
            self.logger.info(f"2D embedding statistics: mean={np.mean(embeddings_2d):.6f}, "
                        f"std={np.std(embeddings_2d):.6f}")
            self.logger.info(f"3D embedding statistics: mean={np.mean(embeddings_3d):.6f}, "
                        f"std={np.std(embeddings_3d):.6f}")
            
        except Exception as e:
            self.logger.error("Failed to perform dimensionality reduction")
            self.logger.error(traceback.format_exc())
            raise
        
        # Perform phase-based affinity propagation clustering
        try:
            self.logger.info("Performing phase-based clustering with natural emergence...")
            
            # Standardize data for numerical stability
            self.logger.info("Standardizing data for numerical stability...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_activations = scaler.fit_transform(activations_array)
            self.logger.info(f"Data standardized: mean={np.mean(scaled_activations):.6f}, "
                        f"std={np.std(scaled_activations):.6f}")
            
            # Create similarity matrix based on cosine similarity
            self.logger.info("Computing cosine similarity matrix to preserve phase relationships...")
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(scaled_activations)
            
            # Log similarity matrix properties
            self.logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
            self.logger.info(f"Similarity statistics: mean={np.mean(similarity_matrix):.6f}, "
                        f"min={np.min(similarity_matrix):.6f}, max={np.max(similarity_matrix):.6f}")
            
            # Measure topological properties before clustering
            self.logger.info("Measuring pre-clustering topological properties...")
            pre_topo = self.controller.analyzer.compute_topological_features(
                scaled_activations, n_neighbors=min(15, scaled_activations.shape[0]-1)
            )
            self.logger.info(f"Pre-clustering topological properties:")
            self.logger.info(f"  - betti_0: {pre_topo['betti_0']}")
            self.logger.info(f"  - betti_1: {pre_topo['betti_1']}")
            self.logger.info(f"  - spectral_gap: {pre_topo['spectral_gap']:.6f}")
            self.logger.info(f"  - clustering_coefficient: {pre_topo['clustering_coefficient']:.6f}")
            
            # Analyze eigenvalue structure of similarity matrix
            self.logger.info("Analyzing eigenvalue structure of similarity matrix...")
            eigenvalues = np.linalg.eigvalsh(similarity_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Log eigenvalue spectrum
            self.logger.info(f"Top 10 eigenvalues of similarity matrix: {sorted_eigenvalues[:10]}")
            
            # Check for golden ratio patterns in eigenvalue ratios
            similarity_eigenratios = sorted_eigenvalues[:-1] / sorted_eigenvalues[1:]
            
            self.logger.info("Checking for golden ratio patterns in similarity eigenvalue distribution...")
            similarity_phi_patterns = []
            for i, ratio in enumerate(similarity_eigenratios[:20]):  # Check first 20 ratios
                if abs(ratio - golden_ratio) < adaptive_phi_tolerance:
                    pattern_type = "phi"
                    similarity_phi_patterns.append((i, ratio, pattern_type))
                    self.logger.info(f"Golden ratio pattern at indices {i}:{i+1}: {ratio:.6f} ({pattern_type})")
                elif abs(ratio - golden_ratio_inverse) < adaptive_phi_tolerance:
                    pattern_type = "1/phi"
                    similarity_phi_patterns.append((i, ratio, pattern_type))
                    self.logger.info(f"Golden ratio pattern at indices {i}:{i+1}: {ratio:.6f} ({pattern_type})")
            
            # Run affinity propagation with adaptive parameters
            self.logger.info("Running affinity propagation with adaptive parameters...")
            
            from sklearn.cluster import AffinityPropagation
            
            # Adaptively set preference based on similarity distribution
            preference = np.median(similarity_matrix)
            self.logger.info(f"Setting preference to median similarity: {preference:.6f}")
            
            # Set damping based on eigenvalue spread to improve stability
            eigenvalue_spread = sorted_eigenvalues[0] / max(abs(sorted_eigenvalues[-1]), 1e-10)
            damping = min(0.9, max(0.5, 0.5 + np.log10(eigenvalue_spread) / 10))
            self.logger.info(f"Setting adaptive damping: {damping:.4f} based on eigenvalue spread: {eigenvalue_spread:.6f}")
            
            # Run affinity propagation
            ap = AffinityPropagation(
                affinity='precomputed',  # Use our cosine similarity matrix
                preference=preference,
                damping=damping,
                max_iter=500,
                convergence_iter=30,
                random_state=42
            )
            
            # Fit the model with proper error handling
            try:
                ap_labels = ap.fit_predict(similarity_matrix)
                ap_clusters = len(set(ap_labels))
                self.logger.info(f"Affinity propagation found {ap_clusters} natural clusters")
                
                if ap_clusters < 2:
                    self.logger.warning(f"Affinity propagation found only {ap_clusters} cluster(s)")
                    self.logger.info("Adjusting parameters to encourage more clusters...")
                    
                    # Try with lower preference to encourage more clusters
                    lower_preference = np.percentile(similarity_matrix, 10)
                    self.logger.info(f"Setting lower preference: {lower_preference:.6f}")
                    
                    ap = AffinityPropagation(
                        affinity='precomputed',
                        preference=lower_preference,
                        damping=damping,
                        max_iter=500,
                        convergence_iter=30,
                        random_state=42
                    )
                    ap_labels = ap.fit_predict(similarity_matrix)
                    ap_clusters = len(set(ap_labels))
                    self.logger.info(f"Adjusted affinity propagation found {ap_clusters} clusters")
            
            except Exception as ap_error:
                self.logger.error(f"Error in affinity propagation: {str(ap_error)}")
                self.logger.error(traceback.format_exc())
                
                # Try with standard parameters on the original data if similarity matrix fails
                self.logger.info("Falling back to standard affinity propagation on original data...")
                ap = AffinityPropagation(random_state=42)
                ap_labels = ap.fit_predict(scaled_activations)
                ap_clusters = len(set(ap_labels))
                self.logger.info(f"Fallback affinity propagation found {ap_clusters} clusters")
            
            # Compute detailed cluster statistics
            cluster_sizes = {}
            for label in set(ap_labels):
                cluster_sizes[label] = sum(1 for l in ap_labels if l == label)
            
            # Sort clusters by size (descending)
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # Log cluster size distribution
            self.logger.info("Cluster size distribution:")
            for i, (cluster, size) in enumerate(sorted_clusters[:min(10, len(sorted_clusters))]):
                self.logger.info(f"  - Cluster {cluster}: {size} points ({size/len(ap_labels)*100:.2f}%)")
            
            if len(sorted_clusters) > 10:
                remaining_clusters = len(sorted_clusters) - 10
                remaining_points = sum(size for _, size in sorted_clusters[10:])
                self.logger.info(f"  - ... and {remaining_clusters} more clusters with {remaining_points} points "
                            f"({remaining_points/len(ap_labels)*100:.2f}%)")
            
            # Compute personality distribution within top clusters
            self.logger.info("Analyzing personality distributions within top clusters...")
            
            # Map personality labels to integers for easier processing
            unique_personalities = sorted(list(set(personality_labels)))
            personality_indices = {p: i for i, p in enumerate(unique_personalities)}
            personality_numeric = [personality_indices[p] for p in personality_labels]
            
            # Track personality distribution for top clusters
            top_cluster_distributions = {}
            for cluster_id, _ in sorted_clusters[:min(10, len(sorted_clusters))]:
                cluster_indices = [i for i, label in enumerate(ap_labels) if label == cluster_id]
                cluster_personalities = [personality_labels[i] for i in cluster_indices]
                
                # Count personalities in this cluster
                personality_counts = {}
                for p in cluster_personalities:
                    personality_counts[p] = personality_counts.get(p, 0) + 1
                    
                top_cluster_distributions[cluster_id] = personality_counts
                
                # Log personality distribution for this cluster
                total = len(cluster_personalities)
                self.logger.info(f"Cluster {cluster_id} personality distribution ({total} points):")
                for p, count in sorted(personality_counts.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"  - {p}: {count} points ({count/total*100:.2f}%)")
            
            # Compute cluster purity
            ap_purity = self._compute_cluster_purity(ap_labels, personality_labels)
            self.logger.info(f"Overall cluster purity: {ap_purity:.6f}")
            
            # Compute normalized purity relative to random assignment
            expected_random_purity = 1 / len(set(personality_labels))
            normalized_purity = (ap_purity - expected_random_purity) / (1 - expected_random_purity)
            self.logger.info(f"Expected random purity: {expected_random_purity:.6f}")
            self.logger.info(f"Normalized purity: {normalized_purity:.6f}")
            
            # Compute silhouette score if we have multiple clusters
            if ap_clusters > 1:
                from sklearn.metrics import silhouette_score
                ap_silhouette = silhouette_score(scaled_activations, ap_labels)
                self.logger.info(f"Silhouette score: {ap_silhouette:.6f}")
            else:
                ap_silhouette = 0
                self.logger.warning("Cannot compute silhouette score with only one cluster")
            
            # Measure topological properties after clustering
            self.logger.info("Measuring post-clustering topological properties...")
            
            # Calculate a per-cluster representation (centroid)
            cluster_centroids = np.zeros((ap_clusters, scaled_activations.shape[1]))
            for label in range(ap_clusters):
                indices = [i for i, l in enumerate(ap_labels) if l == label]
                if indices:
                    cluster_centroids[label] = np.mean(scaled_activations[indices], axis=0)
            
            # Calculate centroid distances
            from scipy.spatial.distance import pdist, squareform
            if ap_clusters > 1:
                centroid_distances = squareform(pdist(cluster_centroids))
                mean_centroid_distance = np.mean(centroid_distances)
                self.logger.info(f"Mean distance between cluster centers: {mean_centroid_distance:.6f}")
            else:
                mean_centroid_distance = 0
                self.logger.warning("Cannot compute centroid distances with only one cluster")
            
            # Compute post-clustering topological features if we have more than one cluster
            if ap_clusters > 1:
                post_topo = self.controller.analyzer.compute_topological_features(
                    cluster_centroids, n_neighbors=min(5, ap_clusters-1)
                )
                self.logger.info(f"Post-clustering topological properties:")
                self.logger.info(f"  - betti_0: {post_topo['betti_0']}")
                self.logger.info(f"  - betti_1: {post_topo['betti_1']}")
                self.logger.info(f"  - spectral_gap: {post_topo['spectral_gap']:.6f}")
                self.logger.info(f"  - clustering_coefficient: {post_topo['clustering_coefficient']:.6f}")
                
                # Compare pre and post topological properties
                topo_change_betti0 = post_topo['betti_0'] - pre_topo['betti_0']
                topo_change_betti1 = post_topo['betti_1'] - pre_topo['betti_1']
                spectral_gap_change = (post_topo['spectral_gap'] - pre_topo['spectral_gap']) / max(pre_topo['spectral_gap'], 1e-10)
                
                self.logger.info(f"Topological changes after clustering:")
                self.logger.info(f"  - Betti_0 change: {topo_change_betti0}")
                self.logger.info(f"  - Betti_1 change: {topo_change_betti1}")
                self.logger.info(f"  - Spectral gap relative change: {spectral_gap_change:.6f}")
                
                # Store post-clustering topological features
                post_topo_features = post_topo
            else:
                # Use pre-clustering features if only one cluster
                post_topo_features = pre_topo
                self.logger.warning("Using pre-clustering topological features as only one cluster was found")
            
            # Store clustering results
            clustering_result = {
                "labels": ap_labels.tolist() if isinstance(ap_labels, np.ndarray) else ap_labels,
                "n_clusters": ap_clusters,
                "cluster_sizes": cluster_sizes,
                "purity": float(ap_purity),
                "silhouette_score": float(ap_silhouette) if ap_clusters > 1 else 0,
                "normalized_purity": float(normalized_purity),
                "expected_random_purity": float(expected_random_purity),
                "mean_cluster_distance": float(mean_centroid_distance) if ap_clusters > 1 else 0,
                "centers": cluster_centroids.tolist() if ap_clusters > 1 else []
            }
            
            # Set metrics for the experiment
            self.set_metric("optimal_n_clusters", ap_clusters)
            self.set_metric("expected_n_clusters", len(set(personality_labels)))
            self.set_metric("cluster_purity", ap_purity)
            self.set_metric("expected_random_purity", expected_random_purity)
            self.set_metric("normalized_purity", normalized_purity)
            self.set_metric("silhouette_score", ap_silhouette)
            self.set_metric("mean_cluster_distance", mean_centroid_distance)
            self.set_metric("optimal_clustering_method", "phase_based_affinity_propagation")
            
            # Log clustering completion
            self.logger.info(f"Clustering completed: found {ap_clusters} natural clusters with purity={ap_purity:.6f}")
            
            # Store raw data for analysis
            self.add_raw_data("clustering_result", clustering_result)
            self.add_raw_data("eigenvalue_analysis", {
                "eigenvalues": eigenvalues.tolist(),
                "eigenratios": eigenratios.tolist()[:20],
                "golden_patterns": golden_patterns
            })
            self.add_raw_data("similarity_analysis", {
                "similarity_eigenvalues": sorted_eigenvalues.tolist()[:20],
                "similarity_eigenratios": similarity_eigenratios.tolist()[:20],
                "similarity_phi_patterns": similarity_phi_patterns
            })
            self.add_raw_data("personality_distributions", top_cluster_distributions)
            
        except Exception as e:
            self.logger.error("Failed to perform phase-based clustering")
            self.logger.error(traceback.format_exc())
            raise
        
        # Analyze field-like properties
        try:
            self.logger.info("Analyzing field-like properties...")
            
            # Compute field coherence metrics
            coherence_metrics = self.controller.analyzer.measure_field_coherence(activations_array)
            # Log field coherence measurements
            self.experiment_logger.log_field_measurement(
                measurement_type="field_coherence",
                measurement_value=coherence_metrics.get("phase_coherence", 0),
                measurement_details=coherence_metrics
            )

            
            # Log detailed coherence metrics
            self.logger.info("Field coherence metrics:")
            for key, value in coherence_metrics.items():
                self.logger.info(f"  - {key}: {value}")
                
            # Add metrics to results
            for key, value in coherence_metrics.items():
                self.set_metric(f"field_{key}", value)
            
            # Analyze dimensional organization with detailed logging
            self.logger.info("Analyzing dimensional hierarchy...")
            dim_hierarchy = self.controller.analyzer.analyze_dimensional_hierarchy(activations_array)

            self.experiment_logger.log_field_measurement(
                measurement_type="dimensional_hierarchy",
                measurement_value=len(dim_hierarchy.get("golden_ratio_patterns", [])),
                measurement_details=dim_hierarchy
            )
                        
            # Log dimensional hierarchy details
            if "dimension_thresholds" in dim_hierarchy:
                self.logger.info(f"Natural dimension thresholds: {dim_hierarchy['dimension_thresholds']}")
                
            if "compression_cascade" in dim_hierarchy:
                for compression_step in dim_hierarchy["compression_cascade"]:
                    from_dim = compression_step.get("from_dim", 0)
                    to_dim = compression_step.get("to_dim", 0)
                    
                    self.experiment_logger.log_dimensional_compression(
                        from_dimensions=from_dim,
                        to_dimensions=to_dim,
                        compression_metrics={
                            "compression_ratio": compression_step.get("compression_ratio", 0),
                            "info_preservation": compression_step.get("info_preservation", 0),
                            "quantum_efficiency": compression_step.get("quantum_efficiency", 0)
                        }
                    )
            
            # Check for golden ratio patterns with detailed logging
            golden_ratio_patterns = dim_hierarchy.get("golden_ratio_patterns", [])
            if golden_ratio_patterns:
                self.logger.info(f"Found {len(golden_ratio_patterns)} golden ratio patterns in dimensional organization:")
                for pattern in golden_ratio_patterns:
                    self.logger.info(f"  - {pattern}")
            else:
                self.logger.info("No golden ratio patterns found in dimensional organization")
                
            self.set_metric("golden_ratio_pattern_count", len(golden_ratio_patterns))
            self.set_metric("dimensional_compression_steps", len(dim_hierarchy.get("compression_cascade", [])))
            
            # Compute topological features with detailed logging
            self.logger.info("Computing topological features...")
            n_neighbors = min(15, len(activations_array)-1)
            topo_features = pre_topo  # Use the previously computed topological features

            self.experiment_logger.log_field_measurement(
                measurement_type="topological_features",
                measurement_value=topo_features.get("quantum_field_score", 0),
                measurement_details=topo_features
            )
            
            # Log topological features in detail
            self.logger.info("Topological features:")
            for key, value in topo_features.items():
                if not isinstance(value, (list, dict)):
                    self.logger.info(f"  - {key}: {value}")
            
            # Calculate continuous quantum score instead of categorical protection level
            betti_0 = topo_features.get("betti_0", 1)
            betti_1 = topo_features.get("betti_1", 0)
            spectral_gap = topo_features.get("spectral_gap", 0)
            clustering_coefficient = topo_features.get("clustering_coefficient", 0)
            
            quantum_field_score = spectral_gap * clustering_coefficient * (1 + betti_1 / max(1, betti_0))
            self.logger.info(f"Calculated continuous quantum field score: {quantum_field_score:.6f}")
            self.set_metric("quantum_field_score", float(quantum_field_score))
            
            # Add key topological metrics
            self.set_metric("topological_protection", topo_features.get("topological_protection"))
            self.set_metric("estimated_topological_charge", topo_features.get("estimated_topological_charge"))
            self.set_metric("betti_0", topo_features.get("betti_0"))
            self.set_metric("betti_1", topo_features.get("betti_1"))
            self.set_metric("spectral_gap", topo_features.get("spectral_gap"))
            self.set_metric("clustering_coefficient", topo_features.get("clustering_coefficient"))
            
            self.logger.info("Field analysis completed")
            
            # Store raw analysis results
            self.add_raw_data("coherence_metrics", coherence_metrics)
            self.add_raw_data("dimensional_hierarchy", dim_hierarchy)
            self.add_raw_data("topological_features", topo_features)
            
        except Exception as e:
            self.logger.error("Failed to analyze field-like properties")
            self.logger.error(traceback.format_exc())
            raise
        
        # Generate visualizations
        try:
            self.logger.info("Generating visualizations...")
            
            # Create 2D embedding visualization
            embedding_path = self.controller.visualizer.plot_activation_space(
                self.embeddings["2d"], personality_labels, 
                title="Personality Activation Space (2D)", dim=2
            )
            self.add_visualization("personality_activation_space_2d", embedding_path)
            
            # Create 3D embedding visualization
            embedding_3d_path = self.controller.visualizer.plot_activation_space(
                self.embeddings["3d"], personality_labels, 
                title="Personality Activation Space (3D)", dim=3
            )
            self.add_visualization("personality_activation_space_3d", embedding_3d_path)
            
            # Create dimensional hierarchy visualization
            dim_hierarchy_path = self.controller.visualizer.visualize_dimensional_hierarchy(
                dim_hierarchy, title="Personality Dimensional Hierarchy"
            )
            self.add_visualization("dimensional_hierarchy", dim_hierarchy_path)
            
            # Create eigenvalue distribution visualization
            eigenvalue_path = self.controller.visualizer.plot_eigenvalue_distribution(
                np.array(dim_hierarchy.get("eigenvalues", [])), 
                title="Personality Eigenvalue Distribution"
            )
            self.add_visualization("eigenvalue_distribution", eigenvalue_path)
            
            self.logger.info("Visualizations completed")
            
        except Exception as e:
            self.logger.error("Failed to generate visualizations")
            self.logger.error(traceback.format_exc())
            raise
        
        # Interpret results
        self._interpret_results()
        
        # Return results
        return self.results


    def perform_phase_based_clustering(self, activations_array, personality_labels):
        """
        Perform phase-based clustering using affinity propagation with enhanced logging.
        
        This method implements clustering based on quantum field principles:
        1. Works with phase relationships through cosine similarity
        2. Allows natural cluster emergence without imposing a specific count
        3. Provides detailed logging of the clustering structure
        4. Monitors topological properties before and after clustering
        
        Args:
            activations_array: Array of activation patterns [samples, features]
            personality_labels: True personality labels
            
        Returns:
            Dictionary containing clustering results with detailed metrics
        """
        self.logger.info("Performing phase-based quantum clustering with natural emergence...")
        
        # First log detailed activation properties
        self.logger.info(f"Activation matrix shape: {activations_array.shape}")
        self.logger.info(f"Activation statistics: mean={np.mean(activations_array):.6f}, "
                        f"std={np.std(activations_array):.6f}")
        
        # Create standardized version for numerical stability
        self.logger.info("Standardizing activations for numerical stability...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_activations = scaler.fit_transform(activations_array)
        
        # Create similarity matrix based on cosine similarity
        self.logger.info("Computing cosine similarity matrix to preserve phase relationships...")
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(scaled_activations)
        
        # Log similarity matrix properties
        self.logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        self.logger.info(f"Similarity statistics: mean={np.mean(similarity_matrix):.6f}, "
                        f"min={np.min(similarity_matrix):.6f}, max={np.max(similarity_matrix):.6f}")
        
        # Measure topological properties before clustering
        self.logger.info("Measuring pre-clustering topological properties...")
        pre_topo = self.controller.analyzer.compute_topological_features(scaled_activations)
        self.logger.info(f"Pre-clustering betti_0: {pre_topo['betti_0']}, betti_1: {pre_topo['betti_1']}")
        self.logger.info(f"Pre-clustering spectral_gap: {pre_topo['spectral_gap']:.6f}")
        self.logger.info(f"Pre-clustering clustering_coefficient: {pre_topo['clustering_coefficient']:.6f}")
        
        # Analyze eigenvalue structure of similarity matrix
        self.logger.info("Analyzing eigenvalue structure of similarity matrix...")
        eigenvalues = np.linalg.eigvalsh(similarity_matrix)
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Log eigenvalue spectrum
        self.logger.info(f"Top 10 eigenvalues: {sorted_eigenvalues[:10]}")
        
        # Check for golden ratio patterns in eigenvalue ratios
        eigenratios = sorted_eigenvalues[:-1] / sorted_eigenvalues[1:]
        golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
        
        self.logger.info("Checking for golden ratio patterns in eigenvalue distribution...")
        phi_patterns = []
        for i, ratio in enumerate(eigenratios[:20]):  # Check first 20 ratios
            if abs(ratio - golden_ratio) < 0.05 or abs(ratio - (1/golden_ratio)) < 0.05:
                pattern_type = "phi" if abs(ratio - golden_ratio) < 0.05 else "1/phi"
                phi_patterns.append((i, ratio, pattern_type))
                self.logger.info(f"Golden ratio pattern at indices {i}:{i+1}: {ratio:.6f} ({pattern_type})")
        
        # Run affinity propagation with adaptive parameters
        self.logger.info("Running affinity propagation with adaptive parameters...")
        
        from sklearn.cluster import AffinityPropagation
        
        # Adaptively set preference based on similarity distribution
        preference = np.median(similarity_matrix)
        self.logger.info(f"Setting preference to median similarity: {preference:.6f}")
        
        # Set damping based on eigenvalue spread to improve stability
        eigenvalue_spread = sorted_eigenvalues[0] / sorted_eigenvalues[-1]
        damping = min(0.9, max(0.5, 0.5 + np.log10(eigenvalue_spread) / 10))
        self.logger.info(f"Setting adaptive damping: {damping:.4f} based on eigenvalue spread: {eigenvalue_spread:.6f}")
        
        # Run affinity propagation
        ap = AffinityPropagation(
            affinity='precomputed',  # Use our cosine similarity matrix
            preference=preference,
            damping=damping,
            max_iter=500,
            convergence_iter=30,
            random_state=42
        )
        
        # Fit the model
        ap_labels = ap.fit_predict(similarity_matrix)
        
        # Analyze clustering results
        ap_clusters = len(set(ap_labels))
        self.logger.info(f"Affinity propagation found {ap_clusters} natural clusters")
        
        # Compute detailed cluster statistics
        cluster_sizes = {}
        for label in set(ap_labels):
            cluster_sizes[label] = sum(1 for l in ap_labels if l == label)
        
        # Sort clusters by size (descending)
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Log cluster size distribution
        self.logger.info("Cluster size distribution:")
        for i, (cluster, size) in enumerate(sorted_clusters[:10]):  # Log top 10 clusters
            self.logger.info(f"  - Cluster {cluster}: {size} points ({size/len(ap_labels)*100:.2f}%)")
        
        if len(sorted_clusters) > 10:
            self.logger.info(f"  - ... and {len(sorted_clusters)-10} more clusters")
        
        # Compute personality distribution within clusters
        self.logger.info("Analyzing personality distributions within clusters...")
        
        # Map personality labels to integers for easier processing
        unique_personalities = sorted(list(set(personality_labels)))
        personality_indices = {p: i for i, p in enumerate(unique_personalities)}
        personality_numeric = [personality_indices[p] for p in personality_labels]
        
        # Track personality distribution for top clusters
        top_cluster_distributions = {}
        for cluster_id, _ in sorted_clusters[:min(10, len(sorted_clusters))]:
            cluster_indices = [i for i, label in enumerate(ap_labels) if label == cluster_id]
            cluster_personalities = [personality_labels[i] for i in cluster_indices]
            
            # Count personalities in this cluster
            personality_counts = {}
            for p in cluster_personalities:
                personality_counts[p] = personality_counts.get(p, 0) + 1
                
            top_cluster_distributions[cluster_id] = personality_counts
            
            # Log personality distribution for this cluster
            total = len(cluster_personalities)
            self.logger.info(f"Cluster {cluster_id} personality distribution ({total} points):")
            for p, count in sorted(personality_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {p}: {count} points ({count/total*100:.2f}%)")
        
        # Compute cluster purity
        ap_purity = self._compute_cluster_purity(ap_labels, personality_labels)
        self.logger.info(f"Overall cluster purity: {ap_purity:.6f}")
        
        # Compute normalized purity relative to random assignment
        expected_random_purity = 1 / len(set(personality_labels))
        normalized_purity = (ap_purity - expected_random_purity) / (1 - expected_random_purity)
        self.logger.info(f"Expected random purity: {expected_random_purity:.6f}")
        self.logger.info(f"Normalized purity: {normalized_purity:.6f}")
        
        # Measure topological properties after clustering
        self.logger.info("Measuring post-clustering topological properties...")
        
        # Calculate a per-cluster representation (centroid)
        cluster_centroids = np.zeros((ap_clusters, scaled_activations.shape[1]))
        for label in range(ap_clusters):
            indices = [i for i, l in enumerate(ap_labels) if l == label]
            if indices:
                cluster_centroids[label] = np.mean(scaled_activations[indices], axis=0)
        
        post_topo = self.controller.analyzer.compute_topological_features(cluster_centroids)
        self.logger.info(f"Post-clustering betti_0: {post_topo['betti_0']}, betti_1: {post_topo['betti_1']}")
        self.logger.info(f"Post-clustering spectral_gap: {post_topo['spectral_gap']:.6f}")
        self.logger.info(f"Post-clustering clustering_coefficient: {post_topo['clustering_coefficient']:.6f}")
        
        # Compare pre and post topological properties to determine if protection is needed
        topo_change_betti0 = post_topo['betti_0'] - pre_topo['betti_0']
        topo_change_betti1 = post_topo['betti_1'] - pre_topo['betti_1']
        spectral_gap_change = (post_topo['spectral_gap'] - pre_topo['spectral_gap']) / pre_topo['spectral_gap']
        
        self.logger.info(f"Topological changes after clustering:")
        self.logger.info(f"  - Betti_0 change: {topo_change_betti0}")
        self.logger.info(f"  - Betti_1 change: {topo_change_betti1}")
        self.logger.info(f"  - Spectral gap relative change: {spectral_gap_change:.6f}")
        
        # Determine if topological protection is needed
        protection_needed = (abs(spectral_gap_change) > 0.2 or
                            (topo_change_betti0 != 0) or
                            (abs(topo_change_betti1 / max(1, pre_topo['betti_1'])) > 0.2))
        
        self.logger.info(f"Topological protection needed: {protection_needed}")
        
        # Prepare result dictionary
        result = {
            "labels": ap_labels.tolist() if isinstance(ap_labels, np.ndarray) else ap_labels,
            "n_clusters": int(ap_clusters),
            "purity": float(ap_purity),
            "normalized_purity": float(normalized_purity),
            "expected_random_purity": float(expected_random_purity),
            "cluster_sizes": cluster_sizes,
            "topological_properties": {
                "pre_clustering": pre_topo,
                "post_clustering": post_topo,
                "protection_needed": protection_needed
            },
            "eigenvalue_analysis": {
                "top_eigenvalues": sorted_eigenvalues[:20].tolist(),
                "golden_ratio_patterns": len(phi_patterns),
                "phi_pattern_details": phi_patterns
            },
            "cluster_distributions": top_cluster_distributions
        }
        
        self.logger.info(f"Phase-based clustering with natural emergence completed successfully")
        return result
    

    def _compute_cluster_purity(self, cluster_labels, personality_labels) -> float:
        """
        Compute the purity of clusters with respect to personality labels.
        
        Args:
            cluster_labels: Cluster assignment for each activation
            personality_labels: True personality label for each activation
            
        Returns:
            Cluster purity score [0-1]
        """
        # Convert labels to integers for easier processing
        unique_personalities = sorted(list(set(personality_labels)))
        personality_indices = [unique_personalities.index(p) for p in personality_labels]
        
        # Count the most common personality in each cluster
        n_samples = len(cluster_labels)
        cluster_personality_counts = {}
        
        for i, cluster in enumerate(cluster_labels):
            personality = personality_indices[i]
            
            if cluster not in cluster_personality_counts:
                cluster_personality_counts[cluster] = {}
            
            cluster_personality_counts[cluster][personality] = (
                cluster_personality_counts[cluster].get(personality, 0) + 1
            )
        
        # Compute purity
        correct = 0
        for cluster, counts in cluster_personality_counts.items():
            correct += max(counts.values())
        
        purity = correct / n_samples
        return purity
    
    def _interpret_results(self):
        """
        Interpret the analysis results and add findings based on natural thresholds.
        """
        self.findings = []
        
        # Get key metrics with fallback values
        purity = self.results["metrics"].get("cluster_purity", 0)
        normalized_purity = self.results["metrics"].get("normalized_purity", 0)
        expected_random_purity = self.results["metrics"].get("expected_random_purity", 0)
        gr_count = self.results["metrics"].get("golden_ratio_pattern_count", 0)
        phase_coherence = self.results["metrics"].get("field_phase_coherence", 0)
        quantum_field_score = self.results["metrics"].get("quantum_field_score", 0)
        
        # Log all metrics being used for interpretation
        self.logger.info("Interpreting results using these metrics:")
        self.logger.info(f"  - purity: {purity:.4f}")
        self.logger.info(f"  - normalized_purity: {normalized_purity:.4f}")
        self.logger.info(f"  - expected_random_purity: {expected_random_purity:.4f}")
        self.logger.info(f"  - golden_ratio_pattern_count: {gr_count}")
        self.logger.info(f"  - field_phase_coherence: {phase_coherence:.4f}")
        self.logger.info(f"  - quantum_field_score: {quantum_field_score:.4f}")
        
        # Check if optimal clusters match expected clusters
        optimal_n_clusters = self.results["metrics"].get("optimal_n_clusters")
        expected_n_clusters = self.results["metrics"].get("expected_n_clusters")
        
        if optimal_n_clusters is not None and expected_n_clusters is not None:
            if optimal_n_clusters == expected_n_clusters:
                self.findings.append(
                    f"Natural clustering exactly matches the expected {expected_n_clusters} personalities, "
                    f"suggesting distinct personality field states naturally emerge."
                )
            elif optimal_n_clusters > expected_n_clusters:
                self.findings.append(
                    f"Natural clustering found {optimal_n_clusters} distinct states (vs {expected_n_clusters} expected), "
                    f"suggesting sub-personalities or contextual variations within personality fields."
                )
            else:
                self.findings.append(
                    f"Natural clustering found {optimal_n_clusters} distinct states (vs {expected_n_clusters} expected), "
                    f"suggesting some personalities occupy similar regions in the field state."
                )
            
            self.logger.info(f"Cluster count finding added: optimal={optimal_n_clusters}, expected={expected_n_clusters}")
        
        # Check cluster purity using normalized purity to account for random baseline
        if normalized_purity > 0.8:
            self.findings.append(
                f"Strong personality clustering (normalized purity={normalized_purity:.2f}): Personalities form "
                f"highly distinct patterns in activation space, suggesting field-like organization. "
                f"This is {normalized_purity/0.5:.1f}× better than chance."
            )
        elif normalized_purity > 0.5:
            self.findings.append(
                f"Moderate personality clustering (normalized purity={normalized_purity:.2f}): Personalities show "
                f"notable separation in activation space, with some overlap. "
                f"This is {normalized_purity/0.5:.1f}× better than chance."
            )
        else:
            self.findings.append(
                f"Weak personality clustering (normalized purity={normalized_purity:.2f}): Personalities show "
                f"significant overlap in activation space, suggesting more shared than "
                f"distinct patterns. This is only {normalized_purity/0.5:.1f}× better than chance."
            )
        
        self.logger.info(f"Purity finding added: purity={purity:.4f}, normalized={normalized_purity:.4f}")
        
        # Check for golden ratio patterns
        if gr_count > 3:
            self.findings.append(
                f"Found {gr_count} golden ratio patterns in eigenvalue distribution, "
                f"suggesting strong natural mathematical organization of personality dimensions "
                f"based on fundamental constants."
            )
        elif gr_count > 0:
            self.findings.append(
                f"Found {gr_count} golden ratio patterns in eigenvalue distribution, "
                f"suggesting some natural mathematical organization of personality dimensions."
            )
        else:
            self.findings.append(
                "No golden ratio patterns detected in eigenvalue distribution. "
                "Personality dimensions do not show evidence of organization "
                "based on this fundamental constant."
            )
        
        self.logger.info(f"Golden ratio finding added: pattern_count={gr_count}")
        
        # Check topological protection using continuous quantum field score instead of categories
        # Define natural thresholds based on the theoretical maximum
        # (typical values range from 0 to ~2.0 based on the formula)
        if quantum_field_score > 1.0:
            self.findings.append(
                f"High quantum field score ({quantum_field_score:.2f}): Personality patterns exhibit "
                f"quantum-like stability properties similar to topologically protected states."
            )
        elif quantum_field_score > 0.5:
            self.findings.append(
                f"Medium quantum field score ({quantum_field_score:.2f}): Personality patterns show some "
                f"evidence of topological stability, with moderate quantum-like properties."
            )
        else:
            self.findings.append(
                f"Low quantum field score ({quantum_field_score:.2f}): Personality patterns exhibit "
                f"limited evidence of topological protection or quantum-like stability."
            )
        
        self.logger.info(f"Quantum field score finding added: score={quantum_field_score:.4f}")
        
        # Check field coherence
        # Phase coherence is naturally normalized between 0 and 1
        # where 1 is perfect coherence and 0 is no coherence
        theoretical_random_coherence = 1.0 / np.sqrt(len(self.personalities))
        self.logger.info(f"Theoretical random coherence: {theoretical_random_coherence:.4f}")
        
        normalized_coherence = (phase_coherence - theoretical_random_coherence) / (1 - theoretical_random_coherence)
        normalized_coherence = max(0, normalized_coherence)  # Ensure non-negative
        
        if normalized_coherence > 0.6:
            self.findings.append(
                f"High phase coherence ({phase_coherence:.2f}, normalized: {normalized_coherence:.2f}): "
                f"Personality states exhibit strong quantum-like coherence across the activation space, "
                f"{normalized_coherence/0.3:.1f}× above random baseline."
            )
        elif normalized_coherence > 0.3:
            self.findings.append(
                f"Moderate phase coherence ({phase_coherence:.2f}, normalized: {normalized_coherence:.2f}): "
                f"Personality states show some quantum-like coherence across the activation space, "
                f"{normalized_coherence/0.3:.1f}× above random baseline."
            )
        else:
            self.findings.append(
                f"Low phase coherence ({phase_coherence:.2f}, normalized: {normalized_coherence:.2f}): "
                f"Personality states exhibit limited quantum-like coherence, close to what would be "
                f"expected from random patterns."
            )
        
        self.logger.info(f"Coherence finding added: phase_coherence={phase_coherence:.4f}, "
                    f"normalized={normalized_coherence:.4f}")
        
        # Overall quantum field assessment
        # Use a weighted score combining all metrics
        qf_indicators = [
            normalized_purity / 0.8 if normalized_purity > 0 else 0,  # Normalized to ~1.0 at strong level
            min(1.0, gr_count / 3),                                  # Normalized to ~1.0 at strong level
            min(1.0, quantum_field_score / 1.0),                     # Normalized to ~1.0 at strong level
            normalized_coherence / 0.6 if normalized_coherence > 0 else 0  # Normalized to ~1.0 at strong level
        ]
        
        # Calculate weighted score
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights sum to 1.0
        weighted_score = sum(w * s for w, s in zip(weights, qf_indicators))
        
        self.logger.info(f"Quantum field indicators: {qf_indicators}")
        self.logger.info(f"Overall weighted score: {weighted_score:.4f}")
        
        # Record the components and final score
        self.set_metric("qf_indicators", qf_indicators)
        self.set_metric("qf_weights", weights)
        self.set_metric("qf_weighted_score", weighted_score)
        
        # Interpret the overall score
        if weighted_score > 0.8:
            self.findings.append(
                f"Overall quantum field score: {weighted_score:.2f}/1.0 - Personality patterns show "
                f"strong evidence of quantum field-like properties, including distinct clustering, "
                f"golden ratio organization, topological protection, and phase coherence."
            )
        elif weighted_score > 0.5:
            self.findings.append(
                f"Overall quantum field score: {weighted_score:.2f}/1.0 - Personality patterns show "
                f"moderate evidence of quantum field-like properties, with several but not all "
                f"of the expected characteristics present."
            )
        else:
            self.findings.append(
                f"Overall quantum field score: {weighted_score:.2f}/1.0 - Personality patterns show "
                f"limited evidence of quantum field-like properties, behaving more like "
                f"emergent patterns than coherent quantum fields."
            )
        
        self.logger.info(f"Overall finding added: weighted_score={weighted_score:.4f}")
        
        # Log all findings
        self.logger.info("Final interpretation findings:")
        for i, finding in enumerate(self.findings):
            self.logger.info(f"Finding {i+1}: {finding}")
    
    # At the end of the generate_report method in PersonalityMappingExperiment

    def generate_report(self) -> str:
        """
        Generate a report of the experiment results.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating personality mapping experiment report")
        
        # Generate summary
        summary = self.generate_summary()
        self.set_summary(summary)
        
        # Save full trace log for complete end-to-end analysis
        trace_log_path = self.experiment_logger.save_full_log()
        self.logger.info(f"Complete trace log saved to: {trace_log_path}")
        
        # Add trace log path to results
        self.add_raw_data("trace_log_path", trace_log_path)
        
        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.md")
        
        try:
            with open(report_path, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error("Failed to save report")
            self.logger.error(traceback.format_exc())
            raise