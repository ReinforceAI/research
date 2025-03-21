# experiments/dimensional_analysis.py

import os
import numpy as np
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from experiments.base.base_experiment import BaseExperiment
import datetime

class DimensionalAnalysisExperiment(BaseExperiment):
    """
    Experiment to examine how personality information organizes across dimensional hierarchies.
    
    This experiment analyzes eigenvalue distributions and compression ratios to identify
    natural dimensional organization of personality information, looking for golden ratio
    relationships and other mathematical patterns.
    """
    
    def __init__(self, controller, config, output_dir, logger=None):
        super().__init__(controller, config, output_dir, logger)
        
        self.description = (
            "This experiment examines how personality information organizes across dimensional "
            "hierarchies. By analyzing eigenvalue distributions and compression ratios, we can "
            "identify natural dimensional organization of personality information, looking for "
            "golden ratio relationships and other mathematical patterns that would indicate "
            "quantum field-like organization."
        )
        
        # Initialize experiment-specific attributes
        self.personalities = {}
        self.questions = []
        self.activations_by_personality = {}
        self.dimensional_hierarchy = {}
        self.findings = []
        
        # Constants for analysis
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
        self.golden_ratio_inverse = 1 / self.golden_ratio  # ≈ 0.618...
        
        self.logger.info("Dimensional Analysis Experiment initialized")
    
    def setup(self):
        """
        Set up the dimensional analysis experiment.
        """
        self.logger.info("Setting up Dimensional Analysis Experiment")
        
        # Load personalities using the standard method
        all_personalities = self.get_personalities()
        if not all_personalities:
            self.logger.error("No personalities defined in configuration")
            raise ValueError("No personalities defined in configuration")
        
        # Get selected personalities
        personality_names = self.config.get("personalities", [])
        if not personality_names:
            self.logger.error("No personalities selected in experiment configuration")
            raise ValueError("No personalities selected in experiment configuration")
        
        # Map personality names to their descriptions
        for persona in personality_names:
            if isinstance(persona, dict) and "name" in persona and "description" in persona:
                # Handle case where personalities are provided as dicts with name and description
                self.personalities[persona["name"]] = persona["description"]
                self.logger.debug(f"Selected personality: {persona['name']}")
            elif isinstance(persona, str) and persona in all_personalities:
                # Handle case where personalities are provided as names to look up
                self.personalities[persona] = all_personalities[persona]
                self.logger.debug(f"Selected personality: {persona}")
            else:
                self.logger.warning(f"Personality '{persona}' not found in configuration")
        
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
        Run the dimensional analysis experiment.
        
        This involves generating responses for each personality-question pair
        and capturing the resulting activation patterns for dimensional analysis.
        """
        self.logger.info("Running Dimensional Analysis Experiment")
        
        # Initialize storage for activations
        self.activations_by_personality = {}
        
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
                
                # Capture activations for this input
                try:
                    # Generate and capture activations
                    result = self.controller.instrumentor.capture_activations(
                        input_text=input_text,
                        generation_config={
                            'max_new_tokens': 150,
                            'do_sample': True,
                            'temperature': 0.7,
                            'top_p': 0.9
                        }
                    )
                    
                    # Extract activations
                    layer_activations = self.controller.instrumentor.extract_layerwise_features()
                    
                    # Store activations
                    for layer_name, activation in layer_activations.items():
                        # Only process if activation has valid shape
                        if isinstance(activation, np.ndarray) and activation.size > 0:
                            self.activations_by_personality[persona_name].append({
                                "layer": layer_name,
                                "activation": activation,
                                "question_idx": i
                            })
                    
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
        
        self.logger.info(f"Completed data collection for dimensional analysis experiment")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the collected activations to identify dimensional organization patterns
        with detailed logging of natural emergence at each level.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing dimensional organization with natural emergence approach")
        
        # Ensure we have activation data to analyze
        if not self.activations_by_personality:
            self.logger.error("No activations collected to analyze")
            raise ValueError("No activations collected")
        
        # Analyze dimensional hierarchy for each personality
        self.dimensional_hierarchy = {}
        golden_ratio_counts = []
        compression_cascades = []
        eigenvalue_ratios = []
        dimension_thresholds = []
        
        # Log overall dataset properties
        total_activations = sum(len(acts) for acts in self.activations_by_personality.values())
        self.logger.info(f"Analyzing {len(self.activations_by_personality)} personalities with {total_activations} total activations")
        
        for persona_name, activations_list in self.activations_by_personality.items():
            self.logger.info(f"Analyzing dimensional hierarchy for personality: {persona_name}")
            
            if not activations_list:
                self.logger.warning(f"No activations for {persona_name}, skipping analysis")
                continue
            
            # Group activations by layer
            layer_activations = {}
            for item in activations_list:
                layer = item["layer"]
                if layer not in layer_activations:
                    layer_activations[layer] = []
                layer_activations[layer].append(item["activation"])
            
            # For each layer with sufficient activations
            personality_hierarchies = {}
            
            for layer, activations in layer_activations.items():
                if len(activations) < 3:
                    self.logger.debug(f"Skipping layer {layer} due to insufficient activations")
                    continue
                
                self.logger.info(f"Analyzing layer {layer} with {len(activations)} activations")
                
                # Ensure activations have consistent shape
                shapes = [act.shape for act in activations]
                shape_counter = {}
                for shape in shapes:
                    shape_str = str(shape)
                    shape_counter[shape_str] = shape_counter.get(shape_str, 0) + 1
                
                # Find most common shape
                most_common_shape = max(shape_counter.items(), key=lambda x: x[1])[0]
                self.logger.info(f"Most common activation shape: {most_common_shape} ({shape_counter[most_common_shape]}/{len(activations)} activations)")
                
                filtered_activations = [act for act in activations if str(act.shape) == most_common_shape]
                
                if len(filtered_activations) < 3:
                    self.logger.debug(f"Skipping layer {layer} due to insufficient activations with consistent shape")
                    continue
                
                # Stack activations into array
                try:
                    activations_array = np.stack(filtered_activations)
                    
                    # Log basic statistics of activation array
                    self.logger.info(f"Activation statistics: mean={np.mean(activations_array):.6f}, "
                                f"std={np.std(activations_array):.6f}, "
                                f"min={np.min(activations_array):.6f}, "
                                f"max={np.max(activations_array):.6f}")
                    
                    # Perform preliminary PCA to examine natural dimensional properties
                    from sklearn.decomposition import PCA
                    
                    # Get the minimum dimension possible
                    min_dim = min(activations_array.shape[0], activations_array.shape[1])
                    self.logger.info(f"Performing PCA with {min_dim} components to analyze natural dimensional properties")
                    
                    pca = PCA(n_components=min_dim)
                    pca.fit(activations_array)
                    
                    # Analyze eigenvalue distribution
                    eigenvalues = pca.explained_variance_
                    total_variance = np.sum(eigenvalues)
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance_ratio)
                    
                    self.logger.info(f"Total eigenvalue variance: {total_variance:.6f}")
                    self.logger.info(f"Top 5 eigenvalues: {eigenvalues[:5]}")
                    self.logger.info(f"Top 5 explained variance ratios: {explained_variance_ratio[:5]}")
                    
                    # Find natural dimension thresholds based on cumulative variance
                    variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
                    natural_dimensions = {}
                    
                    for threshold in variance_thresholds:
                        dim = np.argmax(cumulative_variance >= threshold) + 1 if any(cumulative_variance >= threshold) else min_dim
                        natural_dimensions[f"{int(threshold*100)}%"] = dim
                        self.logger.info(f"{int(threshold*100)}% variance threshold: {dim} dimensions")
                    
                    dimension_thresholds.append(natural_dimensions)
                    
                    # Calculate eigenvalue ratios to analyze for golden ratio and other patterns
                    if len(eigenvalues) > 1:
                        ratios = eigenvalues[:-1] / eigenvalues[1:]
                        self.logger.info(f"First 5 eigenvalue ratios: {ratios[:5]}")
                        
                        # Store all ratios for global analysis
                        eigenvalue_ratios.extend(ratios)
                        
                        # Analyze distribution of ratios
                        from scipy import stats
                        
                        # Compute kernel density estimate of ratio distribution
                        if len(ratios) > 5:
                            kde = stats.gaussian_kde(ratios)
                            x_grid = np.linspace(0, max(5, max(ratios)), 1000)
                            pdf = kde(x_grid)
                            
                            # Find peaks in density
                            from scipy.signal import find_peaks
                            peaks, _ = find_peaks(pdf)
                            peak_positions = x_grid[peaks]
                            
                            self.logger.info(f"Found {len(peaks)} natural peaks in eigenvalue ratio distribution at: " + 
                                        ", ".join([f"{pos:.4f}" for pos in peak_positions]))
                            
                            # Check if any peak is close to golden ratio or its powers
                            gr_values = [
                                self.golden_ratio,           # φ ≈ 1.618
                                self.golden_ratio_inverse,   # 1/φ ≈ 0.618
                                self.golden_ratio**2,        # φ² ≈ 2.618
                                1/self.golden_ratio**2,      # 1/φ² ≈ 0.382
                                self.golden_ratio + 1,       # φ+1 ≈ 2.618
                                self.golden_ratio - 1        # φ-1 ≈ 0.618
                            ]
                            
                            # Calculate natural tolerance based on the distribution itself
                            ratio_std = np.std(ratios)
                            natural_tolerance = min(0.1, max(0.02, ratio_std * 0.25))
                            self.logger.info(f"Using natural tolerance for golden ratio detection: {natural_tolerance:.6f} "
                                        f"(based on ratio std: {ratio_std:.6f})")
                            
                            # Check for golden ratio relationships
                            gr_detections = []
                            for i, ratio in enumerate(ratios[:20]):  # Check first 20 ratios
                                for j, gr_val in enumerate(gr_values):
                                    gr_name = ["φ", "1/φ", "φ²", "1/φ²", "φ+1", "φ-1"][j]
                                    if abs(ratio - gr_val) < natural_tolerance:
                                        deviation = abs(ratio - gr_val) / gr_val
                                        gr_detections.append({
                                            "indices": f"{i}:{i+1}",
                                            "ratio": float(ratio),
                                            "gr_value": float(gr_val),
                                            "gr_name": gr_name,
                                            "deviation": float(deviation)
                                        })
                                        self.logger.info(f"Golden ratio detection: eigenvalues {i}:{i+1}, ratio={ratio:.6f}, "
                                                    f"close to {gr_name}={gr_val:.6f}, deviation={deviation:.2%}")
                    
                    # Analyze dimensional hierarchy with detailed natural metrics
                    hierarchy = self.controller.analyzer.analyze_dimensional_hierarchy(activations_array)
                    
                    # Enhanced analysis of compression cascade
                    cascade = hierarchy.get("compression_cascade", [])
                    if cascade:
                        self.logger.info(f"Analyzing natural compression cascade with {len(cascade)} steps:")
                        
                        for i, step in enumerate(cascade):
                            from_dim = step.get("from_dim", 0)
                            to_dim = step.get("to_dim", 0)
                            compression_ratio = step.get("compression_ratio", 0)
                            info_preservation = step.get("info_preservation", 0)
                            
                            # Calculate quantum efficiency of compression
                            quantum_efficiency = info_preservation / (1/compression_ratio) if compression_ratio > 0 else 0
                            
                            self.logger.info(f"  Step {i+1}: {from_dim}→{to_dim} dimensions (ratio: {compression_ratio:.4f})")
                            self.logger.info(f"    Information preservation: {info_preservation:.6f}")
                            self.logger.info(f"    Quantum compression efficiency: {quantum_efficiency:.6f}")
                            
                            # Analyze golden ratio relationships in compression
                            gr_values = [
                                self.golden_ratio,          # φ
                                self.golden_ratio_inverse,  # 1/φ
                                self.golden_ratio**2,       # φ²
                                1/self.golden_ratio**2      # 1/φ²
                            ]
                            
                            closest_gr = min(gr_values, key=lambda x: abs(compression_ratio - x))
                            gr_index = gr_values.index(closest_gr)
                            gr_name = ["φ", "1/φ", "φ²", "1/φ²"][gr_index]
                            deviation = abs(compression_ratio - closest_gr) / closest_gr
                            
                            if deviation < 0.2:  # Within 20% of golden ratio value
                                self.logger.info(f"    Compression ratio {compression_ratio:.4f} is close to {gr_name}={closest_gr:.4f} "
                                            f"(deviation: {deviation:.2%})")
                        
                        # Add enhanced compression cascade data
                        enhanced_cascade = []
                        for step in cascade:
                            enhanced_step = step.copy()
                            from_dim = step.get("from_dim", 0)
                            to_dim = step.get("to_dim", 0)
                            compression_ratio = step.get("compression_ratio", 0)
                            info_preservation = step.get("info_preservation", 0)
                            
                            # Add quantum efficiency
                            enhanced_step["quantum_efficiency"] = info_preservation / (1/compression_ratio) if compression_ratio > 0 else 0
                            
                            # Add golden ratio analysis
                            closest_gr = min(gr_values, key=lambda x: abs(compression_ratio - x))
                            gr_index = gr_values.index(closest_gr)
                            enhanced_step["closest_gr"] = float(closest_gr)
                            enhanced_step["closest_gr_name"] = ["φ", "1/φ", "φ²", "1/φ²"][gr_index]
                            enhanced_step["gr_deviation"] = float(abs(compression_ratio - closest_gr) / closest_gr)
                            
                            enhanced_cascade.append(enhanced_step)
                        
                        # Replace cascade with enhanced version
                        hierarchy["compression_cascade"] = enhanced_cascade
                    
                    # Enhanced analysis of golden ratio patterns
                    gr_patterns = hierarchy.get("golden_ratio_patterns", [])
                    if gr_patterns:
                        self.logger.info(f"Found {len(gr_patterns)} golden ratio patterns:")
                        for i, pattern in enumerate(gr_patterns):
                            self.logger.info(f"  Pattern {i+1}: {pattern}")
                    
                    # Store hierarchy
                    personality_hierarchies[layer] = hierarchy
                    
                    # Count golden ratio patterns
                    golden_ratio_counts.append(len(gr_patterns))
                    
                    # Save compression cascade
                    if cascade:
                        compression_cascades.append({
                            "personality": persona_name,
                            "layer": layer,
                            "cascade": enhanced_cascade if 'enhanced_cascade' in locals() else cascade
                        })
                    
                    self.logger.info(f"Analyzed dimensional hierarchy for {persona_name}, layer {layer}: "
                                f"found {len(gr_patterns)} golden ratio patterns")
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze hierarchy for {persona_name}, layer {layer}")
                    self.logger.error(traceback.format_exc())
            
            # Store personality hierarchies
            self.dimensional_hierarchy[persona_name] = personality_hierarchies
        
        # Analyze global dimensional patterns across all personalities
        self.logger.info("Analyzing global dimensional patterns across all personalities")
        
        # Analyze eigenvalue ratio distribution globally
        if eigenvalue_ratios:
            # Convert to numpy array
            all_ratios = np.array(eigenvalue_ratios)
            
            # Remove extreme outliers
            q1, q3 = np.percentile(all_ratios, [25, 75])
            iqr = q3 - q1
            outlier_mask = (all_ratios >= q1 - 3*iqr) & (all_ratios <= q3 + 3*iqr)
            filtered_ratios = all_ratios[outlier_mask]
            
            self.logger.info(f"Analyzing global eigenvalue ratio distribution: {len(filtered_ratios)} ratios "
                        f"(removed {len(all_ratios) - len(filtered_ratios)} outliers)")
            
            # Calculate statistics
            mean_ratio = np.mean(filtered_ratios)
            median_ratio = np.median(filtered_ratios)
            std_ratio = np.std(filtered_ratios)
            
            self.logger.info(f"Global ratio statistics: mean={mean_ratio:.6f}, median={median_ratio:.6f}, std={std_ratio:.6f}")
            
            # Compute kernel density estimate
            from scipy import stats
            
            kde = stats.gaussian_kde(filtered_ratios)
            x_grid = np.linspace(max(0, mean_ratio - 3*std_ratio), mean_ratio + 3*std_ratio, 1000)
            pdf = kde(x_grid)
            
            # Find peaks in density
            from scipy.signal import find_peaks
            peaks, peak_props = find_peaks(pdf, height=0, prominence=0.1*max(pdf))
            peak_positions = x_grid[peaks]
            peak_heights = peak_props["peak_heights"]
            
            self.logger.info(f"Found {len(peaks)} natural peaks in global ratio distribution at: " + 
                        ", ".join([f"{pos:.4f}" for pos in peak_positions]))
            
            # Check if any peak is close to golden ratio or its powers
            gr_values = [
                self.golden_ratio,           # φ ≈ 1.618
                self.golden_ratio_inverse,   # 1/φ ≈ 0.618
                self.golden_ratio**2,        # φ² ≈ 2.618
                1/self.golden_ratio**2,      # 1/φ² ≈ 0.382
                self.golden_ratio + 1,       # φ+1 ≈ 2.618
                self.golden_ratio - 1        # φ-1 ≈ 0.618
            ]
            gr_names = ["φ", "1/φ", "φ²", "1/φ²", "φ+1", "φ-1"]
            
            # Calculate adaptive tolerance based on peak width
            peak_width = np.mean(np.diff(peak_positions)) if len(peak_positions) > 1 else 0.1
            adaptive_tolerance = min(0.15, max(0.05, peak_width * 0.25))
            
            self.logger.info(f"Using adaptive tolerance for global golden ratio detection: {adaptive_tolerance:.6f}")
            
            gr_peaks = []
            for i, pos in enumerate(peak_positions):
                for j, (gr_val, gr_name) in enumerate(zip(gr_values, gr_names)):
                    if abs(pos - gr_val) < adaptive_tolerance:
                        height = peak_heights[i]
                        relative_height = height / max(peak_heights)
                        deviation = abs(pos - gr_val) / gr_val
                        
                        gr_peaks.append({
                            "peak_position": float(pos),
                            "gr_value": float(gr_val),
                            "gr_name": gr_name,
                            "height": float(height),
                            "relative_height": float(relative_height),
                            "deviation": float(deviation)
                        })
                        
                        self.logger.info(f"Golden ratio peak detected: position={pos:.6f}, close to {gr_name}={gr_val:.6f}, "
                                    f"relative height={relative_height:.2%}, deviation={deviation:.2%}")
            
            # Store global ratio analysis
            self.set_metric("global_ratio_mean", float(mean_ratio))
            self.set_metric("global_ratio_median", float(median_ratio))
            self.set_metric("global_ratio_std", float(std_ratio))
            self.set_metric("global_ratio_peaks", [float(p) for p in peak_positions])
            self.set_metric("global_ratio_golden_peaks", len(gr_peaks))
            self.add_raw_data("global_ratio_gr_peaks", gr_peaks)
            
            # Try fitting mixture of gaussians to detect natural clusters in ratio distribution
            try:
                from sklearn.mixture import GaussianMixture
                
                # Reshape for GMM
                X = filtered_ratios.reshape(-1, 1)
                
                # Try to fit multiple components
                best_bic = float('inf')
                best_n_components = 1
                best_gmm = None
                
                # Test 1-4 components
                for n_components in range(1, 5):
                    if len(X) >= n_components * 5:  # Ensure enough data points
                        gmm = GaussianMixture(n_components=n_components, 
                                            random_state=42, 
                                            max_iter=200,
                                            covariance_type='full')
                        gmm.fit(X)
                        bic = gmm.bic(X)
                        
                        self.logger.info(f"GMM with {n_components} components: BIC={bic:.2f}")
                        
                        if bic < best_bic:
                            best_bic = bic
                            best_n_components = n_components
                            best_gmm = gmm
                
                # Log best model
                self.logger.info(f"Best GMM has {best_n_components} components with BIC={best_bic:.2f}")
                
                if best_n_components > 1:
                    # We have evidence of multiple natural clusters
                    self.logger.info("Detected multiple natural clusters in ratio distribution")
                    
                    # Get component means and variances
                    for i, (mean, var) in enumerate(zip(best_gmm.means_.flatten(), 
                                                    best_gmm.covariances_.flatten())):
                        self.logger.info(f"Component {i}: mean={mean:.6f}, std={np.sqrt(var):.6f}, "
                                    f"weight={best_gmm.weights_[i]:.4f}")
                        
                        # Check if component mean is close to golden ratio
                        for j, (gr_val, gr_name) in enumerate(zip(gr_values, gr_names)):
                            if abs(mean - gr_val) < adaptive_tolerance:
                                self.logger.info(f"  Component {i} mean ({mean:.6f}) is close to {gr_name}={gr_val:.6f}")
                    
                    # Store GMM results
                    self.set_metric("ratio_gmm_components", best_n_components)
                    self.set_metric("ratio_gmm_means", best_gmm.means_.flatten().tolist())
                    self.set_metric("ratio_gmm_weights", best_gmm.weights_.tolist())
            
            except Exception as e:
                self.logger.warning(f"Failed to fit GMM to ratio distribution: {str(e)}")
        
        # Analyze natural dimension thresholds across personalities
        if dimension_thresholds:
            self.logger.info("Analyzing natural dimension thresholds across personalities")
            
            # Calculate average dimensions for each threshold
            threshold_dims = {}
            for threshold in ['50%', '75%', '90%', '95%', '99%']:
                dims = [d.get(threshold, 0) for d in dimension_thresholds if threshold in d]
                if dims:
                    avg_dim = np.mean(dims)
                    threshold_dims[threshold] = avg_dim
                    self.logger.info(f"Average dimensions for {threshold} variance: {avg_dim:.2f}")
            
            # Calculate dimensional compression ratios between thresholds
            if len(threshold_dims) > 1:
                thresholds = ['50%', '75%', '90%', '95%', '99%']
                compression_ratios = []
                
                for i in range(len(thresholds) - 1):
                    if thresholds[i] in threshold_dims and thresholds[i+1] in threshold_dims:
                        from_dim = threshold_dims[thresholds[i]]
                        to_dim = threshold_dims[thresholds[i+1]]
                        
                        if to_dim > 0:
                            ratio = from_dim / to_dim
                            compression_ratios.append(ratio)
                            self.logger.info(f"Natural compression ratio {thresholds[i]}→{thresholds[i+1]}: {ratio:.4f}")
                
                if compression_ratios:
                    mean_ratio = np.mean(compression_ratios)
                    self.set_metric("natural_compression_ratio", float(mean_ratio))
                    
                    # Check proximity to golden ratio
                    closest_gr = min(gr_values, key=lambda x: abs(mean_ratio - x))
                    gr_index = gr_values.index(closest_gr)
                    gr_name = gr_names[gr_index]
                    deviation = abs(mean_ratio - closest_gr) / closest_gr
                    
                    self.logger.info(f"Average natural compression ratio {mean_ratio:.4f} compared to {gr_name}={closest_gr:.4f} "
                                f"(deviation: {deviation:.2%})")
                    
                    self.set_metric("natural_compression_gr_name", gr_name)
                    self.set_metric("natural_compression_gr_value", float(closest_gr))
                    self.set_metric("natural_compression_gr_deviation", float(deviation))
        
        # Compute overall metrics
        if golden_ratio_counts:
            mean_gr = np.mean(golden_ratio_counts)
            max_gr = np.max(golden_ratio_counts)
            self.set_metric("mean_golden_ratio_patterns", float(mean_gr))
            self.set_metric("max_golden_ratio_patterns", int(max_gr))
            self.logger.info(f"Golden ratio pattern statistics: mean={mean_gr:.2f}, max={max_gr}")
        
        # Analyze compression cascades
        if compression_cascades:
            # Find common compression ratios
            all_ratios = []
            for cascade_data in compression_cascades:
                cascade = cascade_data["cascade"]
                for step in cascade:
                    if "compression_ratio" in step:
                        all_ratios.append(step["compression_ratio"])
            
            if all_ratios:
                mean_ratio = np.mean(all_ratios)
                median_ratio = np.median(all_ratios)
                self.set_metric("mean_compression_ratio", float(mean_ratio))
                self.set_metric("median_compression_ratio", float(median_ratio))
                
                self.logger.info(f"Compression ratio statistics: mean={mean_ratio:.4f}, median={median_ratio:.4f}")
                
                # Use continuous metric instead of binary classification
                gr_proximity = min(
                    abs(mean_ratio - self.golden_ratio) / self.golden_ratio,
                    abs(mean_ratio - self.golden_ratio_inverse) / self.golden_ratio_inverse,
                    abs(mean_ratio - self.golden_ratio**2) / (self.golden_ratio**2),
                    abs(mean_ratio - (1/self.golden_ratio**2)) / (1/self.golden_ratio**2)
                )
                
                self.set_metric("golden_ratio_proximity", float(gr_proximity))
                self.logger.info(f"Golden ratio proximity metric: {gr_proximity:.4f} (smaller is closer)")
                
                # Calculate quantum alignment score instead of binary flag
                quantum_alignment_score = 1.0 - min(gr_proximity, 1.0)
                self.set_metric("quantum_alignment_score", float(quantum_alignment_score))
                self.logger.info(f"Quantum alignment score: {quantum_alignment_score:.4f} (higher is more aligned)")
            
            # Analyze information preservation
            all_preservation = []
            all_efficiency = []
            for cascade_data in compression_cascades:
                cascade = cascade_data["cascade"]
                for step in cascade:
                    if "info_preservation" in step:
                        all_preservation.append(step["info_preservation"])
                    if "quantum_efficiency" in step:
                        all_efficiency.append(step["quantum_efficiency"])
            
            if all_preservation:
                mean_preservation = np.mean(all_preservation)
                median_preservation = np.median(all_preservation)
                self.set_metric("mean_info_preservation", float(mean_preservation))
                self.set_metric("median_info_preservation", float(median_preservation))
                
                self.logger.info(f"Information preservation statistics: mean={mean_preservation:.4f}, "
                            f"median={median_preservation:.4f}")
            
            if all_efficiency:
                mean_efficiency = np.mean(all_efficiency)
                median_efficiency = np.median(all_efficiency)
                self.set_metric("mean_quantum_efficiency", float(mean_efficiency))
                self.set_metric("median_quantum_efficiency", float(median_efficiency))
                
                self.logger.info(f"Quantum efficiency statistics: mean={mean_efficiency:.6f}, "
                            f"median={median_efficiency:.6f}")
        
        # Store raw data
        self.add_raw_data("dimensional_hierarchy", self.dimensional_hierarchy)
        self.add_raw_data("compression_cascades", compression_cascades)
        self.add_raw_data("eigenvalue_ratios", eigenvalue_ratios)
        self.add_raw_data("dimension_thresholds", dimension_thresholds)
        
        # Generate visualizations
        try:
            self.logger.info("Generating visualizations...")
            
            # For each personality, visualize dimensional hierarchy
            for persona_name, hierarchies in self.dimensional_hierarchy.items():
                # Find the layer with most golden ratio patterns
                best_layer = None
                max_patterns = -1
                
                for layer, hierarchy in hierarchies.items():
                    gr_patterns = hierarchy.get("golden_ratio_patterns", [])
                    if len(gr_patterns) > max_patterns:
                        max_patterns = len(gr_patterns)
                        best_layer = layer
                
                if best_layer:
                    hierarchy = hierarchies[best_layer]
                    
                    # Visualize dimensional hierarchy
                    dim_vis_path = self.controller.visualizer.visualize_dimensional_hierarchy(
                        hierarchy, title=f"Dimensional Hierarchy: {persona_name} ({best_layer})"
                    )
                    
                    self.add_visualization(f"dim_hierarchy_{persona_name}", dim_vis_path)
                    
                    # Visualize eigenvalue distribution
                    eigenvalues = hierarchy.get("eigenvalues", [])
                    if eigenvalues:
                        eigen_vis_path = self.controller.visualizer.plot_eigenvalue_distribution(
                            np.array(eigenvalues), 
                            title=f"Eigenvalue Distribution: {persona_name} ({best_layer})"
                        )
                        
                        self.add_visualization(f"eigenvalues_{persona_name}", eigen_vis_path)
            
            # Visualize compression cascade for the first cascade (if available)
            if compression_cascades:
                cascade_data = compression_cascades[0]
                persona_name = cascade_data["personality"]
                layer = cascade_data["layer"]
                cascade = cascade_data["cascade"]
                
                # Create a simplified hierarchy object with just the cascade
                hierarchy = {
                    "compression_cascade": cascade,
                    "eigenvalues": self.dimensional_hierarchy[persona_name][layer].get("eigenvalues", []),
                    "cumulative_variance": self.dimensional_hierarchy[persona_name][layer].get("cumulative_variance", [])
                }
                
                cascade_vis_path = self.controller.visualizer.visualize_dimensional_hierarchy(
                    hierarchy, title=f"Compression Cascade: {persona_name} ({layer})"
                )
                
                self.add_visualization("compression_cascade", cascade_vis_path)
            
            # Create global eigenvalue ratio distribution visualization
            if "global_ratio_peaks" in self.results["metrics"]:
                peaks = self.results["metrics"]["global_ratio_peaks"]
                
                if eigenvalue_ratios and len(eigenvalue_ratios) > 10:
                    try:
                        import matplotlib.pyplot as plt
                        from scipy import stats
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Filter extreme outliers
                        ratios = np.array(eigenvalue_ratios)
                        q1, q3 = np.percentile(ratios, [25, 75])
                        iqr = q3 - q1
                        mask = (ratios >= q1 - 3*iqr) & (ratios <= q3 + 3*iqr)
                        filtered_ratios = ratios[mask]
                        
                        # Plot histogram
                        bins = min(50, max(10, int(len(filtered_ratios) / 10)))
                        ax.hist(filtered_ratios, bins=bins, alpha=0.5, color='blue')
                        
                        # Plot kernel density estimate
                        kde = stats.gaussian_kde(filtered_ratios)
                        x_grid = np.linspace(max(0, np.min(filtered_ratios) - 0.5), np.max(filtered_ratios) + 0.5, 1000)
                        pdf = kde(x_grid)
                        ax.plot(x_grid, pdf * len(filtered_ratios) * (x_grid[1] - x_grid[0]), 'r-', linewidth=2)
                        
                        # Mark golden ratio values
                        gr_values = [
                            self.golden_ratio,           # φ ≈ 1.618
                            self.golden_ratio_inverse,   # 1/φ ≈ 0.618
                            self.golden_ratio**2,        # φ² ≈ 2.618
                            1/self.golden_ratio**2       # 1/φ² ≈ 0.382
                        ]
                        gr_labels = ["φ", "1/φ", "φ²", "1/φ²"]
                        gr_colors = ['red', 'green', 'purple', 'orange']
                        
                        for gr_val, gr_label, color in zip(gr_values, gr_labels, gr_colors):
                            if gr_val > np.min(filtered_ratios) - 0.5 and gr_val < np.max(filtered_ratios) + 0.5:
                                ax.axvline(x=gr_val, color=color, linestyle='--', alpha=0.7,
                                            label=f"{gr_label} = {gr_val:.6f}")
                        
                        # Mark detected peaks
                        for peak in peaks:
                            ax.axvline(x=peak, color='black', linestyle='-.',
                                        alpha=0.5, linewidth=1)
                        
                        # Add labels and title
                        ax.set_xlabel('Eigenvalue Ratio')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Global Eigenvalue Ratio Distribution')
                        ax.legend()
                        
                        # Add timestamp
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', fontsize=8)
                        
                        # Save visualization
                        ratio_vis_path = os.path.join(self.output_dir, "global_eigenvalue_ratios.png")
                        plt.savefig(ratio_vis_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        self.add_visualization("global_eigenvalue_ratios", ratio_vis_path)
                        self.logger.info(f"Created global eigenvalue ratio visualization at {ratio_vis_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create ratio distribution visualization: {str(e)}")
            
            self.logger.info("Visualizations completed")
            
        except Exception as e:
            self.logger.error("Failed to generate visualizations")
            self.logger.error(traceback.format_exc())
        
        # Interpret results
        self._interpret_results()
        
        # Return results
        return self.results

    def _interpret_results(self):
        """
        Interpret the analysis results and add findings based on natural patterns.
        """
        self.findings = []
        
        # Log start of interpretation
        self.logger.info("Interpreting dimensional analysis results based on natural patterns")
        
        # Get key metrics with appropriate fallbacks
        mean_gr_patterns = self.results["metrics"].get("mean_golden_ratio_patterns", 0)
        max_gr_patterns = self.results["metrics"].get("max_golden_ratio_patterns", 0)
        gr_proximity = self.results["metrics"].get("golden_ratio_proximity", 1.0)
        quantum_alignment = self.results["metrics"].get("quantum_alignment_score", 0.0)
        global_gr_peaks = self.results["metrics"].get("global_ratio_golden_peaks", 0)
        mean_info_preservation = self.results["metrics"].get("mean_info_preservation", 0)
        mean_quantum_efficiency = self.results["metrics"].get("mean_quantum_efficiency", 0)
        
        # Log all metrics being used for interpretation
        self.logger.info("Interpreting results using these metrics:")
        self.logger.info(f"  - mean_golden_ratio_patterns: {mean_gr_patterns:.4f}")
        self.logger.info(f"  - max_golden_ratio_patterns: {max_gr_patterns}")
        self.logger.info(f"  - golden_ratio_proximity: {gr_proximity:.4f}")
        self.logger.info(f"  - quantum_alignment_score: {quantum_alignment:.4f}")
        self.logger.info(f"  - global_ratio_golden_peaks: {global_gr_peaks}")
        self.logger.info(f"  - mean_info_preservation: {mean_info_preservation:.4f}")
        self.logger.info(f"  - mean_quantum_efficiency: {mean_quantum_efficiency:.4f}")
        
        # Analyze ratio distribution findings
        if "ratio_gmm_components" in self.results["metrics"]:
            components = self.results["metrics"]["ratio_gmm_components"]
            if components > 1:
                means = self.results["metrics"].get("ratio_gmm_means", [])
                weights = self.results["metrics"].get("ratio_gmm_weights", [])
                
                # Format the component information
                component_info = ", ".join([f"{means[i]:.3f} (weight: {weights[i]:.2f})" 
                                        for i in range(len(means))])
                
                self.findings.append(
                    f"Discovered {components} natural clusters in the eigenvalue ratio distribution "
                    f"with centers at {component_info}. This multi-modal distribution suggests "
                    f"the presence of distinct organizing principles in dimensional compression "
                    f"rather than random statistical patterns."
                )
            else:
                self.findings.append(
                    "Eigenvalue ratio distribution forms a single natural cluster, suggesting "
                    "a unified organizing principle in dimensional compression."
                )
        
        # Analyze golden ratio peaks in distribution
        if global_gr_peaks > 0:
            self.findings.append(
                f"Detected {global_gr_peaks} peaks in eigenvalue ratio distribution aligned with "
                f"golden ratio values (φ, 1/φ, φ², 1/φ²). This natural alignment with fundamental "
                f"mathematical constants suggests non-random organization of dimensional compression."
            )
        
        # Golden ratio patterns
        if mean_gr_patterns > 0:
            # Use natural thresholds based on distribution rather than fixed thresholds
            if max_gr_patterns >= 5:  # Strong evidence
                self.findings.append(
                    f"Strong golden ratio organization: Found an average of {mean_gr_patterns:.1f} "
                    f"golden ratio patterns per layer, with up to {max_gr_patterns} patterns in "
                    f"a single layer. This suggests natural mathematical organization of "
                    f"personality dimensions consistent with quantum field properties."
                )
            elif max_gr_patterns >= 2:  # Moderate evidence
                self.findings.append(
                    f"Moderate golden ratio organization: Found an average of {mean_gr_patterns:.1f} "
                    f"golden ratio patterns per layer, with up to {max_gr_patterns} patterns in "
                    f"a single layer. This provides evidence of natural mathematical organization "
                    f"of personality dimensions."
                )
            else:  # Weak evidence
                self.findings.append(
                    f"Minimal golden ratio organization: Found an average of {mean_gr_patterns:.1f} "
                    f"golden ratio patterns per layer. This suggests limited mathematical organization "
                    f"of personality dimensions."
                )
        else:
            self.findings.append(
                "No golden ratio patterns detected in personality dimensions."
            )
        
        # Compression ratio analysis using continuous metric
        if "mean_compression_ratio" in self.results["metrics"]:
            mean_ratio = self.results["metrics"]["mean_compression_ratio"]
            median_ratio = self.results["metrics"].get("median_compression_ratio", mean_ratio)
            
            if quantum_alignment > 0.9:  # Very strong alignment
                self.findings.append(
                    f"Compression ratio strongly aligned with golden ratio: The mean compression ratio "
                    f"({mean_ratio:.3f}) is remarkably aligned with golden ratio-based values "
                    f"(quantum alignment score: {quantum_alignment:.2f}). This strong mathematical "
                    f"alignment suggests quantum field-like organization of dimensional compression."
                )
            elif quantum_alignment > 0.7:  # Strong alignment
                self.findings.append(
                    f"Compression ratio well-aligned with golden ratio: The mean compression ratio "
                    f"({mean_ratio:.3f}) shows strong alignment with golden ratio-based values "
                    f"(quantum alignment score: {quantum_alignment:.2f}). This mathematical "
                    f"alignment suggests natural organization of dimensional compression."
                )
            elif quantum_alignment > 0.5:  # Moderate alignment
                self.findings.append(
                    f"Compression ratio moderately aligned with golden ratio: The mean compression ratio "
                    f"({mean_ratio:.3f}) shows moderate alignment with golden ratio-based values "
                    f"(quantum alignment score: {quantum_alignment:.2f})."
                )
            else:  # Weak or no alignment
                self.findings.append(
                    f"Compression ratio not significantly aligned with golden ratio: The mean compression "
                    f"ratio ({mean_ratio:.3f}) shows limited alignment with golden ratio-based values "
                    f"(quantum alignment score: {quantum_alignment:.2f})."
                )
        
        # Natural dimension thresholds
        if "natural_compression_ratio" in self.results["metrics"]:
            natural_ratio = self.results["metrics"]["natural_compression_ratio"]
            gr_name = self.results["metrics"].get("natural_compression_gr_name", "")
            gr_value = self.results["metrics"].get("natural_compression_gr_value", 0)
            gr_deviation = self.results["metrics"].get("natural_compression_gr_deviation", 1.0)
            
            if gr_deviation < 0.1:  # Very close to golden ratio value
                self.findings.append(
                    f"Natural dimension thresholds reveal compression ratio ({natural_ratio:.3f}) "
                    f"remarkably close to {gr_name}={gr_value:.3f} (deviation: {gr_deviation:.1%}). "
                    f"This suggests dimensional organization follows fundamental mathematical principles "
                    f"rather than arbitrary statistical patterns."
                )
            elif gr_deviation < 0.2:  # Moderately close
                self.findings.append(
                    f"Natural dimension thresholds show compression ratio ({natural_ratio:.3f}) "
                    f"aligned with {gr_name}={gr_value:.3f} (deviation: {gr_deviation:.1%}). "
                    f"This suggests dimensional organization may follow mathematical principles."
                )
        
        # Information preservation with natural thresholds
        if mean_info_preservation > 0:
            if mean_info_preservation > 0.7:  # Very high preservation
                self.findings.append(
                    f"Exceptional information preservation: Personality information is preserved with "
                    f"{mean_info_preservation:.1%} fidelity during dimensional compression, suggesting "
                    f"highly efficient encoding of personality features across dimensions."
                )
            elif mean_info_preservation > 0.5:  # High preservation
                self.findings.append(
                    f"High information preservation: Personality information is preserved with "
                    f"{mean_info_preservation:.1%} fidelity during dimensional compression, suggesting "
                    f"efficient encoding of personality features across dimensions."
                )
            elif mean_info_preservation > 0.3:  # Moderate preservation
                self.findings.append(
                    f"Moderate information preservation: Personality information is preserved with "
                    f"{mean_info_preservation:.1%} fidelity during dimensional compression."
                )
            else:  # Low preservation
                self.findings.append(
                    f"Limited information preservation: Personality information is preserved with only "
                    f"{mean_info_preservation:.1%} fidelity during dimensional compression."
                )
        
        # Quantum efficiency
        if mean_quantum_efficiency > 0:
            if mean_quantum_efficiency > 0.8:  # Very high efficiency
                self.findings.append(
                    f"Extraordinary quantum compression efficiency: {mean_quantum_efficiency:.2f} suggests "
                    f"dimensional compression achieves near-optimal balance between information preservation "
                    f"and dimensional reduction, characteristic of quantum field-like organization."
                )
            elif mean_quantum_efficiency > 0.5:  # High efficiency
                self.findings.append(
                    f"High quantum compression efficiency: {mean_quantum_efficiency:.2f} suggests "
                    f"dimensional compression achieves efficient balance between information preservation "
                    f"and dimensional reduction, characteristic of quantum-like organization."
                )
            elif mean_quantum_efficiency > 0.3:  # Moderate efficiency
                self.findings.append(
                    f"Moderate quantum compression efficiency: {mean_quantum_efficiency:.2f} suggests "
                    f"some balance between information preservation and dimensional reduction."
                )
            else:  # Low efficiency
                self.findings.append(
                    f"Limited quantum compression efficiency: {mean_quantum_efficiency:.2f} suggests "
                    f"suboptimal balance between information preservation and dimensional reduction."
                )
        
        # Overall dimensional organization assessment based on combined evidence
        # Calculate a weighted quantum organization score
        indicators = [
            global_gr_peaks > 0,                     # Golden ratio peaks in distribution
            mean_gr_patterns > 1,                    # Multiple golden ratio patterns
            quantum_alignment > 0.7,                 # Strong alignment with golden ratio
            "ratio_gmm_components" in self.results["metrics"] and 
            self.results["metrics"]["ratio_gmm_components"] > 1,  # Multiple ratio components
            mean_info_preservation > 0.3,            # Reasonable information preservation
            mean_quantum_efficiency > 0.3            # Reasonable quantum efficiency
        ]
        
        # Calculate weighted score
        weights = [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]  # Weights sum to 1.0
        score_components = [w * int(i) for w, i in zip(weights, indicators)]
        quantum_organization_score = sum(score_components)
        
        # Record the score and components
        self.set_metric("quantum_organization_score", float(quantum_organization_score))
        self.set_metric("quantum_organization_indicators", [int(i) for i in indicators])
        
        self.logger.info(f"Quantum organization indicators: {indicators}")
        self.logger.info(f"Quantum organization score: {quantum_organization_score:.2f}")
        
        # Overall assessment based on score
        if quantum_organization_score > 0.7:  # Strong evidence
            self.findings.append(
                f"OVERALL ASSESSMENT (Score: {quantum_organization_score:.2f}): Personality activations show "
                f"strong evidence of natural dimensional organization aligned with mathematical constants "
                f"like the golden ratio. This organization enables efficient dimensional compression while "
                f"preserving semantic information, consistent with quantum field-like properties."
            )
        elif quantum_organization_score > 0.4:  # Moderate evidence
            self.findings.append(
                f"OVERALL ASSESSMENT (Score: {quantum_organization_score:.2f}): Personality activations show "
                f"moderate evidence of natural dimensional organization, with some alignment to mathematical "
                f"constants and efficient information compression across dimensions."
            )
        else:  # Weak evidence
            self.findings.append(
                f"OVERALL ASSESSMENT (Score: {quantum_organization_score:.2f}): Personality activations show "
                f"limited evidence of quantum field-like dimensional organization, with dimensional properties "
                f"more consistent with statistical patterns than natural mathematical organization."
            )
        
        # Log all findings
        self.logger.info(f"Generated {len(self.findings)} findings based on natural patterns:")
        for i, finding in enumerate(self.findings):
            self.logger.info(f"Finding {i+1}: {finding}")

    def generate_report(self) -> str:
        """
        Generate a report of the experiment results.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating dimensional analysis experiment report")
        
        # Generate summary
        summary = self.generate_summary()
        self.set_summary(summary)
        
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