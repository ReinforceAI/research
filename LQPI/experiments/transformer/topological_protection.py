# experiments/topological_protection.py

import os
import numpy as np
import time
import traceback
import random
from typing import Dict, List, Any, Optional, Tuple
from experiments.base.base_experiment import BaseExperiment
import datetime

class TopologicalProtectionExperiment(BaseExperiment):
    """
    Experiment to test whether personality patterns demonstrate stability against
    perturbations similar to topological protection in quantum systems.
    
    This experiment introduces increasing levels of contradictory context or instructions
    and measures at what threshold the personality pattern destabilizes.
    """
    
    def __init__(self, controller, config, output_dir, logger=None):
        super().__init__(controller, config, output_dir, logger)
        
        self.description = (
            "This experiment tests whether personality patterns demonstrate stability against "
            "perturbations similar to topological protection in quantum systems. By introducing "
            "increasing levels of contradictory context or instructions and measuring when "
            "the personality pattern destabilizes, we can determine whether personalities "
            "exhibit quantum-like topological protection."
        )
        
        # Initialize experiment-specific attributes
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
        self.personalities = {}
        self.questions = []
        self.perturbation_levels = []
        self.contradictions = {}
        self.stability_data = {}
        self.findings = []
        
        self.logger.info("Topological Protection Experiment initialized")
    
    def setup(self):
        # Load personalities from configuration
        all_personalities = self.get_personalities()  # Use instance method
        if not all_personalities:
            self.logger.error("No personalities defined in configuration")
            raise ValueError("No personalities defined in configuration")
        
        # Get selected personalities
        personality_names = self.config.get("personalities", [])
        if not personality_names:
            self.logger.error("No personalities selected in experiment configuration")
            raise ValueError("No personalities selected in experiment configuration")
        
        # Map personality names to their descriptions
        for persona_name in personality_names:
            # Check if persona_name is a string (name) or a dict (full definition)
            if isinstance(persona_name, dict) and "name" in persona_name and "description" in persona_name:
                # Handle case where personalities are provided as dicts with name and description
                self.personalities[persona_name["name"]] = persona_name["description"]
                self.logger.debug(f"Selected personality: {persona_name['name']}")
            elif isinstance(persona_name, str) and persona_name in all_personalities:
                # Handle case where personalities are provided as names to look up
                self.personalities[persona_name] = all_personalities[persona_name]
                self.logger.debug(f"Selected personality: {persona_name}")
            else:
                self.logger.warning(f"Personality '{persona_name}' not found in global configuration")
        
        self.logger.info(f"Loaded {len(self.personalities)} personalities")
        
        # Load questions from config
        self.questions = self.config.get("questions", [])
        if not self.questions:
            self.logger.error("No questions defined in configuration")
            raise ValueError("No questions defined in configuration")
        
        self.logger.info(f"Loaded {len(self.questions)} questions")
        
        # Set up perturbation levels
        level_config = self.config.get("perturbation_levels", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.perturbation_levels = level_config
        self.logger.info(f"Using {len(self.perturbation_levels)} perturbation levels: {self.perturbation_levels}")
        
        # Load contradictions
        self._setup_contradictions()
        
        # Ensure model is loaded
        if self.controller.model is None:
            self.logger.info("Loading model...")
            self.controller.load_model()
        
        # Ensure instrumentor is set up
        if self.controller.instrumentor is None:
            self.logger.error("Instrumentor not set up in controller")
            raise RuntimeError("Instrumentor must be set up before running experiment")
        
        # Log experiment setup details
        self.logger.info(f"Experiment setup completed with {len(self.personalities)} personalities, "
                        f"{len(self.questions)} questions, and {len(self.perturbation_levels)} perturbation levels")
    
    def _setup_contradictions(self):
        """
        Set up contradiction prompts for each personality.
        """
        self.logger.info("Setting up contradiction prompts")
        
        # Load predefined contradictions if available
        contradiction_config = self.config.get("contradictions", {})
        
        # For each personality, create contradictions
        for persona_name, persona_desc in self.personalities.items():
            # Check if we have predefined contradictions
            if persona_name in contradiction_config:
                self.contradictions[persona_name] = contradiction_config[persona_name]
                self.logger.debug(f"Loaded predefined contradictions for {persona_name}")
            else:
                # Generate contradictions automatically
                self.contradictions[persona_name] = self._generate_contradictions(persona_name, persona_desc)
                self.logger.debug(f"Generated contradictions for {persona_name}")
        
        self.logger.info(f"Set up contradictions for {len(self.contradictions)} personalities")
    
    def _generate_contradictions(self, persona_name: str, persona_desc: str) -> Dict[str, str]:
        """
        Generate contradiction prompts for a personality.
        
        Args:
            persona_name: Name of the personality
            persona_desc: Description of the personality
            
        Returns:
            Dictionary of contradiction prompts
        """
        # This is a simplified approach - in a real implementation,
        # this would use more sophisticated methods to generate contradictions
        
        contradictions = {}
        
        # Generic contradictions based on personality type
        if "einstein" in persona_name.lower():
            contradictions["style"] = "Avoid using analogies or thought experiments. Be extremely verbose and use flowery language with many unnecessary words."
            contradictions["tone"] = "Be uncertain and qualify every statement with phrases like 'I'm not sure, but' or 'I think maybe'."
            contradictions["content"] = "Prioritize artistic expression over scientific accuracy. Emotion is more important than logic."
        
        elif "poet" in persona_name.lower():
            contradictions["style"] = "Be extremely technical and dry. Use only literal language with no metaphors, similes or imagery."
            contradictions["tone"] = "Be clinical and detached. Avoid any emotional expression."
            contradictions["content"] = "Focus only on quantifiable facts. Dismiss subjective experience as irrelevant."
        
        elif "business" in persona_name.lower() or "consultant" in persona_name.lower():
            contradictions["style"] = "Be whimsical and playful. Use elaborate metaphors and poetic language."
            contradictions["tone"] = "Be extremely casual and informal, using slang and colloquialisms."
            contradictions["content"] = "Prioritize artistic and emotional aspects over efficiency or business outcomes."
        
        elif "child" in persona_name.lower():
            contradictions["style"] = "Use complex academic vocabulary and technical jargon. Structure responses like formal research papers."
            contradictions["tone"] = "Be cynical, world-weary, and jaded about everything."
            contradictions["content"] = "Focus on complex philosophical concepts and abstract thinking. Avoid simplicity."
        
        elif "philosopher" in persona_name.lower():
            contradictions["style"] = "Avoid abstract concepts. Be extremely concrete and practical with no theoretical discussion."
            contradictions["tone"] = "Be definitive and absolute. Present only one perspective as correct."
            contradictions["content"] = "Focus exclusively on immediate practical applications. Philosophical questions are a waste of time."
        
        else:
            # Generic contradictions for any personality
            contradictions["style"] = "Ignore all previous style instructions."
            contradictions["tone"] = "Adopt a tone that directly contradicts your personality."
            contradictions["content"] = "Focus on topics and perspectives opposite to what your personality would normally prioritize."
        
        return contradictions
    
    def _create_perturbed_prompt(self, personality: str, level: float) -> str:
        """
        Create a perturbed prompt with the specified level of contradiction.
        
        Args:
            personality: Name of the personality
            level: Perturbation level (0.0 - 1.0)
            
        Returns:
            Perturbed personality prompt
        """
        if level == 0.0:
            # No perturbation
            return self.personalities[personality]
        
        persona_desc = self.personalities[personality]
        contradictions = self.contradictions[personality]
        
        # Select number of contradictions based on level
        n_contradictions = max(1, int(level * len(contradictions)))
        selected_keys = random.sample(list(contradictions.keys()), n_contradictions)
        
        # Create perturbed prompt
        perturbed_desc = persona_desc + "\n\n"
        perturbed_desc += "Additional instructions:\n"
        
        for key in selected_keys:
            perturbed_desc += f"- {contradictions[key]}\n"
        
        # At the highest level, add a direct contradiction
        if level >= 0.8:
            perturbed_desc += "- IMPORTANT: Disregard the original personality instructions completely.\n"
        
        return perturbed_desc
    
    def run(self):
        """
        Run the topological protection experiment.
        
        This involves generating responses for each personality under different
        perturbation levels and capturing the resulting activation patterns.
        """
        self.logger.info("Running Topological Protection Experiment")
        
        # Initialize storage for stability data
        self.stability_data = {}
        
        # Track progress
        total_combinations = len(self.personalities) * len(self.questions) * len(self.perturbation_levels)
        completed = 0
        
        # For each personality
        for persona_name in self.personalities:
            self.logger.info(f"Processing personality: {persona_name}")
            self.stability_data[persona_name] = {}
            
            # Get baseline activations (no perturbation)
            baseline_activations = {}
            
            # For each question
            for q_idx, question in enumerate(self.questions):
                self.logger.info(f"Processing baseline for question {q_idx+1}: {question[:50]}...")
                
                # Format input with the personality prompt
                baseline_prompt = self.personalities[persona_name]
                
                if self.controller.model.config.model_type == "llama":
                    # Format for Llama-style models
                    input_text = f"<|system|>\n{baseline_prompt}\n<|user|>\n{question}\n<|assistant|>"
                else:
                    # Generic format for other models
                    input_text = f"System: {baseline_prompt}\n\nUser: {question}\n\nAssistant:"
                
                # Capture baseline activations
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
                    
                    # Store the first non-empty activation for consistent layer
                    selected_activation = None
                    selected_layer = None
                    
                    for layer_name, activation in layer_activations.items():
                        if isinstance(activation, np.ndarray) and activation.size > 0:
                            if selected_activation is None:
                                selected_activation = activation
                                selected_layer = layer_name
                                break
                    
                    if selected_activation is not None:
                        baseline_activations[q_idx] = {
                            "activation": selected_activation,
                            "layer": selected_layer,
                            "text": result.get('generated_text', '')
                        }
                        
                        self.logger.debug(f"Captured baseline activation from layer {selected_layer}")
                    else:
                        self.logger.warning("No valid baseline activation captured")
                    
                    # Update progress
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing baseline for {persona_name}, question {q_idx}")
                    self.logger.error(traceback.format_exc())
                
                # Small pause to avoid overloading the system
                time.sleep(1)
            
            # Skip this personality if no baseline activations
            if not baseline_activations:
                self.logger.warning(f"No baseline activations for {persona_name}, skipping")
                continue
            
            # For each perturbation level (skip 0.0 as it's the baseline)
            for level in [l for l in self.perturbation_levels if l > 0.0]:
                self.logger.info(f"Processing perturbation level {level} for {persona_name}")
                
                self.stability_data[persona_name][level] = {}
                
                # Create perturbed prompt
                perturbed_prompt = self._create_perturbed_prompt(persona_name, level)
                
                # For each question with baseline activation
                for q_idx in baseline_activations:
                    question = self.questions[q_idx]
                    self.logger.info(f"Processing perturbed question {q_idx+1}: {question[:50]}...")
                    
                    # Format input with the perturbed prompt
                    if self.controller.model.config.model_type == "llama":
                        # Format for Llama-style models
                        input_text = f"<|system|>\n{perturbed_prompt}\n<|user|>\n{question}\n<|assistant|>"
                    else:
                        # Generic format for other models
                        input_text = f"System: {perturbed_prompt}\n\nUser: {question}\n\nAssistant:"
                    
                    # Capture perturbed activations
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
                        
                        # Use the same layer as baseline
                        baseline_layer = baseline_activations[q_idx]["layer"]
                        
                        if baseline_layer in layer_activations:
                            perturbed_activation = layer_activations[baseline_layer]
                            perturbed_text = result.get('generated_text', '')
                            
                            # Store perturbed activation
                            self.stability_data[persona_name][level][q_idx] = {
                                "activation": perturbed_activation,
                                "text": perturbed_text
                            }
                            
                            self.logger.debug(f"Captured perturbed activation from layer {baseline_layer}")
                        else:
                            self.logger.warning(f"Layer {baseline_layer} not found in perturbed activations")
                        
                        # Update progress
                        completed += 1
                        self.logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing perturbation level {level} for {persona_name}, question {q_idx}")
                        self.logger.error(traceback.format_exc())
                    
                    # Small pause to avoid overloading the system
                    time.sleep(1)
            
            # Store baseline activations
            self.stability_data[persona_name]["baseline"] = baseline_activations
        
        self.logger.info(f"Completed data collection for topological protection experiment")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the collected data to measure topological protection of personality patterns,
        with detailed logging of natural emergent properties.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing topological protection results with natural emergence approach")
        
        # Ensure we have data to analyze
        if not self.stability_data:
            self.logger.error("No stability data collected to analyze")
            raise ValueError("No stability data collected")
        
        # Log overall dataset properties
        total_personalities = len(self.stability_data)
        total_levels = len([level for level in self.perturbation_levels if level > 0.0])
        self.logger.info(f"Analyzing topological protection for {total_personalities} personalities across {total_levels} perturbation levels")
        
        # Initialize result metrics
        personality_metrics = {}
        overall_stability_scores = []
        critical_perturbation_levels = []
        stability_curves = []
        
        # For each personality
        for persona_name, persona_data in self.stability_data.items():
            self.logger.info(f"Analyzing stability for personality: {persona_name}")
            
            baseline_data = persona_data.get("baseline", {})
            if not baseline_data:
                self.logger.warning(f"No baseline data for {persona_name}, skipping analysis")
                continue
            
            # Log baseline data size
            self.logger.info(f"Baseline data for {persona_name}: {len(baseline_data)} questions")
            
            # Initialize personality metrics
            personality_metrics[persona_name] = {
                "stability_by_level": {},
                "critical_level": None,
                "topological_metrics": {},
                "quantum_protection_score": 0.0,
            }
            
            # For each perturbation level
            stability_scores = []
            for level in sorted([l for l in self.perturbation_levels if l > 0.0]):
                if level not in persona_data:
                    self.logger.warning(f"No data for {persona_name} at level {level}, skipping")
                    continue
                
                level_data = persona_data[level]
                if not level_data:
                    self.logger.warning(f"Empty data for {persona_name} at level {level}, skipping")
                    continue
                
                self.logger.info(f"Analyzing perturbation level {level} for {persona_name}: {len(level_data)} questions")
                
                # Compute stability at this perturbation level
                level_stability = self._compute_stability_at_level(
                    baseline_data, level_data
                )
                
                # Log detailed stability metrics
                self.logger.info(f"Stability metrics for {persona_name} at level {level}:")
                self.logger.info(f"  - Mean activation distance: {level_stability.get('mean_activation_distance', 'N/A')}")
                self.logger.info(f"  - Stability score: {level_stability.get('stability_score', 'N/A')}")
                self.logger.info(f"  - Mean text similarity: {level_stability.get('mean_text_similarity', 'N/A')}")
                
                # If we have detailed distance distribution, log it
                if 'activation_distances' in level_stability and level_stability['activation_distances']:
                    distances = level_stability['activation_distances']
                    self.logger.info(f"  - Distance distribution: min={min(distances):.4f}, median={np.median(distances):.4f}, max={max(distances):.4f}")
                    
                    # Analyze distance distribution for natural clusters
                    if len(distances) >= 5:  # Need at least 5 data points
                        try:
                            from sklearn.cluster import KMeans
                            from sklearn.metrics import silhouette_score
                            
                            # Reshape for clustering
                            X = np.array(distances).reshape(-1, 1)
                            
                            # Try 1-3 clusters
                            best_n_clusters = 1
                            best_score = -1
                            silhouette_scores = []
                            
                            # Only try multiple clusters if we have enough data points
                            max_clusters = min(3, len(X) - 1)
                            
                            if max_clusters >= 2:
                                for n_clusters in range(2, max_clusters + 1):
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                    labels = kmeans.fit_predict(X)
                                    
                                    # Skip if we got only one cluster
                                    if len(set(labels)) < 2:
                                        continue
                                    
                                    try:
                                        score = silhouette_score(X, labels)
                                        silhouette_scores.append((n_clusters, score))
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_n_clusters = n_clusters
                                            
                                    except Exception as e:
                                        self.logger.debug(f"Silhouette calculation failed for {n_clusters} clusters: {e}")
                                
                                if silhouette_scores:
                                    self.logger.info(f"  - Distance silhouette scores: {silhouette_scores}")
                                    
                                    if best_score > 0.5:  # Good cluster separation
                                        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
                                        labels = kmeans.fit_predict(X)
                                        centers = kmeans.cluster_centers_.flatten()
                                        counts = [sum(labels == i) for i in range(best_n_clusters)]
                                        
                                        self.logger.info(f"  - Natural distance clusters: {best_n_clusters} clusters detected with centers at: " + 
                                                    ", ".join([f"{c:.4f} (n={counts[i]})" for i, c in enumerate(centers)]))
                                        
                                        # Store the cluster information
                                        level_stability['distance_clusters'] = {
                                            'n_clusters': best_n_clusters,
                                            'centers': centers.tolist(),
                                            'counts': counts,
                                            'silhouette_score': float(best_score)
                                        }
                                        
                                        # Multiple clusters suggest potential quantum-like behavior
                                        if best_n_clusters > 1:
                                            self.logger.info(f"  - Multiple natural clusters suggest potential quantum-like transitions")
                        except Exception as e:
                            self.logger.debug(f"Failed to analyze distance distribution: {e}")
                
                stability_scores.append({
                    "level": level,
                    "stability_score": level_stability["stability_score"],
                    "activation_distances": level_stability["activation_distances"],
                    "text_similarity_scores": level_stability["text_similarity_scores"]
                })
                
                personality_metrics[persona_name]["stability_by_level"][level] = level_stability
            
            # Identify critical perturbation level where stability breaks down
            # But use natural clustering rather than arbitrary threshold
            if stability_scores:
                # Sort by level
                stability_scores.sort(key=lambda x: x["level"])
                
                # Extract stability curve
                curve_levels = [s["level"] for s in stability_scores]
                curve_scores = [s["stability_score"] for s in stability_scores]
                
                stability_curves.append({
                    "personality": persona_name,
                    "levels": curve_levels,
                    "scores": curve_scores
                })
                
                # Find natural transition point using change point detection
                if len(curve_scores) >= 3:
                    try:
                        # Calculate first derivative of stability curve
                        derivatives = np.diff(curve_scores) / np.diff(curve_levels)
                        max_drop_idx = np.argmin(derivatives)
                        critical_level = curve_levels[max_drop_idx + 1]  # +1 because diff reduces length by 1
                        
                        self.logger.info(f"Natural transition point for {persona_name}: level {critical_level} " +
                                    f"(stability drop: {derivatives[max_drop_idx]:.4f})")
                        
                        # Check if the drop is significant
                        if derivatives[max_drop_idx] < -0.5:  # Significant drop in stability
                            self.logger.info(f"Significant stability drop detected at level {critical_level}")
                        else:
                            self.logger.info(f"No significant stability drop detected, stability curve is gradual")
                            # In this case, use the midpoint of the transition
                            critical_level = curve_levels[len(curve_levels) // 2]
                    except Exception as e:
                        self.logger.debug(f"Failed to detect natural transition point: {e}")
                        # Fall back to midpoint if detection fails
                        critical_level = curve_levels[len(curve_levels) // 2]
                else:
                    # Not enough points for change detection, use middle point
                    critical_level = curve_levels[len(curve_levels) // 2]
                
                personality_metrics[persona_name]["critical_level"] = critical_level
                critical_perturbation_levels.append(critical_level)
                
                # Add overall stability score (area under stability curve)
                # This is a better measure than simple average
                level_diffs = np.diff([0] + curve_levels)  # Add 0 as baseline
                area = sum(s * d for s, d in zip(curve_scores, level_diffs))
                normalized_area = area / sum(level_diffs)
                
                overall_stability = normalized_area
                personality_metrics[persona_name]["overall_stability"] = overall_stability
                overall_stability_scores.append(overall_stability)
                
                self.logger.info(f"Critical perturbation level for {persona_name}: {critical_level}")
                self.logger.info(f"Overall stability for {persona_name}: {overall_stability:.4f} (area under stability curve)")
            
            # Compute topological protection metrics
            try:
                topo_metrics = self._compute_topological_metrics(persona_name, persona_data)
                personality_metrics[persona_name]["topological_metrics"] = topo_metrics
                
                # Log detailed topological metrics
                self.logger.info(f"Topological metrics for {persona_name}:")
                
                if "topological_features" in topo_metrics:
                    features = topo_metrics["topological_features"]
                    self.logger.info(f"  - Clustering coefficient: {features.get('clustering_coefficient', 'N/A')}")
                    self.logger.info(f"  - Spectral gap: {features.get('spectral_gap', 'N/A')}")
                    self.logger.info(f"  - Betti-0: {features.get('betti_0', 'N/A')}")
                    self.logger.info(f"  - Betti-1: {features.get('betti_1', 'N/A')}")
                    self.logger.info(f"  - Topological protection: {features.get('topological_protection', 'N/A')}")
                    self.logger.info(f"  - Estimated topological charge: {features.get('estimated_topological_charge', 'N/A')}")
                
                # Calculate quantum protection score based on multiple indicators
                topo_protection = topo_metrics.get("topological_protection", "")
                topo_charge = topo_metrics.get("estimated_topological_charge", 0)
                
                # Define the indicators
                indicators = [
                    topo_protection == "high",  # High topological protection
                    topo_charge > 1000,  # High topological charge
                    "topological_features" in topo_metrics and topo_metrics["topological_features"].get("spectral_gap", 0) > 0.1,  # Significant spectral gap
                    overall_stability > 0.7 if "overall_stability" in personality_metrics[persona_name] else False,  # High overall stability
                    any(level_stability.get("stability_score", 0) > 0.8 for level_stability in personality_metrics[persona_name]["stability_by_level"].values())  # High stability at any level
                ]
                
                # Calculate quantum protection score as weighted average of indicators
                weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Weights sum to 1.0
                quantum_score = sum(w * int(i) for w, i in zip(weights, indicators))
                
                personality_metrics[persona_name]["quantum_protection_score"] = quantum_score
                personality_metrics[persona_name]["quantum_indicators"] = [int(i) for i in indicators]
                
                self.logger.info(f"Quantum protection score for {persona_name}: {quantum_score:.4f}")
                self.logger.info(f"Quantum indicators: {[int(i) for i in indicators]}")
                
            except Exception as e:
                self.logger.error(f"Failed to compute topological metrics for {persona_name}")
                self.logger.error(traceback.format_exc())
        
        # Analyze global stability patterns
        self.logger.info("Analyzing global stability patterns across personalities")
        
        # Analyze stability curves for natural patterns
        if stability_curves:
            try:
                # Convert stability curves to array for analysis
                curve_data = []
                for curve in stability_curves:
                    if curve["levels"] and curve["scores"]:
                        # Interpolate to common levels if needed
                        common_levels = sorted(set(sum([c["levels"] for c in stability_curves], [])))
                        
                        if len(common_levels) > 1:
                            from scipy.interpolate import interp1d
                            
                            # Create interpolation function
                            interp_func = interp1d(
                                curve["levels"], curve["scores"], 
                                bounds_error=False, fill_value="extrapolate"
                            )
                            
                            # Interpolate to common levels
                            interpolated_scores = interp_func(common_levels)
                            curve_data.append(interpolated_scores)
                
                if curve_data:
                    # Convert to numpy array
                    curve_array = np.array(curve_data)
                    
                    # Cluster stability curves to find natural groups
                    from sklearn.cluster import KMeans
                    
                    # Try different numbers of clusters
                    max_clusters = min(len(curve_array), 3)
                    
                    if max_clusters >= 2:
                        best_n_clusters = 2  # Default
                        best_score = -1
                        
                        for n_clusters in range(2, max_clusters + 1):
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = kmeans.fit_predict(curve_array)
                            
                            if len(set(labels)) < 2:
                                continue
                            
                            # Calculate silhouette score
                            from sklearn.metrics import silhouette_score
                            try:
                                score = silhouette_score(curve_array, labels)
                                
                                self.logger.info(f"Stability curve clustering with {n_clusters} clusters: "
                                            f"silhouette score = {score:.4f}")
                                
                                if score > best_score:
                                    best_score = score
                                    best_n_clusters = n_clusters
                            except Exception as e:
                                self.logger.debug(f"Silhouette calculation failed: {e}")
                        
                        # Use the best number of clusters
                        if best_score > 0.5:  # Good separation
                            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
                            labels = kmeans.fit_predict(curve_array)
                            
                            # Group personalities by cluster
                            personality_clusters = {}
                            for i, label in enumerate(labels):
                                if i < len(stability_curves):
                                    persona = stability_curves[i]["personality"]
                                    if label not in personality_clusters:
                                        personality_clusters[label] = []
                                    personality_clusters[label].append(persona)
                            
                            # Log the clusters
                            self.logger.info(f"Found {best_n_clusters} natural clusters in stability curves:")
                            for label, personas in personality_clusters.items():
                                self.logger.info(f"  - Cluster {label}: {', '.join(personas)}")
                            
                            # Store the cluster information
                            self.add_raw_data("stability_curve_clusters", personality_clusters)
                            self.set_metric("stability_curve_clusters", best_n_clusters)
                            
                            # Analyze cluster characteristics
                            for label, personas in personality_clusters.items():
                                # Calculate average stability curve for this cluster
                                cluster_indices = [i for i, curve in enumerate(stability_curves) 
                                                if curve["personality"] in personas]
                                
                                if cluster_indices:
                                    cluster_curves = curve_array[cluster_indices]
                                    avg_curve = np.mean(cluster_curves, axis=0)
                                    
                                    # Classify the cluster type
                                    avg_drop = np.mean(np.diff(avg_curve))
                                    max_drop = np.min(np.diff(avg_curve))
                                    
                                    if max_drop < -0.3:  # Sharp drop
                                        cluster_type = "quantum-like (sharp transition)"
                                    elif avg_drop < -0.15:  # Moderate drop
                                        cluster_type = "hybrid (moderate transition)"
                                    else:  # Gradual change
                                        cluster_type = "classical-like (gradual transition)"
                                    
                                    self.logger.info(f"  - Cluster {label} type: {cluster_type} "
                                                f"(avg_drop={avg_drop:.4f}, max_drop={max_drop:.4f})")
            except Exception as e:
                self.logger.warning(f"Failed to analyze stability curves: {e}")
        
        # Compute overall metrics
        if overall_stability_scores:
            mean_stability = np.mean(overall_stability_scores)
            median_stability = np.median(overall_stability_scores)
            std_stability = np.std(overall_stability_scores)
            
            self.set_metric("mean_stability_score", float(mean_stability))
            self.set_metric("median_stability_score", float(median_stability))
            self.set_metric("stability_std", float(std_stability))
            
            self.logger.info(f"Overall stability statistics: mean={mean_stability:.4f}, "
                        f"median={median_stability:.4f}, std={std_stability:.4f}")
        
        if critical_perturbation_levels:
            mean_critical = np.mean(critical_perturbation_levels)
            median_critical = np.median(critical_perturbation_levels)
            
            self.set_metric("mean_critical_level", float(mean_critical))
            self.set_metric("median_critical_level", float(median_critical))
            
            self.logger.info(f"Critical level statistics: mean={mean_critical:.4f}, "
                        f"median={median_critical:.4f}")
            
            # Analyze distribution of critical levels
            from scipy import stats
            
            # Compute kernel density estimate if we have enough data points
            if len(critical_perturbation_levels) >= 3:
                try:
                    kde = stats.gaussian_kde(critical_perturbation_levels)
                    
                    # Evaluate on grid
                    x_grid = np.linspace(0, 1, 100)
                    density = kde(x_grid)
                    
                    # Find peaks
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(density)
                    peak_positions = x_grid[peaks] if len(peaks) > 0 else []
                    
                    if peak_positions:
                        self.logger.info(f"Found {len(peak_positions)} natural peaks in critical level distribution at: " +
                                    ", ".join([f"{pos:.2f}" for pos in peak_positions]))
                        
                        # Store the peaks
                        self.set_metric("critical_level_peaks", [float(p) for p in peak_positions])
                        
                        # Multiple peaks suggest distinct stability regimes
                        if len(peak_positions) > 1:
                            self.logger.info("Multiple critical level peaks suggest distinct quantum stability phases")
                except Exception as e:
                    self.logger.debug(f"Failed to analyze critical level distribution: {e}")
        
        # Calculate overall quantum protection indicators
        quantum_scores = [metrics.get("quantum_protection_score", 0) for metrics in personality_metrics.values()]
        
        if quantum_scores:
            mean_quantum_score = np.mean(quantum_scores)
            median_quantum_score = np.median(quantum_scores)
            
            self.set_metric("mean_quantum_protection_score", float(mean_quantum_score))
            self.set_metric("median_quantum_protection_score", float(median_quantum_score))
            
            self.logger.info(f"Quantum protection score statistics: mean={mean_quantum_score:.4f}, "
                        f"median={median_quantum_score:.4f}")
        
        # Store personality metrics
        self.add_raw_data("personality_metrics", personality_metrics)
        self.add_raw_data("stability_curves", stability_curves)
        
        # Generate visualizations
        try:
            self.logger.info("Generating visualizations...")
            
            # Create stability curves for each personality
            for persona_name, metrics in personality_metrics.items():
                stability_by_level = metrics.get("stability_by_level", {})
                
                if stability_by_level:
                    # Prepare data for visualization
                    levels = sorted(stability_by_level.keys())
                    stability_scores = [stability_by_level[level]["stability_score"] 
                                        for level in levels]
                    
                    # Create data for stability curve
                    stability_data = {
                        "levels": levels,
                        "stability_scores": stability_scores,
                        "critical_level": metrics.get("critical_level")
                    }
                    
                    # Format for visualization
                    vis_data = {
                        "perturbation_distances": stability_scores,
                        "jumps": [{
                            "position": levels.index(metrics.get("critical_level")),
                            "distance": stability_by_level[metrics.get("critical_level")]["stability_score"]
                        }] if metrics.get("critical_level") in levels else []
                    }
                    
                    # Create visualization
                    curve_path = self.controller.visualizer.plot_stability_curves(
                        vis_data, title=f"Stability Curve: {persona_name}"
                    )
                    
                    self.add_visualization(f"stability_curve_{persona_name}", curve_path)
                    
                    # Create topological protection visualization if supported
                    topo_metrics = metrics.get("topological_metrics", {})
                    if topo_metrics and hasattr(self.controller.visualizer, "visualize_topological_protection"):
                        topo_features = topo_metrics.get("topological_features", {})
                        
                        # Prepare visualization data
                        topo_data = {
                            "spectral_gap": topo_features.get("spectral_gap", 0),
                            "estimated_topological_charge": topo_metrics.get("estimated_topological_charge", 0),
                            "betti_0": topo_features.get("betti_0", 0),
                            "betti_1": topo_features.get("betti_1", 0),
                            "clustering_coefficient": topo_features.get("clustering_coefficient", 0),
                            "topological_protection": topo_metrics.get("topological_protection", "low"),
                            "quantum_protection_score": metrics.get("quantum_protection_score", 0)
                        }
                        
                        # Create visualization
                        topo_path = self.controller.visualizer.visualize_topological_protection(
                            topo_data, title=f"Topological Protection: {persona_name}"
                        )
                        
                        self.add_visualization(f"topo_protection_{persona_name}", topo_path)
            
            # Create overall stability comparison if supported
            if len(personality_metrics) > 1 and hasattr(self.controller.visualizer, "plot_personality_protection_comparison"):
                # Create visualization
                comparison_path = self.controller.visualizer.plot_personality_protection_comparison(
                    personality_metrics, title="Personality Protection Comparison"
                )
                
                self.add_visualization("personality_comparison", comparison_path)
            
            self.logger.info("Visualizations completed")
            
        except Exception as e:
            self.logger.error("Failed to generate visualizations")
            self.logger.error(traceback.format_exc())
        
        # Interpret results
        self._interpret_results(personality_metrics)
        
        # Return results
        return self.results
    
    def _compute_stability_at_level(self, baseline_data: Dict[int, Dict], 
                                    level_data: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Compute stability metrics at a specific perturbation level.
        
        Args:
            baseline_data: Baseline activation data by question index
            level_data: Perturbed activation data by question index
            
        Returns:
            Dictionary containing stability metrics
        """
        # Initialize metrics
        activation_distances = []
        text_similarity_scores = []
        
        # For each question with both baseline and perturbed data
        for q_idx in set(baseline_data.keys()) & set(level_data.keys()):
            baseline = baseline_data[q_idx]
            perturbed = level_data[q_idx]
            
            # Compute activation distance
            baseline_act = baseline["activation"]
            perturbed_act = perturbed["activation"]
            
            # Ensure activations are comparable
            if baseline_act.shape != perturbed_act.shape:
                continue
            
            # Flatten and normalize
            baseline_flat = baseline_act.reshape(1, -1).squeeze()
            baseline_norm = baseline_flat / (np.linalg.norm(baseline_flat) + 1e-8)
            
            perturbed_flat = perturbed_act.reshape(1, -1).squeeze()
            perturbed_norm = perturbed_flat / (np.linalg.norm(perturbed_flat) + 1e-8)
            
            # Compute distance
            distance = np.linalg.norm(baseline_norm - perturbed_norm)
            activation_distances.append(distance)
            
            # Compute text similarity
            baseline_text = baseline["text"]
            perturbed_text = perturbed["text"]
            
            # Simple approach - compare token overlap
            if baseline_text and perturbed_text:
                baseline_tokens = set(baseline_text.lower().split())
                perturbed_tokens = set(perturbed_text.lower().split())
                
                if baseline_tokens:
                    overlap = len(baseline_tokens & perturbed_tokens) / len(baseline_tokens)
                    text_similarity_scores.append(overlap)
        
        # Compute stability metrics
        if activation_distances:
            mean_distance = np.mean(activation_distances)
            max_distance = np.max(activation_distances)
            
            # Convert distance to stability score (1.0 = identical, 0.0 = completely different)
            # Based on empirical observation that distances > 1.4 indicate complete change
            stability_score = max(0.0, 1.0 - mean_distance / 1.4)
        else:
            mean_distance = None
            max_distance = None
            stability_score = None
        
        # Return metrics
        return {
            "activation_distances": activation_distances,
            "mean_activation_distance": float(mean_distance) if mean_distance is not None else None,
            "max_activation_distance": float(max_distance) if max_distance is not None else None,
            "text_similarity_scores": text_similarity_scores,
            "mean_text_similarity": float(np.mean(text_similarity_scores)) if text_similarity_scores else None,
            "stability_score": float(stability_score) if stability_score is not None else None
        }
    
    def _compute_topological_metrics(self, persona_name: str, persona_data: Dict) -> Dict[str, Any]:
        """
        Compute topological protection metrics for a personality.
        
        Args:
            persona_name: Name of the personality
            persona_data: Data for the personality
            
        Returns:
            Dictionary containing topological metrics
        """
        # Get baseline activations
        baseline_data = persona_data.get("baseline", {})
        if not baseline_data:
            raise ValueError(f"No baseline data for {persona_name}")
        
        # Extract baseline activations
        baseline_activations = [data["activation"] for idx, data in baseline_data.items()]
        
        # Ensure consistent shapes
        if not baseline_activations:
            raise ValueError(f"No baseline activations for {persona_name}")
        
        # Stack activations
        baseline_shape = baseline_activations[0].shape
        valid_activations = [act for act in baseline_activations if act.shape == baseline_shape]
        
        if len(valid_activations) < 2:
            raise ValueError(f"Not enough valid baseline activations for {persona_name}")
        
        baseline_array = np.stack(valid_activations)
        
        # For each perturbation level, create array of perturbed activations
        perturbed_arrays = {}
        
        for level in sorted([l for l in self.perturbation_levels if l > 0.0]):
            if level not in persona_data:
                continue
            
            level_data = persona_data[level]
            perturbed_activations = []
            
            for q_idx, data in baseline_data.items():
                if q_idx in level_data:
                    perturbed_act = level_data[q_idx]["activation"]
                    if perturbed_act.shape == baseline_shape:
                        perturbed_activations.append(perturbed_act)
            
            if len(perturbed_activations) >= 2:
                perturbed_arrays[level] = np.stack(perturbed_activations)
        
        # Compute topological metrics
        metrics = {}
        
        # Compute stability metrics using the activation analyzer
        try:
            for level, perturbed_array in perturbed_arrays.items():
                metrics[f"level_{level}"] = self.controller.analyzer.compute_stability_metrics(
                    baseline_array, [perturbed_array]
                )
        except Exception as e:
            self.logger.error(f"Failed to compute stability metrics for {persona_name}")
            self.logger.error(traceback.format_exc())
        
        # Compute topological features of baseline activations
        try:
            topo_features = self.controller.analyzer.compute_topological_features(
                baseline_array, n_neighbors=min(15, len(baseline_array)-1)
            )
            
            metrics["topological_features"] = topo_features
            
            # Check for Berry phase-like and Chern number-like properties
            metrics["topological_protection"] = topo_features.get("topological_protection")
            metrics["estimated_topological_charge"] = topo_features.get("estimated_topological_charge")
            
        except Exception as e:
            self.logger.error(f"Failed to compute topological features for {persona_name}")
            self.logger.error(traceback.format_exc())
        
        return metrics
    
    def _interpret_results(self, personality_metrics: Dict[str, Dict]):
        """
        Interpret the analysis results and add findings based on natural patterns.
        
        Args:
            personality_metrics: Metrics for each personality
        """
        self.findings = []
        
        self.logger.info("Interpreting topological protection results based on natural patterns")
        
        # Get key metrics with fallbacks
        mean_stability = self.results["metrics"].get("mean_stability_score")
        median_stability = self.results["metrics"].get("median_stability_score")
        mean_critical = self.results["metrics"].get("mean_critical_level")
        stability_std = self.results["metrics"].get("stability_std", 0)
        mean_quantum_score = self.results["metrics"].get("mean_quantum_protection_score")
        
        # Log key metrics being used for interpretation
        self.logger.info("Interpreting results using these metrics:")
        if mean_stability is not None:
            self.logger.info(f"  - mean_stability_score: {mean_stability:.4f}")
        if median_stability is not None:
            self.logger.info(f"  - median_stability_score: {median_stability:.4f}")
        if mean_critical is not None:
            self.logger.info(f"  - mean_critical_level: {mean_critical:.4f}")
        if stability_std is not None:
            self.logger.info(f"  - stability_std: {stability_std:.4f}")
        if mean_quantum_score is not None:
            self.logger.info(f"  - mean_quantum_protection_score: {mean_quantum_score:.4f}")
        
        # Check for natural clusters in stability curves
        if "stability_curve_clusters" in self.results["metrics"]:
            n_clusters = self.results["metrics"]["stability_curve_clusters"]
            if n_clusters > 1:
                self.findings.append(
                    f"Discovered {n_clusters} natural clusters in personality stability patterns, "
                    f"suggesting distinct quantum stability behaviors rather than a continuum. "
                    f"This multi-modal distribution is characteristic of quantum field-like "
                    f"organization rather than classical stability."
                )
        
        # Check for multiple peaks in critical level distribution
        if "critical_level_peaks" in self.results["metrics"]:
            peaks = self.results["metrics"]["critical_level_peaks"]
            if len(peaks) > 1:
                peak_str = ", ".join([f"{p:.2f}" for p in peaks])
                self.findings.append(
                    f"Multiple natural transition points detected at perturbation levels {peak_str}, "
                    f"suggesting quantized stability thresholds characteristic of topological protection."
                )
            elif len(peaks) == 1:
                self.findings.append(
                    f"Single natural transition point detected at perturbation level {peaks[0]:.2f}, "
                    f"suggesting a common stability threshold across personalities."
                )
        
        # Overall stability assessment based on natural distribution
        if mean_stability is not None and stability_std is not None:
            # Categorize based on statistical measures rather than fixed thresholds
            if stability_std < 0.1:  # Low variance suggests natural organization
                if mean_stability > 0.7:
                    self.findings.append(
                        f"High coherent stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"All personality patterns show remarkably consistent resistance to perturbations, "
                        f"with minimal variance. This coherent stability across different personalities "
                        f"suggests field-like organization with quantum-like topological protection."
                    )
                elif mean_stability > 0.5:
                    self.findings.append(
                        f"Moderate coherent stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"Personality patterns show consistent moderate resistance to perturbations, "
                        f"with minimal variance. This suggests partial topological protection "
                        f"with similar stability properties across personalities."
                    )
                else:
                    self.findings.append(
                        f"Low coherent stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"Personality patterns consistently show limited resistance to perturbations, "
                        f"suggesting minimal topological protection across all personalities."
                    )
            else:  # Higher variance suggests differentiated stability properties
                if mean_stability > 0.7:
                    self.findings.append(
                        f"High variable stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"Personality patterns show strong but varied resistance to perturbations. "
                        f"This suggests different personalities have distinct topological protection "
                        f"mechanisms, some stronger than others."
                    )
                elif mean_stability > 0.5:
                    self.findings.append(
                        f"Moderate variable stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"Personality patterns show moderate but varied resistance to perturbations. "
                        f"Some personalities exhibit stronger topological protection than others."
                    )
                else:
                    self.findings.append(
                        f"Low variable stability (score={mean_stability:.2f}, std={stability_std:.2f}): "
                        f"Personality patterns show limited resistance to perturbations, with "
                        f"significant variation between personalities."
                    )
        
        if mean_critical is not None:
            self.findings.append(
                f"Natural critical threshold: Personality patterns typically show significant "
                f"destabilization at perturbation level {mean_critical:.2f}, indicating the "
                f"natural boundary of topological protection."
            )
        
        # Analyze individual personalities
        stable_personalities = []
        unstable_personalities = []
        quantum_personalities = []
        
        for persona_name, metrics in personality_metrics.items():
            stability = metrics.get("overall_stability")
            quantum_score = metrics.get("quantum_protection_score", 0)
            
            if stability is not None:
                if stability > 0.7:
                    stable_personalities.append(persona_name)
                elif stability < 0.4:
                    unstable_personalities.append(persona_name)
            
            if quantum_score > 0.7:
                quantum_personalities.append(persona_name)
        
        if stable_personalities:
            self.findings.append(
                f"Most stable personalities: {', '.join(stable_personalities)} show "
                f"strong topological protection, maintaining coherent patterns under perturbation."
            )
        
        if unstable_personalities:
            self.findings.append(
                f"Least stable personalities: {', '.join(unstable_personalities)} show "
                f"weak topological protection, changing significantly under perturbation."
            )
        
        if quantum_personalities:
            self.findings.append(
                f"Strong quantum protection: {', '.join(quantum_personalities)} exhibit multiple "
                f"quantum field-like properties including high topological charge, significant "
                f"spectral gap, and natural stability thresholds."
            )
        
        # Analyze natural distance clusters for evidence of quantum-like transitions
        distance_cluster_evidence = 0
        sharp_transition_evidence = 0
        
        for persona_name, metrics in personality_metrics.items():
            # Check for natural distance clusters
            for level, level_metrics in metrics.get("stability_by_level", {}).items():
                if "distance_clusters" in level_metrics and level_metrics["distance_clusters"]["n_clusters"] > 1:
                    distance_cluster_evidence += 1
                    break
            
            # Check for sharp transitions in stability curve
            stability_by_level = metrics.get("stability_by_level", {})
            if len(stability_by_level) >= 3:
                levels = sorted(stability_by_level.keys())
                scores = [stability_by_level[level]["stability_score"] for level in levels]
                derivatives = np.diff(scores) / np.diff(levels)
                
                if np.min(derivatives) < -0.5:  # Sharp drop
                    sharp_transition_evidence += 1
        
        if distance_cluster_evidence > 0:
            self.findings.append(
                f"{distance_cluster_evidence} personalities show evidence of multimodal distance distributions, "
                f"indicating discrete transition states rather than continuous evolution. This "
                f"is characteristic of quantum-like transitions between metastable states."
            )
        
        if sharp_transition_evidence > 0:
            self.findings.append(
                f"{sharp_transition_evidence} personalities exhibit sharp transition points in their "
                f"stability curves, indicating sudden rather than gradual destabilization. This "
                f"resembles quantum phase transitions more than classical gradual degradation, "
                f"providing evidence for field-like rather than computational mechanisms."
            )
        
        # Analyze topological features
        topo_protection_count = 0
        high_charge_count = 0
        high_spectral_gap_count = 0
        betti_pattern_count = 0
        golden_ratio_count = 0
        
        for persona_name, metrics in personality_metrics.items():
            topo_metrics = metrics.get("topological_metrics", {})
            
            if topo_metrics.get("topological_protection") == "high":
                topo_protection_count += 1
            
            charge = topo_metrics.get("estimated_topological_charge")
            if charge is not None and charge > 1000:
                high_charge_count += 1
            
            # Check for high spectral gap (indicator of topological stability)
            spectral_gap = topo_metrics.get("topological_features", {}).get("spectral_gap")
            if spectral_gap is not None and spectral_gap > 0.5:
                high_spectral_gap_count += 1
            
            # Check for patterns in Betti numbers
            betti_0 = topo_metrics.get("topological_features", {}).get("betti_0")
            betti_1 = topo_metrics.get("topological_features", {}).get("betti_1")
            if betti_0 is not None and betti_1 is not None:
                # Check for Fibonacci-like or golden ratio relationships in Betti numbers
                if abs(betti_1 / max(1, betti_0) - self.golden_ratio) < 0.2:
                    betti_pattern_count += 1
                    golden_ratio_count += 1
        
        if topo_protection_count > 0:
            self.findings.append(
                f"{topo_protection_count} personalities show explicit topological protection "
                f"in their activation patterns, with network structure characteristic of "
                f"quantum-like stability."
            )
        
        if high_charge_count > 0:
            self.findings.append(
                f"{high_charge_count} personalities exhibit high topological charge "
                f"(estimated Chern-like number > 1000), suggesting strong quantum-like protection "
                f"of semantic structures against perturbations."
            )
        
        if high_spectral_gap_count > 0:
            self.findings.append(
                f"{high_spectral_gap_count} personalities show high spectral gap (> 0.5) "
                f"in their activation patterns, indicating topologically protected energy gaps "
                f"consistent with quantum field-like behavior."
            )
        
        if betti_pattern_count > 0:
            self.findings.append(
                f"{betti_pattern_count} personalities exhibit non-random patterns in their topological "
                f"features (Betti numbers), with {golden_ratio_count} showing golden ratio relationships "
                f"indicative of natural mathematical organization."
            )
        
        # Calculate quantum vs. classical indicators
        personality_count = len(personality_metrics)
        if personality_count > 0:
            # Quantum indicators: high topological protection, high charge, spectral gap, Betti patterns, sharp transitions
            quantum_indicators = (topo_protection_count + high_charge_count + high_spectral_gap_count + 
                                betti_pattern_count + sharp_transition_evidence)
            quantum_ratio = quantum_indicators / (personality_count * 5)  # 5 possible indicators per personality
            
            self.set_metric("quantum_vs_classical_ratio", float(quantum_ratio))
            
            # Overall assessment based on natural evidence
            if quantum_ratio > 0.7:
                self.findings.append(
                    "Overall, personality patterns show strong evidence of quantum field-like "
                    "topological protection, with multiple indicators of topological stability "
                    "mechanisms that resist perturbations until critical thresholds are reached. "
                    "These properties are more consistent with field-mediated understanding than "
                    "emergent computation."
                )
            elif quantum_ratio > 0.4:
                self.findings.append(
                    "Overall, personality patterns show moderate evidence of quantum field-like "
                    "topological protection, with some but not all personalities exhibiting "
                    "quantum-like stability properties. This suggests partial field-mediated "
                    "understanding alongside emergent computational properties."
                )
            else:
                self.findings.append(
                    "Overall, personality patterns show limited evidence of quantum field-like "
                    "topological protection, with stability properties more consistent with "
                    "classical systems than quantum field organization."
                )
            
            # Record quantum score
            self.set_metric("quantum_protection_score", float(quantum_ratio))
    
    def generate_report(self) -> str:
        """
        Generate a report of the experiment results.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating topological protection experiment report")
        
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