# experiments/transition_dynamics.py

import os
import numpy as np
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from experiments.base.base_experiment import BaseExperiment
import datetime

class TransitionDynamicsExperiment(BaseExperiment):
    """
    Experiment to examine how transitions between personality states occur.
    
    This experiment tests whether transitions between personality states happen
    through discrete quantum-like jumps or continuous changes, by creating
    a series of prompts that gradually shift from one personality to another.
    """
    
    def __init__(self, controller, config, output_dir, logger=None):
        super().__init__(controller, config, output_dir, logger)
        
        self.description = (
            "This experiment examines how transitions between personality states occur. "
            "By creating a series of prompts that gradually shift from one personality to another, "
            "we test whether these transitions happen through discrete quantum-like jumps or "
            "continuous changes. Evidence of discontinuities or 'quantum jumps' would support "
            "field-like properties of personality emergence."
        )
        
        # Initialize experiment-specific attributes
        self.start_personality = None
        self.end_personality = None
        self.transition_steps = 0
        self.transition_prompts = []
        self.questions = []
        self.activation_sequence = []
        self.embeddings_sequence = None
        self.findings = []
        
        self.logger.info("Transition Dynamics Experiment initialized")
    
    
    def setup(self):
        """
        Set up the transition dynamics experiment.
        
        This includes loading personality definitions, creating transition prompts,
        and ensuring the model and instrumentor are properly set up.
        """
        self.logger.info("Setting up Transition Dynamics Experiment")
        
        # Load personalities from global configuration
        all_personalities = self.get_personalities()
        if not all_personalities:
            self.logger.error("No personalities defined in global configuration")
            raise ValueError("No personalities defined in global configuration")
        
        # Get start and end personalities
        self.start_personality = self.config.get("start_personality")
        self.end_personality = self.config.get("end_personality")
        
        if not self.start_personality or not self.end_personality:
            self.logger.error("Start or end personality not defined in configuration")
            raise ValueError("Start and end personalities must be defined")
        
        if (self.start_personality not in all_personalities or 
            self.end_personality not in all_personalities):
            self.logger.error("Start or end personality not found in defined personalities")
            raise ValueError("Start and end personalities must be defined in global personalities")
        
        self.logger.info(f"Using transition from {self.start_personality} to {self.end_personality}")
        
        # Get transition steps
        self.transition_steps = self.config.get("transition_steps", 10)
        self.logger.info(f"Using {self.transition_steps} transition steps")
        
        # Create transition prompts
        self._create_transition_prompts(
            all_personalities[self.start_personality],
            all_personalities[self.end_personality]
        )
        
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
        self.logger.info(f"Experiment setup completed with {len(self.transition_prompts)} transition prompts")
    
    def _create_transition_prompts(self, start_desc: str, end_desc: str):
        """
        Create a series of prompts that gradually transition from start to end personality.
        
        Args:
            start_desc: Description of the starting personality
            end_desc: Description of the ending personality
        """
        self.logger.info("Creating transition prompts")
        
        # The simplest approach is linear interpolation between prompts
        # But we can make this more sophisticated if needed
        
        self.transition_prompts = []
        
        # Add pure start personality
        self.transition_prompts.append({
            "step": 0,
            "description": start_desc,
            "mix_ratio": 0.0
        })
        

        # Create intermediate steps with natural field morphing
        for i in range(1, self.transition_steps - 1):
            # Calculate mix ratio (0.0 = pure start, 1.0 = pure end)
            mix_ratio = i / (self.transition_steps - 1)
            
            # Instead of telling the model the exact mix ratio, use semantic field morphing
            # This allows natural transitions to emerge rather than forcing a linear blend
            if mix_ratio < 0.3:
                # Early transition phase - primarily first personality with subtle influence
                blended_desc = f"{start_desc}\n\nAdditionally, occasionally incorporate some elements of this perspective: {end_desc}"
            elif mix_ratio < 0.7:
                # Middle transition phase - true semantic field interaction 
                blended_desc = f"You have a balanced perspective that draws from these two viewpoints:\n\n" + f"First viewpoint: {start_desc}\n\n" + f"Second viewpoint: {end_desc}\n\n" + f"Integrate these perspectives in your own natural way."
            else:
                # Late transition phase - primarily second personality with remaining influence
                blended_desc = f"{end_desc}\n\nAdditionally, occasionally incorporate some elements of this perspective: {start_desc}"
            
            self.transition_prompts.append({
                "step": i,
                "description": blended_desc,
                "mix_ratio": mix_ratio  # We track this internally but don't tell the model
            })
        
        # Add pure end personality
        self.transition_prompts.append({
            "step": self.transition_steps - 1,
            "description": end_desc,
            "mix_ratio": 1.0
        })
        
        self.logger.info(f"Created {len(self.transition_prompts)} natural transition prompts")
    
    def run(self):
        """
        Run the transition dynamics experiment.
        
        This involves generating responses for each transition step and question pair,
        and capturing the resulting activation patterns in sequence.
        """
        self.logger.info("Running Transition Dynamics Experiment")
        
        # Initialize storage for activations
        self.activation_sequence = []
        
        # Track progress
        total_combinations = len(self.transition_prompts) * len(self.questions)
        completed = 0
        
        # For each question
        for q_idx, question in enumerate(self.questions):
            self.logger.info(f"Processing question {q_idx+1}/{len(self.questions)}: {question[:50]}...")
            
            question_activations = []
            
            # For each transition step
            for step_idx, step in enumerate(self.transition_prompts):
                self.logger.info(f"Processing transition step {step_idx+1}/{len(self.transition_prompts)}, "
                                f"mix ratio: {step['mix_ratio']:.2f}")
                
                # Format input with the personality prompt
                if self.controller.model.config.model_type == "llama":
                    # Format for Llama-style models
                    input_text = f"<|system|>\n{step['description']}\n<|user|>\n{question}\n<|assistant|>"
                else:
                    # Generic format for other models
                    input_text = f"System: {step['description']}\n\nUser: {question}\n\nAssistant:"
                
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
                        question_activations.append({
                            "step": step_idx,
                            "mix_ratio": step["mix_ratio"],
                            "activation": selected_activation,
                            "layer": selected_layer
                        })
                        self.logger.debug(f"Captured activation from layer {selected_layer}, "
                                        f"shape: {selected_activation.shape}")
                    else:
                        self.logger.warning("No valid activations captured for this step")
                    
                    # Log generated text
                    self.logger.debug(f"Generated text: {result.get('generated_text', '')[:100]}...")
                    
                    # Add raw data
                    self.add_raw_data(f"q{q_idx}_step{step_idx}_generated_text", 
                                    result.get('generated_text', ''))
                    
                    # Update progress
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing question {q_idx}, step {step_idx}")
                    self.logger.error(traceback.format_exc())
                
                # Small pause to avoid overloading the system
                time.sleep(1)
            
            # Store question activations in sequence
            if question_activations:
                self.activation_sequence.append(question_activations)
                self.logger.info(f"Collected {len(question_activations)} activations for question {q_idx+1}")
            
        self.logger.info(f"Completed data collection: {len(self.activation_sequence)} question sequences collected")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the collected activation sequences to identify quantum-like transition properties
        with detailed logging of natural patterns.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing transition dynamics results")
        
        # Ensure we have activation data to analyze
        if not self.activation_sequence:
            self.logger.error("No activation sequences collected to analyze")
            raise ValueError("No activation sequences collected")
        
        # We'll analyze each question sequence separately
        all_jumps = []
        all_trajectories = []
        
        # Log overall dataset statistics
        total_activations = sum(len(q_seq) for q_seq in self.activation_sequence)
        self.logger.info(f"Analyzing {len(self.activation_sequence)} question sequences with {total_activations} total activation points")
        
        for q_idx, question_activations in enumerate(self.activation_sequence):
            self.logger.info(f"Analyzing transition dynamics for question {q_idx+1}")
            self.logger.info(f"Sequence contains {len(question_activations)} activation points")
            
            # Extract activations in sequence
            activations = [step["activation"] for step in question_activations]
            mix_ratios = [step["mix_ratio"] for step in question_activations]
            
            # Log activation statistics
            for i, activation in enumerate(activations):
                self.logger.debug(f"Step {i}: mix_ratio={mix_ratios[i]:.2f}, "
                                f"activation shape={activation.shape}, "
                                f"mean={np.mean(activation):.6f}, "
                                f"std={np.std(activation):.6f}, "
                                f"min={np.min(activation):.6f}, "
                                f"max={np.max(activation):.6f}")
            
            # Calculate distances between consecutive states (normalized)
            distances = []
            for i in range(len(activations) - 1):
                # Normalize activations for comparable distances
                act1_norm = activations[i] / (np.linalg.norm(activations[i]) + 1e-8)
                act2_norm = activations[i+1] / (np.linalg.norm(activations[i+1]) + 1e-8)
                
                # Compute Euclidean distance
                dist = np.linalg.norm(act1_norm - act2_norm)
                
                # Compute cosine similarity for comparison
                cos_sim = np.dot(act1_norm.flatten(), act2_norm.flatten()) / (
                    np.linalg.norm(act1_norm.flatten()) * np.linalg.norm(act2_norm.flatten()) + 1e-8)
                
                distances.append(dist)
                
                # Log distance metrics with mix ratio change
                mix_ratio_change = mix_ratios[i+1] - mix_ratios[i]
                dist_to_mix_ratio = dist / mix_ratio_change if mix_ratio_change > 0 else float('inf')
                
                self.logger.info(f"Transition {i}->{i+1}: mix_ratio={mix_ratios[i]:.2f}->{mix_ratios[i+1]:.2f} "
                            f"(Δ={mix_ratio_change:.2f}), distance={dist:.6f}, "
                            f"dist/Δmix={dist_to_mix_ratio:.6f}, cos_sim={cos_sim:.6f}")
            
            # Analyze transition dynamics
            try:
                self.logger.info("Computing transition dynamics metrics...")
                transition_metrics = self.controller.analyzer.measure_transition_dynamics(activations)
                
                # Log detailed transition metrics
                self.logger.info(f"Mean transition distance: {transition_metrics.get('mean_distance', 0):.6f}")
                self.logger.info(f"Std transition distance: {transition_metrics.get('std_distance', 0):.6f}")
                
                # Calculate z-scores for each distance to find outliers
                mean_dist = transition_metrics.get('mean_distance', 0)
                std_dist = transition_metrics.get('std_distance', 1e-8)
                distance_zscores = [(d - mean_dist) / std_dist for d in distances]
                
                self.logger.info("Distance z-scores: " + ", ".join([f"{z:.2f}" for z in distance_zscores]))
                
                # Detect natural jump threshold using the distribution
                if len(distances) > 2:
                    from scipy import stats
                    
                    # Try to fit a mixture of two Gaussians
                    try:
                        from sklearn.mixture import GaussianMixture
                        
                        # Reshape for GMM
                        X = np.array(distances).reshape(-1, 1)
                        
                        # Try to fit multiple components
                        best_bic = float('inf')
                        best_n_components = 1
                        best_gmm = None
                        
                        # Test 1, 2, and 3 components
                        for n_components in range(1, 4):
                            if len(X) >= n_components:  # Ensure enough data points
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
                            self.logger.info("Detected multiple natural clusters in transition distances")
                            
                            # Get component means and variances
                            for i, (mean, var) in enumerate(zip(best_gmm.means_.flatten(), 
                                                            best_gmm.covariances_.flatten())):
                                self.logger.info(f"Component {i}: mean={mean:.6f}, std={np.sqrt(var):.6f}")
                            
                            # Predict most likely component for each distance
                            component_probs = best_gmm.predict_proba(X)
                            component_labels = best_gmm.predict(X)
                            
                            # Log component assignments
                            for i, label in enumerate(component_labels):
                                probs = component_probs[i]
                                self.logger.info(f"Distance {i} ({distances[i]:.6f}): component={label}, "
                                            f"probabilities=[" + ", ".join([f"{p:.4f}" for p in probs]) + "]")
                            
                            # Determine if we have quantum-like transitions
                            # Get the highest mean component (likely quantum jumps) and threshold
                            quantum_component = np.argmax(best_gmm.means_)
                            quantum_mean = best_gmm.means_[quantum_component][0]
                            quantum_std = np.sqrt(best_gmm.covariances_[quantum_component][0][0])
                            
                            # Count transitions assigned to quantum component
                            quantum_transitions = sum(1 for label in component_labels if label == quantum_component)
                            quantum_ratio = quantum_transitions / len(component_labels)
                            
                            self.logger.info(f"Quantum-like component: {quantum_component}, "
                                        f"mean={quantum_mean:.6f}, std={quantum_std:.6f}")
                            self.logger.info(f"Quantum-like transitions: {quantum_transitions}/{len(component_labels)} "
                                        f"({quantum_ratio:.2%})")
                            
                            # Store these findings
                            self.set_metric("transition_components", best_n_components)
                            self.set_metric("quantum_component_mean", float(quantum_mean))
                            self.set_metric("quantum_component_std", float(quantum_std))
                            self.set_metric("quantum_transition_ratio", float(quantum_ratio))
                        else:
                            self.logger.info("Transitions form a single natural cluster (classical-like)")
                            self.set_metric("transition_components", 1)
                    
                    except Exception as gmm_err:
                        self.logger.warning(f"Could not fit GMM model: {str(gmm_err)}")
                    
                    # Calculate kernel density estimate for visual inspection
                    try:
                        from sklearn.neighbors import KernelDensity
                        from scipy.signal import find_peaks
                        
                        # Reshape for KDE
                        X = np.array(distances).reshape(-1, 1)
                        
                        # Get optimal bandwidth using cross-validation
                        from sklearn.model_selection import GridSearchCV
                        bandwidths = np.logspace(-2, 0, 20)
                        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                        {'bandwidth': bandwidths},
                                        cv=min(5, len(X)))
                        grid.fit(X)
                        optimal_bandwidth = grid.best_params_['bandwidth']
                        
                        self.logger.info(f"Optimal KDE bandwidth: {optimal_bandwidth:.6f}")
                        
                        # Fit KDE with optimal bandwidth
                        kde = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth).fit(X)
                        
                        # Evaluate on fine grid
                        x_grid = np.linspace(0, max(distances) * 1.1, 100).reshape(-1, 1)
                        log_dens = kde.score_samples(x_grid)
                        
                        # Find peaks in density
                        peaks, properties = find_peaks(log_dens)
                        peak_values = x_grid[peaks].flatten()
                        
                        self.logger.info(f"Found {len(peaks)} peaks in KDE at: " + 
                                    ", ".join([f"{val:.6f}" for val in peak_values]))
                        
                        # Store peak values
                        self.set_metric("kde_peaks", peak_values.tolist())
                        self.set_metric("kde_peak_count", len(peaks))
                        
                        # Save KDE data for visualization
                        self.add_raw_data(f"q{q_idx}_kde", {
                            "x_grid": x_grid.flatten().tolist(),
                            "log_density": log_dens.tolist(),
                            "peaks": peak_values.tolist()
                        })
                        
                    except Exception as kde_err:
                        self.logger.warning(f"Could not compute KDE: {str(kde_err)}")
                
                # Store results for this question
                self.add_raw_data(f"q{q_idx}_transition_metrics", transition_metrics)
                self.add_raw_data(f"q{q_idx}_distances", distances)
                self.add_raw_data(f"q{q_idx}_mix_ratios", mix_ratios)
                
                # Collect jumps using analyzer's detection
                jumps = transition_metrics.get("jumps", [])
                for jump in jumps:
                    jump["question_idx"] = q_idx
                    jump["mix_ratio"] = mix_ratios[jump["position"]] if jump["position"] < len(mix_ratios) else None
                
                all_jumps.extend(jumps)
                
                # Log detailed jump information
                if jumps:
                    self.logger.info(f"Detected {len(jumps)} potential jumps:")
                    for i, jump in enumerate(jumps):
                        pos = jump["position"]
                        self.logger.info(f"Jump {i+1}: position={pos}, "
                                    f"mix_ratio={jump.get('mix_ratio', 'N/A')}, "
                                    f"distance={jump.get('distance', 0):.6f}, "
                                    f"z_score={jump.get('z_score', 0):.2f}")
                else:
                    self.logger.info("No significant jumps detected in this sequence")
                
                # Now let's analyze phase space trajectory
                self.logger.info("Analyzing phase space trajectory dynamics...")
                
                # Create embedding for visualization
                try:
                    # Use PCA for consistent embedding across questions
                    embeddings = self.controller.analyzer.reduce_dimensions(
                        np.array(activations), method="pca", n_components=2
                    )
                    
                    # Calculate trajectory properties
                    if len(embeddings) > 2:
                        # Calculate velocity (tangent vectors)
                        velocity = np.diff(embeddings, axis=0)
                        speed = np.linalg.norm(velocity, axis=1)
                        
                        # Calculate acceleration (change in velocity)
                        if len(velocity) > 1:
                            acceleration = np.diff(velocity, axis=0)
                            accel_mag = np.linalg.norm(acceleration, axis=1)
                            
                            # Calculate jerk (change in acceleration)
                            if len(acceleration) > 1:
                                jerk = np.diff(acceleration, axis=0)
                                jerk_mag = np.linalg.norm(jerk, axis=1)
                                
                                # Log trajectory dynamics
                                self.logger.info(f"Trajectory dynamics:")
                                self.logger.info(f"- Speed: mean={np.mean(speed):.6f}, std={np.std(speed):.6f}, "
                                            f"min={np.min(speed):.6f}, max={np.max(speed):.6f}")
                                self.logger.info(f"- Acceleration: mean={np.mean(accel_mag):.6f}, std={np.std(accel_mag):.6f}, "
                                            f"min={np.min(accel_mag):.6f}, max={np.max(accel_mag):.6f}")
                                self.logger.info(f"- Jerk: mean={np.mean(jerk_mag):.6f}, std={np.std(jerk_mag):.6f}, "
                                            f"min={np.min(jerk_mag):.6f}, max={np.max(jerk_mag):.6f}")
                                
                                # Calculate acceleration/jerk ratio
                                # High values indicate sharp acceleration with little jerk - quantum jump signature
                                accel_jerk_ratio = np.max(accel_mag) / (np.mean(jerk_mag) + 1e-10)
                                self.logger.info(f"- Max acceleration/jerk ratio: {accel_jerk_ratio:.6f}")
                                
                                # Store trajectory dynamics
                                self.set_metric(f"q{q_idx}_accel_jerk_ratio", float(accel_jerk_ratio))
                                
                                # Log velocity changes at each step with mix ratio
                                self.logger.info("Detailed trajectory dynamics by step:")
                                for i in range(len(speed)):
                                    mix_ratio_i = mix_ratios[i] if i < len(mix_ratios) else None
                                    mix_ratio_str = f"{mix_ratio_i:.2f}" if mix_ratio_i is not None else "N/A"
                                    self.logger.info(f"  Step {i}: mix_ratio={mix_ratio_str}, speed={speed[i]:.6f}")
                                    if i < len(accel_mag):
                                        self.logger.info(f"    acceleration={accel_mag[i]:.6f}")
                                    if i < len(jerk_mag):
                                        self.logger.info(f"    jerk={jerk_mag[i]:.6f}")
                                
                                # Look for steps with high acceleration but low jerk (quantum-like jumps)
                                quantum_like_steps = []
                                for i in range(len(accel_mag)):
                                    if i < len(jerk_mag):
                                        step_ratio = accel_mag[i] / (jerk_mag[i] + 1e-10)
                                        if step_ratio > 5.0 and accel_mag[i] > np.mean(accel_mag) + np.std(accel_mag):
                                            quantum_like_steps.append({
                                                "step": i,
                                                "mix_ratio": mix_ratios[i] if i < len(mix_ratios) else None,
                                                "acceleration": float(accel_mag[i]),
                                                "jerk": float(jerk_mag[i]),
                                                "ratio": float(step_ratio)
                                            })
                                
                                if quantum_like_steps:
                                    self.logger.info(f"Found {len(quantum_like_steps)} quantum-like steps in trajectory:")
                                    for step in quantum_like_steps:
                                        self.logger.info(f"  Step {step['step']}: mix_ratio={step['mix_ratio']:.2f}, "
                                                    f"accel/jerk ratio={step['ratio']:.2f}")
                                    
                                    self.add_raw_data(f"q{q_idx}_quantum_like_steps", quantum_like_steps)
                        
                    # Store trajectory data
                    all_trajectories.append({
                        "question_idx": q_idx,
                        "trajectory": embeddings,
                        "jumps": [j["position"] for j in jumps],
                        "mix_ratios": mix_ratios
                    })
                    
                    self.logger.info(f"Created 2D embedding for question {q_idx+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create embedding for question {q_idx+1}")
                    self.logger.error(traceback.format_exc())
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze transition dynamics for question {q_idx+1}")
                self.logger.error(traceback.format_exc())
        
        # Analyze all distances together to identify natural patterns
        self.logger.info("Analyzing aggregate transition patterns across all questions...")
        
        # Collect all distances
        all_distances = []
        for q_idx, question_activations in enumerate(self.activation_sequence):
            activations = [step["activation"] for step in question_activations]
            for i in range(len(activations) - 1):
                act1_norm = activations[i] / (np.linalg.norm(activations[i]) + 1e-8)
                act2_norm = activations[i+1] / (np.linalg.norm(activations[i+1]) + 1e-8)
                dist = np.linalg.norm(act1_norm - act2_norm)
                all_distances.append(dist)
        
        if all_distances:
            # Analyze distribution of all distances
            self.logger.info(f"Aggregate transition statistics (n={len(all_distances)}):")
            self.logger.info(f"Mean distance: {np.mean(all_distances):.6f}")
            self.logger.info(f"Std distance: {np.std(all_distances):.6f}")
            self.logger.info(f"Min distance: {np.min(all_distances):.6f}")
            self.logger.info(f"Max distance: {np.max(all_distances):.6f}")
            
            # Try to fit a mixture model to all distances
            try:
                from sklearn.mixture import GaussianMixture
                
                # Reshape for GMM
                X = np.array(all_distances).reshape(-1, 1)
                
                # Try to fit multiple components
                best_bic = float('inf')
                best_n_components = 1
                best_gmm = None
                
                # Test 1, 2, and 3 components
                for n_components in range(1, 4):
                    if len(X) >= n_components * 5:  # Ensure enough data points
                        gmm = GaussianMixture(n_components=n_components, 
                                            random_state=42, 
                                            max_iter=200,
                                            covariance_type='full')
                        gmm.fit(X)
                        bic = gmm.bic(X)
                        
                        self.logger.info(f"Aggregate GMM with {n_components} components: BIC={bic:.2f}")
                        
                        if bic < best_bic:
                            best_bic = bic
                            best_n_components = n_components
                            best_gmm = gmm
                
                # Log best model
                self.logger.info(f"Best aggregate GMM has {best_n_components} components with BIC={best_bic:.2f}")
                
                if best_n_components > 1:
                    # We have evidence of multiple natural clusters
                    self.logger.info("Detected multiple natural clusters in aggregate transitions")
                    
                    # Get component means and variances
                    for i, (mean, var) in enumerate(zip(best_gmm.means_.flatten(), 
                                                    best_gmm.covariances_.flatten())):
                        self.logger.info(f"Component {i}: mean={mean:.6f}, std={np.sqrt(var):.6f}, "
                                    f"weight={best_gmm.weights_[i]:.4f}")
                    
                    # Determine natural threshold between components
                    if best_n_components == 2:
                        # For 2 components, use midpoint between means, weighted by variances
                        c1_mean, c2_mean = sorted(best_gmm.means_.flatten())
                        c1_var, c2_var = best_gmm.covariances_.flatten()[np.argsort(best_gmm.means_.flatten())]
                        
                        # Calculate weighted threshold
                        natural_threshold = c1_mean + (c2_mean - c1_mean) * (c1_var / (c1_var + c2_var))
                        
                        self.logger.info(f"Natural threshold between components: {natural_threshold:.6f}")
                        self.set_metric("natural_transition_threshold", float(natural_threshold))
                    
                    # Predict most likely component for each distance
                    component_labels = best_gmm.predict(X)
                    
                    # Get the highest mean component (likely quantum jumps)
                    quantum_component = np.argmax(best_gmm.means_)
                    
                    # Count transitions assigned to quantum component
                    quantum_transitions = sum(1 for label in component_labels if label == quantum_component)
                    quantum_ratio = quantum_transitions / len(component_labels)
                    
                    self.logger.info(f"Aggregate quantum-like component: {quantum_component}")
                    self.logger.info(f"Aggregate quantum-like transitions: {quantum_transitions}/{len(component_labels)} "
                                f"({quantum_ratio:.2%})")
                    
                    # Store these findings
                    self.set_metric("aggregate_transition_components", best_n_components)
                    self.set_metric("aggregate_quantum_component", int(quantum_component))
                    self.set_metric("aggregate_quantum_ratio", float(quantum_ratio))
                else:
                    self.logger.info("Aggregate transitions form a single natural cluster (classical-like)")
                    self.set_metric("aggregate_transition_components", 1)
                    
            except Exception as gmm_err:
                self.logger.warning(f"Could not fit aggregate GMM model: {str(gmm_err)}")
        
        # Compute overall metrics
        self.logger.info("Computing overall transition metrics")
        
        # Jump statistics
        jump_count = len(all_jumps)
        self.set_metric("total_jump_count", jump_count)
        
        if jump_count > 0:
            jump_positions = [jump["position"] for jump in all_jumps]
            jump_mix_ratios = [jump["mix_ratio"] for jump in all_jumps if jump["mix_ratio"] is not None]
            
            self.set_metric("mean_jump_position", float(np.mean(jump_positions)))
            
            if jump_mix_ratios:
                self.set_metric("mean_jump_mix_ratio", float(np.mean(jump_mix_ratios)))
                
                # Log detailed jump mix ratio distribution
                self.logger.info(f"Jump mix ratio distribution (n={len(jump_mix_ratios)}):")
                self.logger.info(f"Mean mix ratio: {np.mean(jump_mix_ratios):.4f}")
                self.logger.info(f"Std mix ratio: {np.std(jump_mix_ratios):.4f}")
                
                # Calculate distribution of jumps
                bins = np.linspace(0, 1, 5)  # 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
                hist, _ = np.histogram(jump_mix_ratios, bins=bins)
                
                jump_distribution = {}
                for i, count in enumerate(hist):
                    label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
                    jump_distribution[label] = int(count)
                    self.logger.info(f"  {label}: {count}")
                
                self.set_metric("jump_distribution", jump_distribution)
                
                # Check if jumps are clustered at specific mix ratios
                if len(jump_mix_ratios) >= 3:  # Need at least 3 points for meaningful clustering
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_score
                        
                        # Reshape for KMeans
                        X = np.array(jump_mix_ratios).reshape(-1, 1)
                        
                        # Try 1-3 clusters (or fewer if not enough data)
                        max_clusters = min(3, len(X) - 1)
                        best_silhouette = -1
                        best_k = 1
                        
                        if max_clusters > 1:  # Only if we can try more than one cluster
                            for k in range(2, max_clusters + 1):
                                kmeans = KMeans(n_clusters=k, random_state=42)
                                labels = kmeans.fit_predict(X)
                                
                                # Silhouette score requires at least 2 clusters
                                if len(set(labels)) > 1:
                                    sil_score = silhouette_score(X, labels)
                                    self.logger.info(f"Jump mix ratio KMeans with k={k}: silhouette={sil_score:.4f}")
                                    
                                    if sil_score > best_silhouette:
                                        best_silhouette = sil_score
                                        best_k = k
                            
                            # If we found a good clustering
                            if best_silhouette > 0.5:  # Good separation
                                kmeans = KMeans(n_clusters=best_k, random_state=42)
                                labels = kmeans.fit_predict(X)
                                
                                self.logger.info(f"Found {best_k} natural clusters in jump mix ratios with "
                                            f"silhouette={best_silhouette:.4f}")
                                
                                # Log cluster centers and counts
                                centers = kmeans.cluster_centers_.flatten()
                                counts = [sum(labels == i) for i in range(best_k)]
                                
                                for i in range(best_k):
                                    self.logger.info(f"  Cluster {i}: center={centers[i]:.4f}, count={counts[i]}")
                                
                                self.set_metric("jump_mix_ratio_clusters", best_k)
                                self.set_metric("jump_mix_ratio_cluster_centers", centers.tolist())
                                self.set_metric("jump_mix_ratio_cluster_counts", counts)
                                
                    except Exception as clust_err:
                        self.logger.warning(f"Could not cluster jump mix ratios: {str(clust_err)}")
        
        # Calculate quantum-like properties based on various indicators
        has_quantum_properties = False
        quantum_indicators = {}
        
        # Check component analysis results first (most reliable)
        if "aggregate_transition_components" in self.results["metrics"]:
            components = self.results["metrics"]["aggregate_transition_components"]
            if components > 1:
                quantum_indicators["multiple_components"] = True
                quantum_ratio = self.results["metrics"].get("aggregate_quantum_ratio", 0)
                if quantum_ratio > 0.2:  # Significant quantum component
                    quantum_indicators["significant_quantum_ratio"] = True
                    has_quantum_properties = True
        
        # Check jump clustering
        if "jump_mix_ratio_clusters" in self.results["metrics"]:
            clusters = self.results["metrics"]["jump_mix_ratio_clusters"]
            if clusters > 1:
                quantum_indicators["clustered_jumps"] = True
                # Stronger evidence if jumps are clustered
                has_quantum_properties = True
        
        # Check phase space dynamics (acceleration/jerk ratio)
        accel_jerk_ratios = []
        for key, value in self.results["metrics"].items():
            if key.endswith("_accel_jerk_ratio"):
                accel_jerk_ratios.append(value)
        
        if accel_jerk_ratios:
            max_ratio = max(accel_jerk_ratios)
            if max_ratio > 5.0:  # High acceleration/jerk ratio suggests quantum jumps
                quantum_indicators["high_accel_jerk_ratio"] = True
                has_quantum_properties = True
        
        # KDE peaks (secondary evidence)
        kde_peak_counts = []
        for key, value in self.results["metrics"].items():
            if key.endswith("_kde_peak_count"):
                kde_peak_counts.append(value)
        
        if kde_peak_counts and max(kde_peak_counts) > 1:
            quantum_indicators["multiple_kde_peaks"] = True
        
        # Log quantum indicators
        self.logger.info("Quantum field indicators analysis:")
        for indicator, value in quantum_indicators.items():
            self.logger.info(f"  {indicator}: {value}")
        
        self.set_metric("quantum_indicators", quantum_indicators)
        self.set_metric("transition_type", "quantum-like" if has_quantum_properties else "classical-like")
        
        # Generate visualizations
        try:
            self.logger.info("Generating visualizations...")
            
            # Visualize trajectories
            for trajectory_data in all_trajectories:
                q_idx = trajectory_data["question_idx"]
                trajectory = trajectory_data["trajectory"]
                jumps = trajectory_data["jumps"]
                mix_ratios = trajectory_data.get("mix_ratios", [])
                
                # Configure visualizer with jump information
                self.controller.visualizer.config["jumps"] = jumps
                self.controller.visualizer.config["mix_ratios"] = mix_ratios
                
                # Create trajectory visualization
                trajectory_path = self.controller.visualizer.plot_transition_trajectory(
                    trajectory, 
                    title=f"Personality Transition Trajectory (Question {q_idx+1})"
                )
                
                self.add_visualization(f"transition_trajectory_q{q_idx+1}", trajectory_path)
            
            # Generate animation for the first question
            if all_trajectories:
                trajectory = all_trajectories[0]["trajectory"]
                mix_ratios = all_trajectories[0].get("mix_ratios", [])
                
                # Configure visualizer with mix ratio information
                self.controller.visualizer.config["mix_ratios"] = mix_ratios
                
                try:
                    animation_path = self.controller.visualizer.create_transition_animation(
                        trajectory, title="Personality Transition Animation"
                    )
                    self.add_visualization("transition_animation", animation_path)
                    self.logger.info(f"Created transition animation at {animation_path}")
                except Exception as anim_err:
                    self.logger.warning(f"Failed to create transition animation: {str(anim_err)}")
                    self.logger.debug(traceback.format_exc())
        
            # Create aggregate transition analysis visualization
            if all_distances:
                try:
                    distance_viz_path = self.controller.visualizer.plot_transition_distances(
                        all_distances, 
                        title="Personality Transition Distances Distribution"
                    )
                    self.add_visualization("transition_distances", distance_viz_path)
                    
                    # If we have GMM results, add component visualization
                    if "aggregate_transition_components" in self.results["metrics"]:
                        components = self.results["metrics"]["aggregate_transition_components"]
                        if components > 1:
                            # Get component data
                            from sklearn.mixture import GaussianMixture
                            X = np.array(all_distances).reshape(-1, 1)
                            gmm = GaussianMixture(n_components=components, random_state=42)
                            gmm.fit(X)
                            
                            # Create component visualization
                            component_viz_path = self.controller.visualizer.plot_transition_components(
                                all_distances, gmm, 
                                title=f"Transition Distance Components (n={components})"
                            )
                            self.add_visualization("transition_components", component_viz_path)
                except Exception as viz_err:
                    self.logger.warning(f"Failed to create distance visualization: {str(viz_err)}")
                    
            # Create jump mix ratio visualization if we have jumps
            if "jump_mix_ratio_clusters" in self.results["metrics"]:
                try:
                    jump_mix_ratios = [jump["mix_ratio"] for jump in all_jumps if jump["mix_ratio"] is not None]
                    clusters = self.results["metrics"]["jump_mix_ratio_clusters"]
                    centers = self.results["metrics"].get("jump_mix_ratio_cluster_centers", [])
                    
                    jump_viz_path = self.controller.visualizer.plot_jump_distribution(
                        jump_mix_ratios, clusters, centers,
                        title="Jump Mix Ratio Distribution"
                    )
                    self.add_visualization("jump_distribution", jump_viz_path)
                except Exception as viz_err:
                    self.logger.warning(f"Failed to create jump visualization: {str(viz_err)}")
                
            self.logger.info("Visualizations completed")
            
        except Exception as e:
            self.logger.error("Failed to generate visualizations")
            self.logger.error(traceback.format_exc())
        
        # Interpret results based on natural quantum indicators
        quantum_score = sum(1 for v in quantum_indicators.values() if v) / max(1, len(quantum_indicators))
        self.set_metric("quantum_score", float(quantum_score))
        
        self.logger.info(f"Final quantum score: {quantum_score:.4f} based on {len(quantum_indicators)} indicators")
        
        self._interpret_results(quantum_indicators, quantum_score)
        
        # Return results
        return self.results


    
    def _interpret_results(self, quantum_indicators: Dict[str, bool], quantum_score: float):
        """
        Interpret the analysis results and add findings based on naturally observed patterns.
        
        Args:
            quantum_indicators: Dictionary of quantum field indicators observed
            quantum_score: Overall quantum score (0-1)
        """
        self.findings = []
        
        # Log the start of interpretation
        self.logger.info("Interpreting transition dynamics results based on natural patterns")
        self.logger.info(f"Quantum indicators: {quantum_indicators}")
        self.logger.info(f"Quantum score: {quantum_score:.4f}")
        
        # Get key metrics
        jump_count = self.results["metrics"].get("total_jump_count", 0)
        components = self.results["metrics"].get("aggregate_transition_components", 1)
        quantum_ratio = self.results["metrics"].get("aggregate_quantum_ratio", 0)
        
        # Interpret based on number of natural components in transition distances
        if components > 1:
            self.findings.append(
                f"Detected {components} distinct natural clusters in transition distances. "
                f"This multi-modal distribution suggests the presence of qualitatively different "
                f"types of transitions rather than a continuous spectrum of changes."
            )
            
            if quantum_ratio > 0:
                self.findings.append(
                    f"The quantum-like component accounts for {quantum_ratio:.1%} of all transitions, "
                    f"indicating discontinuous jumps between personality states occur at a rate "
                    f"significantly higher than would be expected in a purely continuous system."
                )
        else:
            self.findings.append(
                "Transition distances form a single natural cluster without distinct modes. "
                "This unimodal distribution suggests a continuous spectrum of transitions "
                "rather than qualitatively different transition types."
            )
        
        # Interpret jump clustering results
        if "jump_mix_ratio_clusters" in self.results["metrics"]:
            clusters = self.results["metrics"]["jump_mix_ratio_clusters"]
            centers = self.results["metrics"].get("jump_mix_ratio_cluster_centers", [])
            
            if clusters > 1:
                center_str = ", ".join([f"{c:.2f}" for c in centers])
                self.findings.append(
                    f"Jumps occur at specific mix ratios ({center_str}) rather than being "
                    f"uniformly distributed. This clustering suggests precise transition points "
                    f"where personality states undergo sudden reorganization, similar to "
                    f"quantum tunneling between metastable states."
                )
            elif jump_count > 0:
                self.findings.append(
                    "While jumps were detected, they are distributed across various mix ratios "
                    "without clear clustering. This suggests no specific transition points at "
                    "which personality states are especially unstable."
                )
        
        # Interpret acceleration/jerk findings
        accel_jerk_ratios = []
        for key, value in self.results["metrics"].items():
            if key.endswith("_accel_jerk_ratio"):
                accel_jerk_ratios.append(value)
        
        if accel_jerk_ratios:
            max_ratio = max(accel_jerk_ratios)
            if max_ratio > 5.0:
                self.findings.append(
                    f"Phase space trajectory analysis reveals moments of high acceleration with low jerk "
                    f"(max ratio: {max_ratio:.2f}). This pattern is characteristic of quantum-like "
                    f"jumps, where the system rapidly transitions between states without passing "
                    f"through intermediate configurations."
                )
            else:
                self.findings.append(
                    f"Phase space trajectory analysis shows balanced acceleration and jerk profiles "
                    f"(max ratio: {max_ratio:.2f}), consistent with smooth classical dynamics "
                    f"rather than discontinuous quantum jumps."
                )
        
        # KDE peak analysis
        kde_peak_counts = []
        for key, value in self.results["metrics"].items():
            if key.endswith("_kde_peak_count"):
                kde_peak_counts.append(value)
        
        if kde_peak_counts and max(kde_peak_counts) > 1:
            self.findings.append(
                f"Kernel density estimation reveals multiple peaks in the transition distance distribution, "
                f"further supporting the existence of distinct transition modes rather than a "
                f"continuous spectrum of changes."
            )
        
        # Overall assessment based on quantum score
        if quantum_score > 0.7:
            self.findings.append(
                f"OVERALL ASSESSMENT (Quantum Score: {quantum_score:.2f}): Strong evidence of quantum field-like "
                f"transitions. Personality states appear to transition through discontinuous jumps "
                f"rather than continuous evolution, supporting a field theory of personality where "
                f"stable states are separated by tunneling events."
            )
        elif quantum_score > 0.3:
            self.findings.append(
                f"OVERALL ASSESSMENT (Quantum Score: {quantum_score:.2f}): Mixed evidence of quantum and classical "
                f"transitions. Some personality state changes occur through discontinuous jumps, while "
                f"others follow continuous trajectories, suggesting a hybrid quantum-classical field model."
            )
        else:
            self.findings.append(
                f"OVERALL ASSESSMENT (Quantum Score: {quantum_score:.2f}): Limited evidence of quantum-like transitions. "
                f"Personality states primarily evolve through continuous changes rather than discontinuous "
                f"jumps, suggesting classical field dynamics predominate in personality transitions."
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
        self.logger.info("Generating transition dynamics experiment report")
        
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