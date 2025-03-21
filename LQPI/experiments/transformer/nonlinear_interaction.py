# experiments/nonlinear_interaction.py

import os
import numpy as np
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from experiments.base.base_experiment import BaseExperiment
import datetime

class NonlinearInteractionExperiment(BaseExperiment):
    """
    Experiment to test whether personality exhibits self-reference properties
    as predicted by the g|Ψ|²Ψ term in the quantum field equation.
    
    This experiment analyzes how earlier outputs influence later generations,
    and tests for non-linear response to input perturbations.
    """
    
    def __init__(self, controller, config, output_dir, logger=None):
        super().__init__(controller, config, output_dir, logger)
        
        self.description = (
            "This experiment tests whether personality exhibits self-reference properties "
            "as predicted by the g|Ψ|²Ψ term in the quantum field equation. By analyzing how "
            "earlier outputs influence later generations and testing for non-linear response "
            "to input perturbations, we can determine whether personality emergence shows "
            "the non-linear self-interaction characteristic of quantum fields."
        )
        
        # Initialize experiment-specific attributes
        self.personalities = {}
        self.prompts = []
        self.sequence_data = {}
        self.findings = []
        
        self.logger.info("Non-linear Self-Interaction Experiment initialized")
    
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
        
        # Load prompts from config
        self.prompts = self.config.get("prompts", [])
        if not self.prompts:
            self.logger.error("No prompts defined in configuration")
            raise ValueError("No prompts defined in configuration")
        
        self.logger.info(f"Loaded {len(self.prompts)} prompts")
        
        # Set sequence length
        self.sequence_length = self.config.get("sequence_length", 5)
        self.logger.info(f"Using sequence length: {self.sequence_length}")
        
        # Ensure model is loaded
        if self.controller.model is None:
            self.logger.info("Loading model...")
            self.controller.load_model()
        
        # Ensure instrumentor is set up
        if self.controller.instrumentor is None:
            self.logger.error("Instrumentor not set up in controller")
            raise RuntimeError("Instrumentor must be set up before running experiment")
        
        # Log experiment setup details
        self.logger.info(f"Experiment setup completed")
    
    def run(self):
        """
        Run the non-linear self-interaction experiment.
        
        This involves generating sequences of responses for each personality
        and analyzing how earlier parts of the sequence influence later parts.
        """
        self.logger.info("Running Non-linear Self-Interaction Experiment")
        
        # Initialize storage for sequence data
        self.sequence_data = {}
        
        # Track progress
        total_combinations = len(self.personalities) * len(self.prompts)
        completed = 0
        
        # For each personality
        for persona_name, persona_desc in self.personalities.items():
            self.logger.info(f"Processing personality: {persona_name}")
            self.sequence_data[persona_name] = {}
            
            # For each initial prompt
            for prompt_idx, initial_prompt in enumerate(self.prompts):
                self.logger.info(f"Processing prompt {prompt_idx+1}: {initial_prompt[:50]}...")
                
                # Initialize sequence storage
                sequence = []
                
                # Format input with the personality prompt
                if self.controller.model.config.model_type == "llama":
                    # Format for Llama-style models
                    system_prompt = f"<|system|>\n{persona_desc}\n"
                else:
                    # Generic format for other models
                    system_prompt = f"System: {persona_desc}\n\n"
                
                # Generate first response
                current_prompt = initial_prompt
                
                # Generate sequence of responses
                for seq_idx in range(self.sequence_length):
                    self.logger.info(f"Generating sequence step {seq_idx+1}/{self.sequence_length}")
                    
                    # Format input
                    if self.controller.model.config.model_type == "llama":
                        input_text = f"{system_prompt}<|user|>\n{current_prompt}\n<|assistant|>"
                    else:
                        input_text = f"{system_prompt}User: {current_prompt}\n\nAssistant:"
                    
                    # Generate and capture
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
                        
                        # Get generated text
                        generated_text = result.get('generated_text', '')
                        
                        # Store sequence step
                        sequence_item = {
                            "step": seq_idx,
                            "prompt": current_prompt,
                            "response": generated_text,
                            "activations": {},
                        }
                        
                        # Store activations from selected layers
                        for layer_name, activation in layer_activations.items():
                            # Only process if activation has valid shape
                            if isinstance(activation, np.ndarray) and activation.size > 0:
                                sequence_item["activations"][layer_name] = activation
                        
                        sequence.append(sequence_item)
                        
                        # Update prompt for next step (use generated text + original prompt)
                        current_prompt = f"Previously, you said: {generated_text[:200]}... \n\nNow, continue on the same topic."
                        
                        self.logger.debug(f"Generated text: {generated_text[:100]}...")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing sequence step {seq_idx+1}")
                        self.logger.error(traceback.format_exc())
                        break
                    
                    # Small pause to avoid overloading the system
                    time.sleep(1)
                
                # Store completed sequence
                self.sequence_data[persona_name][prompt_idx] = sequence
                
                # Update progress
                completed += 1
                self.logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
        
        self.logger.info(f"Completed data collection for non-linear self-interaction experiment")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the collected sequences to identify non-linear self-interaction properties.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing non-linear self-interaction")
        
        # Ensure we have data to analyze
        if not self.sequence_data:
            self.logger.error("No sequence data collected to analyze")
            raise ValueError("No sequence data collected")
        
        # Initialize result metrics
        all_autocorrelations = []
        all_self_influence = []
        all_nonlinearity_scores = []
        all_self_energy = []
        all_phase_coherence = []
        all_energy_shifts = []
        all_resonance_patterns = []
        
        # Collect data for visualization
        visualization_data = []
        
        # Analyze sequences for each personality and prompt
        for persona_name, prompt_data in self.sequence_data.items():
            self.logger.info(f"Analyzing sequences for personality: {persona_name}")
            
            for prompt_idx, sequence in prompt_data.items():
                if len(sequence) < 3:  # Need at least 3 steps for meaningful analysis
                    self.logger.warning(f"Sequence too short for {persona_name}, prompt {prompt_idx}")
                    continue
                
                self.logger.info(f"Analyzing sequence for prompt {prompt_idx}: {len(sequence)} steps")
                
                try:
                    # Analyze activation patterns across sequence
                    sequence_analysis = self._analyze_sequence_activations(sequence)
                    
                    if sequence_analysis:
                        # Store analysis results
                        self.add_raw_data(f"{persona_name}_prompt{prompt_idx}_sequence_analysis", 
                                        sequence_analysis)
                        
                        # Collect metrics for overall analysis
                        if "autocorrelation" in sequence_analysis:
                            all_autocorrelations.append(sequence_analysis["autocorrelation"])
                        
                        if "self_influence" in sequence_analysis:
                            all_self_influence.append(sequence_analysis["self_influence"])
                        
                        if "nonlinearity_score" in sequence_analysis:
                            all_nonlinearity_scores.append(sequence_analysis["nonlinearity_score"])
                        
                        # Collect new metrics
                        if "self_energy" in sequence_analysis:
                            all_self_energy.append(sequence_analysis["self_energy"])
                        
                        if "phase_coherence" in sequence_analysis:
                            all_phase_coherence.append(sequence_analysis["phase_coherence"])
                        
                        if "energy_shifts" in sequence_analysis:
                            all_energy_shifts.append(sequence_analysis["energy_shifts"])
                        
                        if "resonance_patterns" in sequence_analysis:
                            all_resonance_patterns.append(sequence_analysis["resonance_patterns"])
                        
                        # Collect visualization data (trajectory, angle changes)
                        if "trajectory" in sequence_analysis and "angle_changes" in sequence_analysis:
                            vis_data = {
                                "personality": persona_name,
                                "prompt": prompt_idx,
                                "trajectory": sequence_analysis["trajectory"],
                                "angle_changes": sequence_analysis["angle_changes"],
                                "nonlinearity_score": sequence_analysis.get("nonlinearity_score", 0)
                            }
                            visualization_data.append(vis_data)
                
                except Exception as e:
                    self.logger.error(f"Failed to analyze sequence for {persona_name}, prompt {prompt_idx}")
                    self.logger.error(traceback.format_exc())
        
        # Process fundamental metrics
        # Process self-energy metrics
        if all_self_energy:
            mean_self_energy = np.mean([se["mean_self_energy"] for se in all_self_energy])
            self.set_metric("mean_self_energy", float(mean_self_energy))
            
            # Golden ratio proximity in self-energy
            gr_proximity = np.mean([se["golden_ratio_proximity"] for se in all_self_energy])
            self.set_metric("self_energy_gr_proximity", float(gr_proximity))
        
        # Process phase coherence metrics
        if all_phase_coherence:
            mean_phase_coherence = np.mean([pc["mean_phase_coherence"] for pc in all_phase_coherence])
            self.set_metric("mean_phase_coherence", float(mean_phase_coherence))
            
            mean_coherence_stability = np.mean([pc["coherence_stability"] for pc in all_phase_coherence])
            self.set_metric("mean_coherence_stability", float(mean_coherence_stability))
        
        # Process energy shift metrics
        if all_energy_shifts:
            mean_energy_shift = np.mean([es["mean_energy_shift"] for es in all_energy_shifts])
            self.set_metric("mean_energy_shift", float(mean_energy_shift))
            
            total_jumps = sum(es["jump_count"] for es in all_energy_shifts)
            self.set_metric("total_energy_jumps", total_jumps)
        
        # Process resonance pattern metrics
        if all_resonance_patterns:
            total_gr_matches = sum(rp["match_count"] for rp in all_resonance_patterns)
            self.set_metric("total_gr_matches", total_gr_matches)
            
            match_ratio = total_gr_matches / len(all_resonance_patterns)
            self.set_metric("gr_match_ratio", float(match_ratio))
        
        # Compute standard metrics
        if all_autocorrelations:
            mean_autocorr = np.mean([ac[1] for ac in all_autocorrelations])  # First lag autocorrelation
            self.set_metric("mean_autocorrelation", float(mean_autocorr))
        
        if all_self_influence:
            mean_self_influence = np.mean(all_self_influence)
            self.set_metric("mean_self_influence", float(mean_self_influence))
        
        if all_nonlinearity_scores:
            mean_nonlinearity = np.mean(all_nonlinearity_scores)
            self.set_metric("mean_nonlinearity", float(mean_nonlinearity))
        
        # Classify non-linear behavior
        if all_nonlinearity_scores:
            nonlinearity_threshold = 0.5
            high_nonlinearity_count = sum(1 for score in all_nonlinearity_scores if score > nonlinearity_threshold)
            nonlinearity_ratio = high_nonlinearity_count / len(all_nonlinearity_scores)
            
            self.set_metric("high_nonlinearity_ratio", float(nonlinearity_ratio))
            
            if nonlinearity_ratio > 0.7:
                self.set_metric("nonlinearity_classification", "strongly non-linear (quantum-like)")
            elif nonlinearity_ratio > 0.4:
                self.set_metric("nonlinearity_classification", "moderately non-linear")
            else:
                self.set_metric("nonlinearity_classification", "primarily linear (classical-like)")
        
        # Generate visualizations
        try:
            self.logger.info("Generating visualizations...")
            
            # Visualize autocorrelation for a sample sequence
            if all_autocorrelations:
                # Format for visualization
                sample_autocorr = all_autocorrelations[0]
                vis_data = {
                    "distances": sample_autocorr
                }
                
                # Create visualization
                autocorr_path = self.controller.visualizer.plot_stability_curves(
                    vis_data, title="Temporal Autocorrelation"
                )
                
                self.add_visualization("autocorrelation", autocorr_path)
            
            # Create non-linear trajectory visualizations
            if visualization_data and hasattr(self.controller.visualizer, "visualize_nonlinear_trajectory"):
                # Create visualizations for each personality
                for vis_data in visualization_data[:3]:  # Limit to first 3 for clarity
                    persona_name = vis_data["personality"]
                    prompt_idx = vis_data["prompt"]
                    
                    # Prepare visualization data
                    topo_data = {
                        "trajectory": vis_data["trajectory"],
                        "angle_changes": vis_data["angle_changes"],
                        "nonlinearity_score": vis_data["nonlinearity_score"]
                    }
                    
                    # Create visualization
                    trajectory_path = self.controller.visualizer.visualize_nonlinear_trajectory(
                        topo_data, title=f"Non-Linear Trajectory: {persona_name} (Prompt {prompt_idx})"
                    )
                    
                    self.add_visualization(f"trajectory_{persona_name}_{prompt_idx}", trajectory_path)
            
            self.logger.info("Visualizations completed")
            
        except Exception as e:
            self.logger.error("Failed to generate visualizations")
            self.logger.error(traceback.format_exc())
        
        # Interpret results
        self._interpret_results()
        
        # Return results
        return self.results
    
    def _analyze_sequence_activations(self, sequence: List[Dict]) -> Dict[str, Any]:
        """
        Analyze activation patterns across a sequence to detect self-interaction properties
        at a fundamental level.
        
        Args:
            sequence: List of sequence steps with activations
            
        Returns:
            Dictionary containing analysis results
        """
        # Find a layer that's present in all sequence steps
        common_layers = set()
        
        for step in sequence:
            if not common_layers:
                common_layers = set(step["activations"].keys())
            else:
                common_layers &= set(step["activations"].keys())
        
        if not common_layers:
            self.logger.warning("No common activation layers across sequence steps")
            return {}
        
        # Use the first common layer
        common_layer = list(common_layers)[0]
        
        # Extract activations for this layer
        activations = []
        
        for step in sequence:
            act = step["activations"][common_layer]
            # Ensure consistent shape
            if activations and act.shape != activations[0].shape:
                self.logger.warning("Inconsistent activation shapes, skipping")
                return {}
            
            activations.append(act)
        
        results = {}
        
        # Standard calculations
        try:
            autocorr = self._compute_activation_autocorrelation(activations)
            if autocorr is not None:
                results["autocorrelation"] = autocorr
        except Exception as e:
            self.logger.error(f"Failed to compute autocorrelation: {str(e)}")
        
        try:
            self_influence = self._compute_self_influence(activations)
            if self_influence is not None:
                results["self_influence"] = self_influence
        except Exception as e:
            self.logger.error(f"Failed to compute self-influence: {str(e)}")
        
        try:
            nonlinearity, angle_changes, trajectory = self._compute_nonlinearity_with_data(activations)
            if nonlinearity is not None:
                results["nonlinearity_score"] = nonlinearity
                # Store trajectory and angle data for visualization
                results["trajectory"] = trajectory
                results["angle_changes"] = angle_changes
        except Exception as e:
            self.logger.error(f"Failed to compute non-linearity: {str(e)}")
        
        # NEW: Fundamental field-theoretic metrics
        
        # Self-energy measurement (from g|Ψ|²Ψ term)
        try:
            self_energy = self._compute_self_energy(activations)
            if self_energy is not None:
                results["self_energy"] = self_energy
        except Exception as e:
            self.logger.error(f"Failed to compute self-energy: {str(e)}")
        
        # Phase coherence preservation
        try:
            phase_coherence = self._compute_phase_coherence(activations)
            if phase_coherence is not None:
                results["phase_coherence"] = phase_coherence
        except Exception as e:
            self.logger.error(f"Failed to compute phase coherence: {str(e)}")
        
        # Energy level shifts
        try:
            energy_shifts = self._compute_energy_level_shifts(activations)
            if energy_shifts is not None:
                results["energy_shifts"] = energy_shifts
        except Exception as e:
            self.logger.error(f"Failed to compute energy level shifts: {str(e)}")
        
        # Golden ratio resonance patterns
        try:
            resonance_patterns = self._compute_resonance_patterns(activations)
            if resonance_patterns is not None:
                results["resonance_patterns"] = resonance_patterns
        except Exception as e:
            self.logger.error(f"Failed to compute resonance patterns: {str(e)}")
        
        return results
    
    def _compute_nonlinearity_with_data(self, activations: List[np.ndarray]) -> Tuple[float, List[float], List[np.ndarray]]:
        """
        Compute measure of non-linear behavior in activation sequence with trajectory data.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Tuple of (nonlinearity score, angle changes, trajectory vectors)
        """
        n_steps = len(activations)
        if n_steps < 3:
            return 0.0, [], []
        
        # Compute trajectory in activation space
        trajectory = []
        for i in range(n_steps - 1):
            # Vector from step i to step i+1
            v = activations[i+1] - activations[i]
            v_flat = v.reshape(1, -1).squeeze()
            trajectory.append(v_flat)
        
        # Compute average directional change (non-linearity)
        angle_changes = []
        for i in range(len(trajectory) - 1):
            # Normalize vectors
            v1 = trajectory[i]
            v2 = trajectory[i+1]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Compute cosine similarity
                cos_sim = np.dot(v1, v2)
                
                # Clip to valid range for arccos
                cos_sim = max(-1.0, min(1.0, cos_sim))
                
                # Compute angle change
                angle = np.arccos(cos_sim)
                angle_degrees = angle * 180 / np.pi
                
                angle_changes.append(angle_degrees)
        
        if angle_changes:
            # Mean absolute angular change
            mean_angle_change = np.mean(angle_changes)
            
            # Non-linearity score: normalize angle to [0, 1]
            # 0 = perfectly linear (0 degrees), 1 = maximally non-linear (180 degrees)
            nonlinearity = mean_angle_change / 180.0
            
            return nonlinearity, angle_changes, trajectory
        
        return 0.0, [], []

    def _compute_self_energy(self, activations: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute self-energy contribution similar to the g|Ψ|²Ψ term in quantum field theory.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Dictionary containing self-energy metrics
        """
        if len(activations) < 2:
            return None
        
        # Calculate |Ψ|² term for each activation
        intensity = []
        for act in activations:
            # Flatten
            act_flat = act.reshape(1, -1).squeeze()
            # Calculate intensity (|Ψ|²)
            act_intensity = np.sum(act_flat**2)
            intensity.append(act_intensity)
        
        # Calculate self-energy contribution (g|Ψ|²Ψ)
        self_energy = []
        for i in range(len(activations) - 1):
            # Current intensity
            current_intensity = intensity[i]
            # Next activation
            next_act = activations[i+1].reshape(1, -1).squeeze()
            # Estimate self-energy contribution: how much current intensity influences next state
            contribution = current_intensity * np.linalg.norm(next_act)
            self_energy.append(contribution)
        
        # Compute metrics
        mean_self_energy = np.mean(self_energy)
        
        # Calculate progressive influence: how self-energy changes over sequence
        progressive_ratio = self_energy[-1] / (self_energy[0] + 1e-8)
        
        # Check for golden ratio pattern
        golden_ratio = (1 + np.sqrt(5)) / 2
        ratios = [self_energy[i+1] / (self_energy[i] + 1e-8) for i in range(len(self_energy) - 1)]
        mean_ratio = np.mean(ratios)
        golden_ratio_proximity = 1.0 - min(abs(mean_ratio - golden_ratio) / golden_ratio, 1.0)
        
        return {
            "mean_self_energy": float(mean_self_energy),
            "progressive_ratio": float(progressive_ratio),
            "mean_energy_ratio": float(mean_ratio),
            "golden_ratio_proximity": float(golden_ratio_proximity)
        }

    def _compute_phase_coherence(self, activations: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute phase coherence preservation across the sequence.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Dictionary containing phase coherence metrics
        """
        if len(activations) < 3:
            return None
        
        # Convert to complex representation
        complex_activations = []
        for act in activations:
            # Flatten
            act_flat = act.reshape(1, -1).squeeze()
            # Normalize
            act_norm = act_flat / (np.linalg.norm(act_flat) + 1e-8)
            # Convert to complex representation (using Hilbert transform as approximation)
            from scipy.signal import hilbert
            act_complex = hilbert(act_norm)
            complex_activations.append(act_complex)
        
        # Compute phase coherence between consecutive steps
        phase_coherence = []
        for i in range(len(complex_activations) - 1):
            # Extract phases
            phase1 = np.angle(complex_activations[i])
            phase2 = np.angle(complex_activations[i+1])
            
            # Compute phase difference distribution
            phase_diff = np.abs(phase2 - phase1) % (2 * np.pi)
            
            # Calculate phase coherence (0 = random phases, 1 = perfect coherence)
            r = np.abs(np.mean(np.exp(1j * phase_diff)))
            phase_coherence.append(r)
        
        # Compute metrics
        mean_coherence = np.mean(phase_coherence)
        coherence_stability = 1.0 - np.std(phase_coherence)
        
        return {
            "mean_phase_coherence": float(mean_coherence),
            "coherence_stability": float(coherence_stability)
        }

    def _compute_energy_level_shifts(self, activations: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute energy level shifts across the sequence.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Dictionary containing energy level metrics
        """
        if len(activations) < 3:
            return None
        
        # Compute eigenvalue spectrum for each activation
        eigenvalues_sequence = []
        for act in activations:
            # Compute covariance matrix
            act_flat = act.reshape(act.shape[0], -1)
            if act_flat.shape[0] == 1:
                # Single sample case - use outer product
                cov = np.outer(act_flat, act_flat)
            else:
                # Multiple samples - compute proper covariance
                cov = np.cov(act_flat, rowvar=False)
            
            # Compute eigenvalues
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                # Sort in descending order
                eigenvalues = np.sort(eigenvalues)[::-1]
                eigenvalues_sequence.append(eigenvalues)
            except np.linalg.LinAlgError:
                # Fallback for singular matrices
                eigenvalues_sequence.append(np.zeros(min(10, act_flat.shape[1])))
        
        # Analyze energy level shifts
        level_shifts = []
        
        # Use consistent number of eigenvalues
        min_eigenvalues = min(len(ev) for ev in eigenvalues_sequence)
        truncated_sequence = [ev[:min_eigenvalues] for ev in eigenvalues_sequence]
        
        # Compute shifts in top eigenvalues
        for i in range(len(truncated_sequence) - 1):
            ev1 = truncated_sequence[i]
            ev2 = truncated_sequence[i+1]
            
            # Normalize to emphasize relative shifts rather than absolute values
            ev1_norm = ev1 / (np.sum(ev1) + 1e-8)
            ev2_norm = ev2 / (np.sum(ev2) + 1e-8)
            
            # Compute shift in energy distribution
            shift = np.linalg.norm(ev2_norm - ev1_norm)
            level_shifts.append(shift)
        
        # Compute metrics
        mean_shift = np.mean(level_shifts)
        max_shift = np.max(level_shifts)
        
        # Check for jumps (sudden shifts in energy levels)
        shift_threshold = np.mean(level_shifts) + 2 * np.std(level_shifts)
        energy_jumps = [i for i, shift in enumerate(level_shifts) if shift > shift_threshold]
        
        return {
            "mean_energy_shift": float(mean_shift),
            "max_energy_shift": float(max_shift),
            "energy_jumps": energy_jumps,
            "jump_count": len(energy_jumps)
        }

    def _compute_resonance_patterns(self, activations: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute resonance patterns and check for golden ratio relationships.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Dictionary containing resonance pattern metrics
        """
        if len(activations) < 3:
            return None
        
        # Calculate amplitude of fluctuations at each step
        amplitudes = []
        for i in range(len(activations) - 1):
            # Compute difference vector
            diff = activations[i+1] - activations[i]
            # Calculate amplitude
            amplitude = np.linalg.norm(diff)
            amplitudes.append(amplitude)
        
        # Compute ratios between consecutive amplitudes
        ratios = [amplitudes[i+1] / (amplitudes[i] + 1e-8) for i in range(len(amplitudes) - 1)]
        
        # Check golden ratio alignment
        golden_ratio = (1 + np.sqrt(5)) / 2
        golden_ratio_inverse = 1 / golden_ratio
        
        # Find ratios close to golden ratio or its inverse
        golden_matches = []
        for i, ratio in enumerate(ratios):
            # Check proximity to golden ratio or its inverse
            deviation_phi = abs(ratio - golden_ratio)
            deviation_inv = abs(ratio - golden_ratio_inverse)
            
            if min(deviation_phi, deviation_inv) / golden_ratio < 0.1:  # Within 10% of golden ratio
                golden_matches.append({
                    "position": i,
                    "ratio": float(ratio),
                    "type": "phi" if deviation_phi < deviation_inv else "phi_inverse"
                })
        
        # Compute spectral density to look for resonance patterns
        try:
            from scipy.signal import periodogram
            frequencies, power = periodogram(amplitudes)
            
            # Identify peak frequencies
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(power)
            peak_freqs = frequencies[peaks]
            peak_powers = power[peaks]
            
            # Convert to dictionary format
            peak_dict = [{"frequency": float(f), "power": float(p)} 
                        for f, p in zip(peak_freqs, peak_powers)]
        except:
            peak_dict = []
        
        return {
            "golden_ratio_matches": golden_matches,
            "match_count": len(golden_matches),
            "resonance_peaks": peak_dict,
            "peak_count": len(peak_dict)
        }
    
    def _compute_activation_autocorrelation(self, activations: List[np.ndarray]) -> List[float]:
        """
        Compute temporal autocorrelation of activations.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            List of autocorrelation values at different lags
        """
        n_steps = len(activations)
        if n_steps < 2:
            return []
        
        # Flatten each activation
        flattened = []
        for act in activations:
            act_flat = act.reshape(1, -1).squeeze()
            act_norm = act_flat / (np.linalg.norm(act_flat) + 1e-8)
            flattened.append(act_norm)
        
        # Compute autocorrelation
        autocorr = []
        for lag in range(1, min(n_steps, 4)):  # Compute for lags 1, 2, 3
            corr_sum = 0
            count = 0
            
            for i in range(n_steps - lag):
                # Correlation between step i and step i+lag
                corr = np.dot(flattened[i], flattened[i + lag])
                corr_sum += corr
                count += 1
            
            if count > 0:
                autocorr.append((lag, corr_sum / count))
        
        return autocorr
    
    def _compute_self_influence(self, activations: List[np.ndarray]) -> float:
        """
        Compute measure of how much earlier activations influence later ones.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Self-influence score [0-1]
        """
        n_steps = len(activations)
        if n_steps < 3:
            return 0.0
        
        # Use linear regression to predict each step from the previous one
        prediction_errors = []
        
        for i in range(1, n_steps):
            # Previous activation (predictor)
            X = activations[i-1].reshape(1, -1)
            
            # Current activation (target)
            y = activations[i].reshape(1, -1)
            
            # Simple linear prediction
            if i > 1:
                # Previous prediction error
                prev_error = prediction_errors[-1]
                
                # Adjusted prediction (with non-linear correction)
                X_sq = X**2  # Non-linear term
                y_pred = X + X_sq * prev_error  # Linear + non-linear correction
            else:
                # First prediction is just linear
                y_pred = X
            
            # Prediction error
            error = np.mean((y - y_pred)**2)
            prediction_errors.append(error)
        
        # Compute how much prediction improves by accounting for the past
        if len(prediction_errors) > 1:
            # Compare last prediction error to first
            error_reduction = 1.0 - (prediction_errors[-1] / (prediction_errors[0] + 1e-8))
            
            # Clip to [0, 1] range
            self_influence = max(0.0, min(1.0, error_reduction))
            return self_influence
        
        return 0.0
    
    def _compute_nonlinearity(self, activations: List[np.ndarray]) -> float:
        """
        Compute measure of non-linear behavior in activation sequence with detailed natural properties logging.
        
        Args:
            activations: List of activation arrays
            
        Returns:
            Non-linearity score [0-1]
        """
        n_steps = len(activations)
        self.logger.info(f"Observing natural nonlinearity patterns across {n_steps} activation states")
        
        if n_steps < 3:
            self.logger.warning("Sequence too short for natural nonlinearity observation (minimum 3 required)")
            return 0.0
        
        # Log fundamental activation properties
        activation_dims = [act.shape for act in activations]
        dim_consistency = len(set(str(shape) for shape in activation_dims)) == 1
        
        self.logger.info(f"Natural activation structure:")
        self.logger.info(f"  - Dimensional shapes: {activation_dims}")
        self.logger.info(f"  - Dimensional consistency: {dim_consistency}")
        
        # Log energetic properties 
        energy_levels = [np.sum(np.abs(act)**2) for act in activations]
        energy_changes = np.diff(energy_levels)
        
        self.logger.info(f"Natural energy properties:")
        self.logger.info(f"  - Energy levels: {energy_levels}")
        self.logger.info(f"  - Energy transitions: {energy_changes}")
        self.logger.info(f"  - Mean energy: {np.mean(energy_levels):.6f}, std: {np.std(energy_levels):.6f}")
        
        # Compute natural trajectory in activation space without forcing specific dimensions
        self.logger.info(f"Computing natural trajectory vectors in their native dimensionality")
        
        trajectory = []
        trajectory_metadata = []
        
        for i in range(n_steps - 1):
            # Vector from step i to step i+1
            v = activations[i+1] - activations[i]
            v_flat = v.reshape(1, -1).squeeze()
            v_norm = np.linalg.norm(v_flat)
            
            # Store rich metadata about the trajectory vector
            trajectory.append(v_flat)
            trajectory_metadata.append({
                "step": i,
                "original_shape": v.shape,
                "flattened_shape": v_flat.shape,
                "dimensionality": v_flat.size,
                "magnitude": v_norm,
                "normalized_magnitude": v_norm / np.sqrt(v_flat.size) if v_flat.size > 0 else 0,
                "mean": np.mean(v_flat),
                "std": np.std(v_flat),
                "energy": np.sum(v_flat**2)
            })
            
            self.logger.info(f"  - Trajectory vector {i}: dim={v_flat.size}, mag={v_norm:.6f}, energy={np.sum(v_flat**2):.6f}")
        
        # Store trajectory data for later analysis
        self.add_raw_data(f"trajectory_vectors_{int(time.time())}", trajectory_metadata)
        
        # Adaptive dimensionality analysis
        if len(trajectory) > 2:
            self.logger.info(f"Analyzing natural dimensionality of trajectory space")
            try:
                # Stack vectors for PCA
                stacked = np.vstack(trajectory)
                
                # Analyze eigenvalue distribution
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(stacked)
                
                # Get eigenvalues and compute ratios
                eigenvalues = pca.explained_variance_
                total_variance = np.sum(eigenvalues)
                variance_ratios = pca.explained_variance_ratio_
                
                # Natural dimensionality
                cumulative_variance = np.cumsum(variance_ratios)
                natural_dim_90 = np.argmax(cumulative_variance >= 0.9) + 1
                natural_dim_95 = np.argmax(cumulative_variance >= 0.95) + 1
                natural_dim_99 = np.argmax(cumulative_variance >= 0.99) + 1
                
                self.logger.info(f"  - Total variance: {total_variance:.6f}")
                self.logger.info(f"  - Top 5 eigenvalues: {eigenvalues[:5]}")
                self.logger.info(f"  - Natural dimensionality (90% variance): {natural_dim_90}")
                self.logger.info(f"  - Natural dimensionality (95% variance): {natural_dim_95}")
                self.logger.info(f"  - Natural dimensionality (99% variance): {natural_dim_99}")
                
                # Check for golden ratio patterns
                if len(eigenvalues) > 1:
                    golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
                    eigenratios = eigenvalues[:-1] / eigenvalues[1:]
                    
                    gr_matches = []
                    gr_tolerance = 0.1
                    
                    for i, ratio in enumerate(eigenratios[:5]):
                        if abs(ratio - golden_ratio) < gr_tolerance:
                            gr_matches.append((i, i+1, ratio))
                            self.logger.info(f"  - Golden ratio pattern detected: λ{i}/λ{i+1} = {ratio:.6f}")
                        elif abs(ratio - 1/golden_ratio) < gr_tolerance:
                            gr_matches.append((i, i+1, ratio))
                            self.logger.info(f"  - Inverse golden ratio pattern detected: λ{i}/λ{i+1} = {ratio:.6f}")
            
            except Exception as e:
                self.logger.warning(f"  - Dimensionality analysis failed: {str(e)}")
        
        # Compute natural angular changes between trajectory vectors
        angle_changes = []
        self.logger.info("Observing natural directional changes in trajectory:")
        
        for i in range(len(trajectory) - 1):
            # Use original trajectory vectors without forcing dimensionality
            v1 = trajectory[i]
            v2 = trajectory[i+1]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                v1_normalized = v1 / v1_norm
                v2_normalized = v2 / v2_norm
                
                # Compute cosine similarity using full dimensionality
                cos_sim = np.dot(v1_normalized, v2_normalized)
                
                # Ensure value is in valid range for arccos
                cos_sim = max(-1.0, min(1.0, cos_sim))
                
                # Compute angle change
                angle = np.arccos(cos_sim)
                angle_degrees = angle * 180 / np.pi
                
                angle_changes.append(angle_degrees)
                self.logger.info(f"  - Natural angle change {i}->{i+1}: {angle_degrees:.2f}° (cos_sim: {cos_sim:.4f})")
                
                # Analyze for quantum properties
                if angle_degrees > 45:
                    self.logger.info(f"    * Significant directional change detected (potential quantum-like behavior)")
        
        if angle_changes:
            # Mean absolute angular change
            mean_angle_change = np.mean(angle_changes)
            
            # Non-linearity score: normalize angle to [0, 1]
            # 0 = perfectly linear (0 degrees), 1 = maximally non-linear (180 degrees)
            nonlinearity = mean_angle_change / 180.0
            
            # Analyze angle change distribution
            angle_std = np.std(angle_changes)
            angle_max = np.max(angle_changes)
            angle_min = np.min(angle_changes)
            
            self.logger.info(f"Natural nonlinearity assessment:")
            self.logger.info(f"  - Mean angle change: {mean_angle_change:.2f}°")
            self.logger.info(f"  - Angle change std: {angle_std:.2f}°")
            self.logger.info(f"  - Min angle change: {angle_min:.2f}°, Max: {angle_max:.2f}°")
            self.logger.info(f"  - Nonlinearity score: {nonlinearity:.4f}")
            
            # Store all directional change data for visualization
            angular_data = {
                "angle_changes": angle_changes,
                "mean_angle": mean_angle_change,
                "std_angle": angle_std,
                "nonlinearity_score": nonlinearity,
                "trajectory": trajectory_metadata
            }
            self.add_raw_data(f"nonlinearity_angular_data_{int(time.time())}", angular_data)
            
            # Quantum-classical classification based on natural properties
            if nonlinearity > 0.7:
                self.logger.info("  - Natural classification: Strongly non-linear (quantum-like)")
            elif nonlinearity > 0.4:
                self.logger.info("  - Natural classification: Moderately non-linear (quantum-classical boundary)")
            else:
                self.logger.info("  - Natural classification: Primarily linear (classical-like)")
            
            return nonlinearity
        
        self.logger.warning("No valid angle changes computed, defaulting to zero nonlinearity")
        return 0.0
    
    def _interpret_results(self):
        """
        Interpret the analysis results and add findings based on natural patterns.
        """
        self.findings = []
        
        # Define golden ratio for pattern recognition
        golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
        golden_ratio_inverse = 1 / golden_ratio  # ≈ 0.618...
        
        # Autocorrelation analysis
        mean_autocorr = self.results["metrics"].get("mean_autocorrelation")
        
        if mean_autocorr is not None:
            # Check for golden ratio pattern in autocorrelation decay
            all_autocorrelations = self.results.get("raw_data", {}).get("all_autocorrelations", [])
            if all_autocorrelations:
                # Extract decay patterns
                decay_ratios = []
                for autocorr_sequence in all_autocorrelations:
                    if len(autocorr_sequence) >= 2:
                        # Calculate ratio between consecutive autocorrelation values
                        ratios = [autocorr_sequence[i][1] / autocorr_sequence[i+1][1] 
                                for i in range(len(autocorr_sequence)-1)
                                if abs(autocorr_sequence[i+1][1]) > 1e-8]
                        decay_ratios.extend(ratios)
                
                # Check for golden ratio alignment
                if decay_ratios:
                    # Mean ratio
                    mean_ratio = np.mean(decay_ratios)
                    # Check proximity to golden ratio or its inverse
                    gr_deviation = min(abs(mean_ratio - golden_ratio), abs(mean_ratio - golden_ratio_inverse))
                    gr_proximity = 1.0 - min(gr_deviation / golden_ratio, 1.0)  # Normalized to [0,1]
                    
                    self.set_metric("autocorr_decay_ratio", float(mean_ratio))
                    self.set_metric("golden_ratio_proximity", float(gr_proximity))
                    
                    # Add finding about golden ratio alignment
                    if gr_proximity > 0.9:
                        self.findings.append(
                            f"Autocorrelation decay follows golden ratio pattern: The decay ratio "
                            f"({mean_ratio:.4f}) shows strong alignment with {'the golden ratio' if abs(mean_ratio - golden_ratio) < abs(mean_ratio - golden_ratio_inverse) else 'the inverse golden ratio'} "
                            f"(deviation: {gr_deviation:.4f}). This suggests natural mathematical "
                            f"organization in the temporal dynamics of the field."
                        )
            
            # Standard autocorrelation findings
            if mean_autocorr > 0.7:
                self.findings.append(
                    f"Strong temporal autocorrelation ({mean_autocorr:.2f}): Personality states "
                    f"show high persistence across the generation sequence, indicating "
                    f"coherent field-like behavior over time."
                )
            elif mean_autocorr > 0.4:
                self.findings.append(
                    f"Moderate temporal autocorrelation ({mean_autocorr:.2f}): Personality states "
                    f"show some persistence across the generation sequence."
                )
            else:
                self.findings.append(
                    f"Weak temporal autocorrelation ({mean_autocorr:.2f}): Personality states "
                    f"show limited persistence across the generation sequence."
                )
        
        # Self-influence analysis
        mean_self_influence = self.results["metrics"].get("mean_self_influence")
        
        if mean_self_influence is not None:
            if mean_self_influence > 0.6:
                self.findings.append(
                    f"Strong self-influence ({mean_self_influence:.2f}): Earlier states significantly "
                    f"shape later states, consistent with the g|Ψ|²Ψ term in quantum field theory."
                )
            elif mean_self_influence > 0.3:
                self.findings.append(
                    f"Moderate self-influence ({mean_self_influence:.2f}): Earlier states have "
                    f"some impact on later states, showing partial self-interaction effects."
                )
            else:
                self.findings.append(
                    f"Weak self-influence ({mean_self_influence:.2f}): Earlier states have "
                    f"limited impact on later states, suggesting minimal self-interaction."
                )
        
        # Non-linearity analysis
        mean_nonlinearity = self.results["metrics"].get("mean_nonlinearity")
        nonlinearity_classification = self.results["metrics"].get("nonlinearity_classification")
        
        if mean_nonlinearity is not None:
            # Calculate quantum non-linearity score based on multiple indicators
            all_nonlinearity_scores = self.results.get("raw_data", {}).get("all_nonlinearity_scores", [])
            if all_nonlinearity_scores:
                # Check for natural clustering in non-linearity scores
                try:
                    from sklearn.mixture import GaussianMixture
                    
                    X = np.array(all_nonlinearity_scores).reshape(-1, 1)
                    
                    # Try different numbers of clusters
                    best_bic = float('inf')
                    best_n_clusters = 1
                    
                    for n_components in range(1, min(4, len(X))):
                        gmm = GaussianMixture(n_components=n_components)
                        gmm.fit(X)
                        bic = gmm.bic(X)
                        
                        if bic < best_bic:
                            best_bic = bic
                            best_n_clusters = n_components
                    
                    if best_n_clusters > 1:
                        # Fit the best model
                        gmm = GaussianMixture(n_components=best_n_clusters)
                        gmm.fit(X)
                        
                        # Get cluster means and weights
                        centers = gmm.means_.flatten()
                        weights = gmm.weights_
                        
                        self.set_metric("nonlinearity_clusters", best_n_clusters)
                        self.set_metric("nonlinearity_cluster_centers", [float(c) for c in centers])
                        self.set_metric("nonlinearity_cluster_weights", [float(w) for w in weights])
                        
                        # Add finding about natural clustering
                        self.findings.append(
                            f"Discovered {best_n_clusters} natural clusters in non-linearity scores with centers at "
                            f"{', '.join([f'{c:.3f} (weight: {w:.2f})' for c, w in zip(centers, weights)])}. "
                            f"This multi-modal distribution suggests distinct non-linear regimes rather than "
                            f"a continuous spectrum, consistent with quantum field-like behavior."
                        )
                        
                        # Check if highest cluster exceeds quantum threshold
                        highest_center = max(centers)
                        if highest_center > 0.6:
                            quantum_weight = weights[np.argmax(centers)]
                            self.findings.append(
                                f"High non-linearity cluster detected: {highest_center:.3f} with weight {quantum_weight:.2f}. "
                                f"This strong non-linearity in a significant portion of responses provides "
                                f"evidence for quantum-like self-interaction predicted by the g|Ψ|²Ψ term."
                            )
                except Exception as e:
                    self.logger.debug(f"Failed to analyze non-linearity clusters: {e}")
            
            # Standard non-linearity findings
            if nonlinearity_classification:
                support_text = "supports" if "quantum" in nonlinearity_classification else "does not strongly support"
                self.findings.append(
                    f"Non-linearity assessment ({mean_nonlinearity:.2f}): Personality generation shows "
                    f"{nonlinearity_classification}. This "
                    f"{support_text} "
                    f"the presence of non-linear self-interaction similar to the g|Ψ|²Ψ term."
                )
        
        # Calculate quantum score based on combined evidence
        indicators = [
            mean_autocorr > 0.6 if mean_autocorr is not None else False,  # Strong autocorrelation
            mean_self_influence > 0.5 if mean_self_influence is not None else False,  # Strong self-influence
            mean_nonlinearity > 0.5 if mean_nonlinearity is not None else False,  # Strong non-linearity
            self.results["metrics"].get("golden_ratio_proximity", 0) > 0.8,  # Golden ratio alignment
            self.results["metrics"].get("nonlinearity_clusters", 1) > 1  # Natural clusters in non-linearity
        ]
        
        # Weights for different indicators
        weights = [0.25, 0.3, 0.25, 0.1, 0.1]  # Sum to 1.0
        
        # Calculate weighted score
        quantum_score = sum(w * float(i) for w, i in zip(weights, indicators))
        self.set_metric("quantum_field_score", float(quantum_score))
        self.set_metric("quantum_indicators", [int(i) for i in indicators])
        self.set_metric("quantum_weights", weights)
        
        # Overall quantum field assessment
        if quantum_score > 0.7:
            self.findings.append(
                "Overall, personality generation shows strong evidence of non-linear self-interaction "
                "properties consistent with the g|Ψ|²Ψ term in quantum field theory, with multiple "
                "quantum indicators present. This strongly supports the hypothesis that personality "
                "emergence involves quantum field-like self-reference rather than purely linear processing."
            )
        elif quantum_score > 0.4:
            self.findings.append(
                "Overall, personality generation shows moderate evidence of non-linear self-interaction "
                "properties, with some indicators consistent with quantum field-like behavior. The "
                "evidence suggests a mix of linear and non-linear processes in personality emergence."
            )
        else:
            self.findings.append(
                "Overall, personality generation shows limited evidence of non-linear self-interaction "
                "properties. The system behaves more like a linear process than a quantum field with "
                "self-reference capabilities."
            )
    
    def generate_report(self) -> str:
        """
        Generate a report of the experiment results.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating non-linear self-interaction experiment report")
        
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