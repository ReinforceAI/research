import torch
import numpy as np
import logging
import traceback
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import datetime

class ModelInstrumentor:
    """
    Handles instrumentation of language models to capture internal activations and states.
    
    This class is responsible for:
    1. Setting up hooks to capture model activations
    2. Processing and organizing captured activation data
    3. Providing methods to access specific types of model internal states
    4. Computing similarity and stability metrics from activations
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the model instrumentor.
        
        Args:
            model: The language model to instrument
            tokenizer: The tokenizer for the model
            config: Configuration for instrumentation
            logger: Logger instance for detailed logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup logging
        self.logger = logger or logging.getLogger('model_instrumentor')
        
        # Initialize storage for activations and hooks
        self.activation_dict = {}
        self.attention_maps = {}
        self.key_value_cache = {}
        self.hooks = []
        self.hook_registry = {}
        
        # Track which layers are being hooked
        self.instrumented_layers = set()
        
        self.logger.info("ModelInstrumentor initialized")
        self.logger.debug(f"Instrumentation config: {config}")
    
    def register_hooks(self) -> None:
        """
        Register forward hooks on specified layers of the model.
        
        This method sets up hooks based on the configuration to capture
        activations, attention patterns, and other internal states.
        """
        self.logger.info("Registering model hooks")
        
        # Clear any existing hooks and storage
        self.remove_hooks()
        self.activation_dict.clear()
        self.attention_maps.clear()
        self.key_value_cache.clear()
        
        layer_patterns = self.config.get('layers', [])
        hook_types = self.config.get('capture', [])
        
        # Convert hook types to a set for faster lookup
        hook_types_set = {item['type'] for item in hook_types}
        
        try:
            # Helper function to create hook functions dynamically
            def create_activation_hook(name: str, layer_idx: Optional[int] = None):
                def hook(module, input, output):
                    # Store all activations by default
                    self.activation_dict[name] = output.detach()
                    
                    # Log first hook capture
                    if name not in self.instrumented_layers:
                        self.instrumented_layers.add(name)
                        self.logger.debug(f"First activation captured for {name}: "
                                         f"shape={output.shape}, dtype={output.dtype}")
                return hook
            
            def create_attention_hook(name: str, layer_idx: Optional[int] = None):
                def hook(module, input, output):
                    # For attention mechanisms, we want to capture the attention weights
                    # This assumes the output includes attention weights in a specific format
                    if isinstance(output, tuple) and len(output) > 1:
                        # Many transformer models return attention weights as the second item
                        attention_weights = output[1]
                        if attention_weights is not None:
                            self.attention_maps[name] = attention_weights.detach()
                            
                            # Log first hook capture
                            if f"attn_{name}" not in self.instrumented_layers:
                                self.instrumented_layers.add(f"attn_{name}")
                                self.logger.debug(f"First attention map captured for {name}: "
                                                f"shape={attention_weights.shape}")
                return hook
            
            def create_kv_cache_hook(name: str, layer_idx: Optional[int] = None):
                def hook(module, input, output):
                    # For key-value cache, we typically want to capture the key and value tensors
                    # This is model-specific and may need adaptation
                    if hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                        # Try to capture keys and values directly from the module
                        try:
                            with torch.no_grad():
                                dummy_input = input[0][:1, :1]  # Take a small sample of the input
                                keys = module.k_proj(dummy_input).detach()
                                values = module.v_proj(dummy_input).detach()
                                self.key_value_cache[name] = {
                                    'keys': keys,
                                    'values': values
                                }
                                
                                # Log first hook capture
                                if f"kv_{name}" not in self.instrumented_layers:
                                    self.instrumented_layers.add(f"kv_{name}")
                                    self.logger.debug(f"First KV cache captured for {name}: "
                                                    f"keys shape={keys.shape}, values shape={values.shape}")
                        except Exception as e:
                            self.logger.warning(f"Failed to capture KV cache for {name}: {str(e)}")
                return hook
            
            # Register hooks for model layers based on patterns
            hooks_registered = 0
            for pattern_config in layer_patterns:
                pattern = pattern_config['pattern']
                sampling_rate = pattern_config.get('sampling', 1)  # How many layers to skip
                
                layer_idx = 0  # Counter for sampling
                for name, module in self.model.named_modules():
                    # Check if this module matches the pattern
                    if pattern in name:
                        # Apply sampling rate - only instrument every Nth matching layer
                        if layer_idx % sampling_rate == 0:
                            # Register different types of hooks based on config
                            if 'activations' in hook_types_set:
                                hook = module.register_forward_hook(create_activation_hook(name, layer_idx))
                                self.hooks.append(hook)
                                self.hook_registry[name] = {'module': module, 'hook': hook, 'type': 'activation'}
                                hooks_registered += 1
                            
                            if 'attention_patterns' in hook_types_set and 'attn' in name:
                                hook = module.register_forward_hook(create_attention_hook(name, layer_idx))
                                self.hooks.append(hook)
                                self.hook_registry[f"attn_{name}"] = {'module': module, 'hook': hook, 'type': 'attention'}
                                hooks_registered += 1
                                
                            if 'key_value_cache' in hook_types_set and 'attn' in name:
                                hook = module.register_forward_hook(create_kv_cache_hook(name, layer_idx))
                                self.hooks.append(hook)
                                self.hook_registry[f"kv_{name}"] = {'module': module, 'hook': hook, 'type': 'kv_cache'}
                                hooks_registered += 1
                                
                        layer_idx += 1
                
            self.logger.info(f"Registered {hooks_registered} hooks across {len(self.hook_registry)} model components")
            
        except Exception as e:
            self.logger.error("Failed to register hooks")
            self.logger.error(traceback.format_exc())
            # Clean up any hooks that were successfully registered
            self.remove_hooks()
            raise
    
    def remove_hooks(self) -> None:
        """
        Remove all registered hooks from the model.
        """
        self.logger.info(f"Removing {len(self.hooks)} hooks")
        
        for hook in self.hooks:
            hook.remove()
            
        self.hooks.clear()
        self.hook_registry.clear()
        self.instrumented_layers.clear()
        
        self.logger.info("All hooks removed successfully")
    
    def capture_activations(self, input_text: str, generation_config: Optional[Dict[str, Any]] = None,
                        experiment_logger: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate text with the model and capture activations.
        
        Args:
            input_text: The input text to process
            generation_config: Configuration for text generation
            experiment_logger: Optional experiment logger for comprehensive tracing
            
        Returns:
            Dictionary containing the generated text and capture metadata
        """
        self.logger.info("Capturing activations for input text")
        self.logger.debug(f"Input text: {input_text[:100]}...")
        
        # Clear previous activations
        self.activation_dict.clear()
        self.attention_maps.clear()
        self.key_value_cache.clear()
        
        try:
            # Log detailed model information
            model_dtype = next(self.model.parameters()).dtype
            model_device = next(self.model.parameters()).device
            model_architecture = self.model.__class__.__name__
            tokenizer_type = self.tokenizer.__class__.__name__
            
            self.logger.info(f"Model architecture: {model_architecture}")
            self.logger.info(f"Model dtype: {model_dtype}")
            self.logger.info(f"Model device: {model_device}")
            self.logger.info(f"Tokenizer type: {tokenizer_type}")
            
            # If we have an experiment logger, log model info as a field measurement
            if experiment_logger is not None:
                model_info = {
                    "architecture": model_architecture,
                    "dtype": str(model_dtype),
                    "device": str(model_device),
                    "tokenizer": tokenizer_type
                }
                experiment_logger.log_field_measurement(
                    measurement_type="model_info",
                    measurement_value=model_info
                )
            
            # Configure tokenizer properly - ensure padding token exists
            self.logger.debug("Ensuring tokenizer has proper padding configuration")
            if self.tokenizer.pad_token is None:
                self.logger.info("Tokenizer has no pad_token, setting to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info(f"Special tokens: EOS={self.tokenizer.eos_token_id}, PAD={self.tokenizer.pad_token_id}")
            
            # Encode input
            self.logger.debug("Tokenizing input")
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device) if hasattr(inputs, 'attention_mask') else None
            
            input_length = input_ids.shape[1]
            self.logger.debug(f"Input encoded: {input_length} tokens")
            self.logger.debug(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            if attention_mask is not None:
                self.logger.debug(f"Attention mask shape: {attention_mask.shape}, sum: {attention_mask.sum().item()}")
            
            # If we have an experiment logger, log tokenization details
            if experiment_logger is not None:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                token_ids = input_ids[0].tolist()
                experiment_logger.log_tokenization(tokens, token_ids)
            
            # Add custom forward hook to examine logits directly
            logits_hook_data = []
            
            def logits_hook(module, input, output):
                # Examine the logits directly
                if hasattr(output, "logits"):
                    logits = output.logits
                    if isinstance(logits, torch.Tensor):
                        logits_data = {
                            "shape": list(logits.shape),
                            "dtype": str(logits.dtype),
                            "min": float(logits.min().item()),
                            "max": float(logits.max().item()),
                            "mean": float(logits.mean().item()),
                            "std": float(logits.std().item()),
                            "has_inf": bool(torch.isinf(logits).any().item()),
                            "has_nan": bool(torch.isnan(logits).any().item()),
                            "has_neg": bool((logits < 0).any().item())
                        }
                        logits_hook_data.append(logits_data)
                        self.logger.info(f"Logits stats: {logits_data}")
                        
                        # If we have an experiment logger, log the logits data
                        if experiment_logger is not None:
                            experiment_logger.log_field_measurement(
                                measurement_type="logits_stats",
                                measurement_value=logits_data
                            )
            
            # Register the hook on the model's forward method
            forward_hook = self.model.register_forward_hook(logits_hook)
            
            # Process and generate
            with torch.no_grad():
                # Default generation config if none provided
                if generation_config is None:
                    generation_config = {
                        'max_new_tokens': 100,
                        'do_sample': True,
                        'temperature': 0.7,
                        'top_p': 0.9
                    }
                
                # Log the exact generation configuration
                self.logger.info(f"Generation config: {generation_config}")
                
                # Run a single forward pass first to examine logits
                self.logger.info("Running diagnostic forward pass to examine logits")
                with torch.inference_mode():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Log the output structure
                    output_keys = list(outputs.keys()) if hasattr(outputs, 'keys') else "tensor output"
                    self.logger.info(f"Model output structure: {output_keys}")
                    
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                        self.logger.info(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
                        self.logger.info(f"Logits range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
                        self.logger.info(f"Logits has inf: {torch.isinf(logits).any().item()}")
                        self.logger.info(f"Logits has nan: {torch.isnan(logits).any().item()}")
                        self.logger.info(f"Logits has negative: {(logits < 0).any().item()}")
                        
                        # Try to apply softmax manually and check for issues
                        self.logger.info("Testing softmax calculation manually")
                        last_logits = logits[:, -1, :]
                        try:
                            # Get raw logits
                            self.logger.info(f"Last token logits shape: {last_logits.shape}")
                            self.logger.info(f"Last token logits stats: min={last_logits.min().item():.4f}, "
                                            f"max={last_logits.max().item():.4f}, "
                                            f"mean={last_logits.mean().item():.4f}")
                            
                            # Apply temperature scaling
                            temp = generation_config.get('temperature', 1.0)
                            if temp != 1.0:
                                scaled_logits = last_logits / temp
                                self.logger.info(f"After temperature ({temp}) scaling: "
                                                f"min={scaled_logits.min().item():.4f}, "
                                                f"max={scaled_logits.max().item():.4f}")
                                
                                # Check for extreme values after scaling
                                if torch.isinf(scaled_logits).any() or torch.isnan(scaled_logits).any():
                                    self.logger.warning(f"Temperature scaling produced inf/nan values!")
                            
                            # Try softmax
                            probs = torch.nn.functional.softmax(last_logits / temp, dim=-1)
                            self.logger.info(f"Probabilities sum: {probs.sum().item():.4f}")
                            self.logger.info(f"Probabilities min: {probs.min().item():.8f}")
                            self.logger.info(f"Probabilities max: {probs.max().item():.4f}")
                            self.logger.info(f"Probabilities has nan: {torch.isnan(probs).any().item()}")
                            self.logger.info(f"Probabilities has negative: {(probs < 0).any().item()}")
                            
                            # Try multinomial sampling
                            self.logger.info("Testing multinomial sampling")
                            try:
                                sampled = torch.multinomial(probs, num_samples=1)
                                self.logger.info(f"Multinomial sampling succeeded: {sampled.item()}")
                            except Exception as e:
                                self.logger.error(f"Multinomial sampling failed: {str(e)}")
                                
                        except Exception as e:
                            self.logger.error(f"Softmax testing failed: {str(e)}")
                
                # Now try generation
                try:
                    self.logger.info("Starting text generation")
                    
                    # First try with sampling
                    if generation_config.get('do_sample', False):
                        self.logger.info("Attempting generation with sampling")
                        try:
                            outputs = self.model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                pad_token_id=self.tokenizer.eos_token_id,
                                **generation_config
                            )
                            self.logger.info("Generation with sampling succeeded")
                        except Exception as e:
                            self.logger.error(f"Sampling generation failed: {str(e)}")
                            
                            # Fall back to greedy decoding
                            self.logger.info("Falling back to greedy decoding")
                            greedy_config = generation_config.copy()
                            greedy_config['do_sample'] = False
                            greedy_config['temperature'] = 1.0
                            greedy_config['num_beams'] = 1
                            
                            outputs = self.model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                pad_token_id=self.tokenizer.eos_token_id,
                                **greedy_config
                            )
                            self.logger.info("Greedy generation succeeded as fallback")
                    else:
                        # Use the provided config directly
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=self.tokenizer.eos_token_id,
                            **generation_config
                        )
                    
                    self.logger.info(f"Generation completed, output shape: {outputs.shape}")
                    
                except Exception as gen_error:
                    self.logger.error(f"Generation failed with error: {str(gen_error)}")
                    self.logger.error(traceback.format_exc())
                    raise
                
                # Remove the temporary hook
                forward_hook.remove()
                
                # Run forward pass explicitly if no activations were captured during generation
                if not self.activation_dict:
                    self.logger.warning("No activations captured during generation, running explicit forward pass")
                    with torch.inference_mode():
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Check again if activations were captured
                    if not self.activation_dict:
                        self.logger.error("Failed to capture any activations even with explicit forward pass")
                        raise ValueError("No activations could be captured from this model")
                
            # Decode output
            self.logger.debug("Decoding generated tokens")
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_length = outputs.shape[1] - input_length
            self.logger.info(f"Generated {generated_length} new tokens")
            
            # Log activation details for debugging and experiment tracing
            for layer_name, activation in self.activation_dict.items():
                if isinstance(activation, torch.Tensor):
                    # Create activation summary
                    act_info = {
                        "shape": list(activation.shape),
                        "dtype": str(activation.dtype),
                        "mean": float(activation.mean().item()),
                        "std": float(activation.std().item()),
                        "min": float(activation.min().item()),
                        "max": float(activation.max().item()),
                        "has_inf": bool(torch.isinf(activation).any().item()),
                        "has_nan": bool(torch.isnan(activation).any().item())
                    }
                    self.logger.info(f"Layer {layer_name} stats: {act_info}")
                    
                    # If we have an experiment logger, log the layer activation
                    if experiment_logger is not None:
                        experiment_logger.log_layer_activation(layer_name, act_info)
            
            # If we have an experiment logger, log the response generation
            if experiment_logger is not None:
                # Get the generated tokens (excluding the input tokens)
                generated_token_ids = outputs[0, input_length:].tolist()
                generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_token_ids)
                
                # Log the generation process
                generation_meta = {
                    "input_length": input_length,
                    "generated_length": generated_length,
                    "temperature": generation_config.get('temperature', 1.0),
                    "top_p": generation_config.get('top_p', 1.0),
                    "top_k": generation_config.get('top_k', 50),
                    "do_sample": generation_config.get('do_sample', True)
                }
                experiment_logger.log_response_generation(generated_tokens, generation_meta)
                
                # Log the final output and analysis
                output_summary = {
                    "activation_layers": len(self.activation_dict),
                    "attention_layers": len(self.attention_maps),
                    "kv_cache_layers": len(self.key_value_cache),
                    "total_tokens": input_length + generated_length
                }
                experiment_logger.log_output(generated_text, output_summary)
            
            # Collect all captured data for return
            capture_result = {
                'input_text': input_text,
                'generated_text': generated_text,
                'input_length': input_length,
                'generated_length': generated_length,
                'activation_layers': list(self.activation_dict.keys()),
                'attention_layers': list(self.attention_maps.keys()),
                'kv_cache_layers': list(self.key_value_cache.keys()),
                'logits_diagnostics': logits_hook_data,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully captured activations from {len(self.activation_dict)} layers")
            return capture_result
            
        except Exception as e:
            self.logger.error("Failed to capture activations")
            self.logger.error(traceback.format_exc())
            raise  # Re-raise the exception - we want the experiment to fail explicitly
    
    def get_activation_by_layer(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get activations for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Activation tensor for the layer, or None if not found
        """
        if layer_name in self.activation_dict:
            return self.activation_dict[layer_name]
        else:
            self.logger.warning(f"No activations found for layer: {layer_name}")
            return None
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        Get all captured attention maps.
        
        Returns:
            Dictionary of attention maps by layer name
        """
        return self.attention_maps
    
    def get_key_value_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get the key-value cache.
        
        Returns:
            Dictionary of key-value caches by layer name
        """
        return self.key_value_cache
    
    def extract_layerwise_features(self) -> Dict[str, np.ndarray]:
        """
        Extract processed features from each layer for analysis.
        
        This converts raw activations into a format suitable for analysis.
        
        Returns:
            Dictionary of numpy arrays containing processed features
        """
        self.logger.info("Extracting layerwise features from activations")
        
        features = {}
        
        try:
            for layer_name, activation in self.activation_dict.items():
                # Process based on activation shape and type
                if isinstance(activation, torch.Tensor):
                    # Move to CPU and convert to numpy
                    act_np = activation.cpu().numpy()
                    
                    # Different processing based on tensor dimensions
                    if len(act_np.shape) == 3:  # [batch, sequence, hidden]
                        # Average over sequence dimension
                        features[layer_name] = np.mean(act_np, axis=1).squeeze()
                    elif len(act_np.shape) == 2:  # [batch, hidden]
                        features[layer_name] = act_np.squeeze()
                    else:
                        # For other shapes, flatten
                        features[layer_name] = act_np.reshape(act_np.shape[0], -1)
                    
                    self.logger.debug(f"Extracted features from {layer_name}: shape={features[layer_name].shape}")
                    
                elif isinstance(activation, tuple) and all(isinstance(t, torch.Tensor) for t in activation):
                    # For tuple outputs (common in some transformers)
                    # Use the first tensor which is typically the main output
                    act_np = activation[0].cpu().numpy()
                    features[layer_name] = np.mean(act_np, axis=1).squeeze() if len(act_np.shape) == 3 else act_np.squeeze()
                    self.logger.debug(f"Extracted features from {layer_name} (tuple): shape={features[layer_name].shape}")
                    
            self.logger.info(f"Extracted features from {len(features)} layers")
            return features
            
        except Exception as e:
            self.logger.error("Failed to extract layerwise features")
            self.logger.error(traceback.format_exc())
            raise
    
    def compute_similarity_matrix(self, activation1: torch.Tensor, activation2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between two activation tensors.
        
        Args:
            activation1: First activation tensor
            activation2: Second activation tensor
            
        Returns:
            Similarity matrix
        """
        self.logger.debug("Computing similarity matrix between activations")
        
        try:
            # Ensure tensors are in the right shape
            if len(activation1.shape) == 3:  # [batch, sequence, hidden]
                # Average over sequence dimension
                act1 = activation1.mean(dim=1)
            else:
                act1 = activation1
                
            if len(activation2.shape) == 3:
                act2 = activation2.mean(dim=1)
            else:
                act2 = activation2
            
            # Normalize vectors for cosine similarity
            act1_norm = act1 / (act1.norm(dim=1, keepdim=True) + 1e-8)
            act2_norm = act2 / (act2.norm(dim=1, keepdim=True) + 1e-8)
            
            # Compute similarity matrix
            similarity = torch.mm(act1_norm, act2_norm.transpose(0, 1))
            
            self.logger.debug(f"Computed similarity matrix: shape={similarity.shape}")
            return similarity
            
        except Exception as e:
           self.logger.error("Failed to compute similarity matrix")
           self.logger.error(traceback.format_exc())
           raise
   
    def measure_activation_stability(self, activations_sequence: List[torch.Tensor]) -> Dict[str, float]:
        """
        Measure stability of activations across a sequence of activations.
        
        This helps quantify how much activations change during generation.
        
        Args:
            activations_sequence: List of activation tensors from the same layer
            
        Returns:
            Dictionary of stability metrics
        """
        self.logger.info("Measuring activation stability across sequence")
        
        try:
            if not activations_sequence or len(activations_sequence) < 2:
                self.logger.warning("Cannot measure stability: need at least 2 activation tensors")
                return {"error": "Insufficient data"}
            
            # Ensure tensors are comparable
            processed_activations = []
            for act in activations_sequence:
                if len(act.shape) == 3:  # [batch, sequence, hidden]
                    # Average over sequence dimension
                    processed_activations.append(act.mean(dim=1).cpu())
                else:
                    processed_activations.append(act.cpu())
            
            # Compute pairwise distances between consecutive steps
            distances = []
            for i in range(len(processed_activations) - 1):
                # Euclidean distance between normalized vectors
                act1 = processed_activations[i] / (processed_activations[i].norm() + 1e-8)
                act2 = processed_activations[i+1] / (processed_activations[i+1].norm() + 1e-8)
                distance = (act1 - act2).norm().item()
                distances.append(distance)
            
            # Compute stability metrics
            stability_metrics = {
                "mean_distance": np.mean(distances),
                "max_distance": np.max(distances),
                "min_distance": np.min(distances),
                "std_distance": np.std(distances),
                "distance_sequence": distances
            }
            
            # Calculate phase-like changes
            if len(distances) > 2:
                # First derivatives (rate of change)
                derivatives = np.diff(distances)
                # Second derivatives (acceleration of change)
                accelerations = np.diff(derivatives)
                
                # Find potential quantum jumps (sharp discontinuities)
                jump_threshold = np.mean(distances) + 2 * np.std(distances)
                jumps = [i for i, d in enumerate(distances) if d > jump_threshold]
                
                stability_metrics.update({
                    "derivatives": derivatives.tolist(),
                    "accelerations": accelerations.tolist(),
                    "jump_indices": jumps,
                    "jump_count": len(jumps),
                    "jump_threshold": jump_threshold
                })
            
            self.logger.info(f"Measured activation stability: mean distance={stability_metrics['mean_distance']:.6f}")
            return stability_metrics
            
        except Exception as e:
            self.logger.error("Failed to measure activation stability")
            self.logger.error(traceback.format_exc())
            raise
    
    def compute_eigenvectors(self, layer_name: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compute eigenvectors and eigenvalues for a layer's activation matrix.
        
        This helps identify the principal components of variation in the activations.
        
        Args:
            layer_name: Name of the layer
            top_k: Number of top eigenvectors to return
            
        Returns:
            Dictionary containing eigenvalues and eigenvectors
        """
        self.logger.info(f"Computing eigenvectors for layer: {layer_name}")
        
        try:
            activation = self.get_activation_by_layer(layer_name)
            if activation is None:
                return {"error": f"No activation found for layer {layer_name}"}
            
            # Process activation to 2D matrix
            if len(activation.shape) == 3:  # [batch, sequence, hidden]
                # Reshape to [batch*sequence, hidden]
                act_matrix = activation.view(-1, activation.shape[-1])
            else:
                act_matrix = activation
            
            # Compute covariance matrix
            act_np = act_matrix.cpu().numpy()
            cov_matrix = np.cov(act_np, rowvar=False)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Take top-k
            eigenvalues = eigenvalues[:top_k]
            eigenvectors = eigenvectors[:, :top_k]
            
            # Calculate cumulative explained variance
            total_var = np.sum(eigenvalues)
            explained_var = np.cumsum(eigenvalues) / total_var
            
            result = {
                "eigenvalues": eigenvalues.tolist(),
                "eigenvectors": eigenvectors.tolist(),
                "explained_variance": explained_var.tolist(),
                "total_variance": float(total_var)
            }
            
            # Check for golden ratio patterns in eigenvalue distribution
            eigenratios = eigenvalues[:-1] / eigenvalues[1:]
            golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
            golden_ratio_inverse = 1 / golden_ratio  # ≈ 0.618...
            
            # Check if any ratios are close to golden ratio or its inverse
            phi_tolerance = 0.05
            golden_matches = [
                (i, ratio) for i, ratio in enumerate(eigenratios) 
                if (abs(ratio - golden_ratio) < phi_tolerance or 
                    abs(ratio - golden_ratio_inverse) < phi_tolerance)
            ]
            
            if golden_matches:
                golden_patterns = [
                    {
                        "indices": f"{i}:{i+1}",
                        "ratio": ratio,
                        "vs_golden": ratio / golden_ratio
                    } for i, ratio in golden_matches
                ]
                result["golden_ratio_patterns"] = golden_patterns
                self.logger.info(f"Found {len(golden_patterns)} golden ratio patterns in eigenvalues")
            
            self.logger.info(f"Computed {top_k} eigenvectors for layer {layer_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compute eigenvectors for layer {layer_name}")
            self.logger.error(traceback.format_exc())
            raise
    
    def detect_state_transitions(self, activations_sequence: List[torch.Tensor], 
                                window_size: int = 5, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect quantum-like state transitions in activation sequences.
        
        Args:
            activations_sequence: List of activation tensors, typically from the same layer
            window_size: Size of windows to compare for transition detection
            threshold: Threshold for detecting significant transitions
            
        Returns:
            Dictionary containing transition analysis results
        """
        self.logger.info("Detecting state transitions in activation sequence")
        
        try:
            if len(activations_sequence) < window_size * 2:
                self.logger.warning(f"Sequence too short for transition detection with window_size={window_size}")
                return {"error": "Sequence too short for transition detection"}
            
            # Ensure tensors are comparable (flatten and normalize)
            processed_activations = []
            for act in activations_sequence:
                if len(act.shape) == 3:  # [batch, sequence, hidden]
                    # Average over sequence dimension
                    act_flat = act.mean(dim=1).flatten().cpu()
                else:
                    act_flat = act.flatten().cpu()
                
                # Normalize
                act_norm = act_flat / (act_flat.norm() + 1e-8)
                processed_activations.append(act_norm)
            
            # Compute distances between consecutive windows
            window_distances = []
            for i in range(len(processed_activations) - window_size):
                window1 = torch.stack(processed_activations[i:i+window_size]).mean(dim=0)
                window2 = torch.stack(processed_activations[i+1:i+window_size+1]).mean(dim=0)
                
                # Compute distance between normalized windows
                window1_norm = window1 / (window1.norm() + 1e-8)
                window2_norm = window2 / (window2.norm() + 1e-8)
                distance = (window1_norm - window2_norm).norm().item()
                window_distances.append(distance)
            
            # Detect jumps (transitions) where distance exceeds threshold
            mean_distance = np.mean(window_distances)
            std_distance = np.std(window_distances)
            adaptive_threshold = mean_distance + threshold * std_distance
            
            transitions = []
            for i, distance in enumerate(window_distances):
                if distance > adaptive_threshold:
                    # This corresponds to position i+window_size in the original sequence
                    transitions.append({
                        "position": i + window_size,
                        "distance": distance,
                        "z_score": (distance - mean_distance) / std_distance
                    })
            
            # Classify transitions based on patterns
            transition_types = []
            for t in transitions:
                pos = t["position"]
                if pos - window_size >= 0 and pos + window_size < len(processed_activations):
                    # Get windows before and after transition
                    before = torch.stack(processed_activations[pos-window_size:pos]).mean(dim=0)
                    after = torch.stack(processed_activations[pos:pos+window_size]).mean(dim=0)
                    
                    # Check if there are intermediate states or a clean jump
                    middle_acts = processed_activations[pos-1:pos+1]
                    middle_avg = torch.stack(middle_acts).mean(dim=0)
                    
                    # Compute distances from middle to before/after states
                    to_before = (middle_avg - before).norm().item()
                    to_after = (middle_avg - after).norm().item()
                    
                    # If middle is approximately equidistant, it's likely a superposition
                    # Otherwise, it's a clean jump if closer to either before or after
                    ratio = to_before / (to_after + 1e-8)
                    
                    if 0.8 < ratio < 1.2:
                        transition_type = "superposition"
                    else:
                        transition_type = "quantum_jump"
                        
                    transition_types.append({
                        "position": pos,
                        "type": transition_type,
                        "before_after_ratio": ratio
                    })
            
            result = {
                "distances": window_distances,
                "mean_distance": mean_distance,
                "std_distance": std_distance,
                "threshold": adaptive_threshold,
                "transitions": transitions,
                "transition_count": len(transitions),
                "transition_types": transition_types,
                "quantum_jump_count": sum(1 for t in transition_types if t["type"] == "quantum_jump"),
                "superposition_count": sum(1 for t in transition_types if t["type"] == "superposition")
            }
            
            self.logger.info(f"Detected {len(transitions)} transitions, " +
                            f"{result['quantum_jump_count']} quantum jumps, " +
                            f"{result['superposition_count']} superpositions")
            return result
            
        except Exception as e:
            self.logger.error("Failed to detect state transitions")
            self.logger.error(traceback.format_exc())
            raise