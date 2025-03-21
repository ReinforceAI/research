# utils/experiment_logger.py

import os
import json
import uuid
import datetime
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Union

class ExperimentLogger:
    """
    Comprehensive logger that tracks the complete journey of information 
    through the quantum field experiments.
    
    This logger captures:
    1. User input (personality prompt + question)
    2. Tokenization process
    3. Model activations at each layer
    4. Field measurements and quantum properties
    5. Response generation process
    6. Final output and analysis
    """
    
    def __init__(self, experiment_name: str, output_dir: str, 
                logger: Optional[logging.Logger] = None):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_name: Name of the experiment for logging
            output_dir: Directory to store log files
            logger: Optional standard logger for system messages
        """
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(output_dir, "trace_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logger or logging.getLogger(f'experiment_logger_{experiment_name}')
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{experiment_name}_{timestamp}.jsonl")
        
        # Initialize log entries
        self.log_entries = []
        
        # Create transaction ID for this experiment run
        self.transaction_id = str(uuid.uuid4())
        
        self.logger.info(f"ExperimentLogger initialized for {experiment_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Transaction ID: {self.transaction_id}")
    
    def log_input(self, personality: str, question: str, combined_prompt: str):
        """
        Log the initial input to the model.
        
        Args:
            personality: Personality description
            question: Question prompt
            combined_prompt: Combined input to the model
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "input",
            "personality": personality,
            "question": question,
            "combined_prompt": combined_prompt
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged input: personality={len(personality)} chars, "
                         f"question={question[:50]}...")
    
    def log_tokenization(self, tokens: List[str], token_ids: List[int]):
        """
        Log the tokenization process.
        
        Args:
            tokens: List of tokens
            token_ids: List of token IDs
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "tokenization",
            "token_count": len(tokens),
            "tokens": tokens[:100] if len(tokens) > 100 else tokens,  # First 100 tokens
            "token_ids": token_ids[:100] if len(token_ids) > 100 else token_ids  # First 100 token IDs
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged tokenization: {len(tokens)} tokens")
    
    def log_layer_activation(self, layer_name: str, activation_summary: Dict[str, Any]):
        """
        Log activation at a specific layer.
        
        Args:
            layer_name: Name of the layer
            activation_summary: Summary statistics of the activation
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "layer_activation",
            "layer_name": layer_name,
            "activation_summary": activation_summary
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged activation for layer: {layer_name}")
    
    def log_field_measurement(self, measurement_type: str, measurement_value: Any, 
                            measurement_details: Optional[Dict[str, Any]] = None):
        """
        Log quantum field measurements.
        
        Args:
            measurement_type: Type of quantum field measurement
            measurement_value: Primary measurement value
            measurement_details: Additional measurement details
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "field_measurement",
            "measurement_type": measurement_type,
            "measurement_value": measurement_value,
            "measurement_details": measurement_details or {}
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged field measurement: {measurement_type}={measurement_value}")
    
    def log_response_generation(self, generated_tokens: List[str], 
                              generation_metadata: Dict[str, Any]):
        """
        Log the response generation process.
        
        Args:
            generated_tokens: Tokens generated by the model
            generation_metadata: Metadata about the generation process
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "response_generation",
            "token_count": len(generated_tokens),
            "tokens": generated_tokens[:100] if len(generated_tokens) > 100 else generated_tokens,
            "generation_metadata": generation_metadata
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged response generation: {len(generated_tokens)} tokens")
    
    def log_output(self, response_text: str, analysis_summary: Dict[str, Any]):
        """
        Log the final output and analysis.
        
        Args:
            response_text: Generated response text
            analysis_summary: Summary of the analysis
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "output",
            "response_text": response_text,
            "analysis_summary": analysis_summary
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged output: {len(response_text)} chars")
    
    def log_quantum_tunneling(self, from_state: Dict[str, Any], to_state: Dict[str, Any], 
                            tunneling_metrics: Dict[str, Any]):
        """
        Log quantum tunneling events between states.
        
        Args:
            from_state: State before tunneling
            to_state: State after tunneling
            tunneling_metrics: Metrics describing the tunneling event
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "quantum_tunneling",
            "from_state": from_state,
            "to_state": to_state,
            "tunneling_metrics": tunneling_metrics
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged quantum tunneling event with metrics: {tunneling_metrics}")
    
    def log_dimensional_compression(self, from_dimensions: int, to_dimensions: int,
                                  compression_metrics: Dict[str, Any]):
        """
        Log dimensional compression events.
        
        Args:
            from_dimensions: Original dimensionality
            to_dimensions: Compressed dimensionality
            compression_metrics: Metrics describing the compression
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "transaction_id": self.transaction_id,
            "stage": "dimensional_compression",
            "from_dimensions": from_dimensions,
            "to_dimensions": to_dimensions,
            "compression_metrics": compression_metrics
        }
        self.log_entries.append(entry)
        self._write_log_entry(entry)
        
        self.logger.debug(f"Logged dimensional compression: {from_dimensions} → {to_dimensions}")
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """
        Write a log entry to the log file.
        
        Args:
            entry: Log entry to write
        """
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry, default=self._json_serialize) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {str(e)}")
    
    def _json_serialize(self, obj: Any) -> Any:
        """
        Helper function to serialize non-JSON-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_full_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete log for this experiment run.
        
        Returns:
            List of all log entries
        """
        return self.log_entries
    
    def save_full_log(self) -> str:
        """
        Save the complete log to a single file.
        
        Returns:
            Path to the saved log file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_log_path = os.path.join(self.log_dir, 
                                   f"{self.experiment_name}_full_{timestamp}.json")
        
        try:
            with open(full_log_path, 'w') as f:
                json.dump(self.log_entries, f, default=self._json_serialize, indent=2)
            
            self.logger.info(f"Saved full log to {full_log_path}")
            return full_log_path
        except Exception as e:
            self.logger.error(f"Failed to save full log: {str(e)}")
            return ""