# experiments/base_experiment.py

import os
import logging
import yaml
import json
import datetime
import traceback
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger

class BaseExperiment:
    """
    Abstract base class for quantum field personality experiments.
    
    This class defines the common structure and methods for all experiments,
    providing a consistent interface for experiment execution and reporting.
    """
    
    def __init__(self, controller, config: Dict[str, Any], output_dir: str, 
                logger: Optional[logging.Logger] = None):
        """
        Initialize the base experiment.
        
        Args:
            controller: The experiment controller managing this experiment
            config: Configuration for this experiment
            output_dir: Directory to store outputs
            logger: Logger instance for detailed logging
        """
        self.controller = controller
        self.config = config
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.data_dir = os.path.join(output_dir, "data")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.experiment_logger = ExperimentLogger(
            experiment_name=self.__class__.__name__,
            output_dir=output_dir,
            logger=self.logger.getChild('tracer'))

        self.logger.info(f"Experiment trace logger initialized")

        
        # Initialize results
        self.results = {
            "experiment_name": self.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "metrics": {},
            "summary": "",
            "visualizations": {},
            "raw_data": {}
        }
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.debug(f"Experiment config: {config}")
    
    def setup(self):
        """
        Set up the experiment before running.
        
        This method should be overridden by subclasses to perform
        experiment-specific setup.
        """
        raise NotImplementedError("Subclasses must implement setup()")
    
    def run(self):
        """
        Run the experiment.
        
        This method should be overridden by subclasses to perform
        the actual experiment execution.
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the results of the experiment.
        
        This method should be overridden by subclasses to perform
        experiment-specific analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze_results()")
    
    def generate_report(self) -> str:
        """
        Generate a report of the experiment results.
        
        This method should be overridden by subclasses to generate
        experiment-specific reports.
        
        Returns:
            Path to the generated report
        """
        raise NotImplementedError("Subclasses must implement generate_report()")
    
    def save_results(self):
        """
        Save the experiment results to disk.
        """
        self.logger.info("Saving experiment results")
        
        try:
            # Save results as JSON
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.results_dir, f"results_{timestamp}.json")
            
            # Ensure all data is JSON serializable
            def json_serialize(obj):
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
            
            with open(result_path, 'w') as f:
                json.dump(self.results, f, default=json_serialize, indent=2)
            
            self.logger.info(f"Results saved to {result_path}")
            return result_path
            
        except Exception as e:
            self.logger.error("Failed to save experiment results")
            self.logger.error(traceback.format_exc())
            raise
    
    def add_visualization(self, name: str, path: str):
        """
        Add a visualization to the results.
        
        Args:
            name: Name of the visualization
            path: Path to the visualization file
        """
        self.results["visualizations"][name] = path
    
    def set_metric(self, name: str, value: Any):
        """
        Set a metric value in the results.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.results["metrics"][name] = value
    
    def add_raw_data(self, name: str, data: Any):
        """
        Add raw data to the results.
        
        Args:
            name: Name of the data
            data: The data to add
        """
        self.results["raw_data"][name] = data
    
    def set_summary(self, summary: str):
        """
        Set the summary of the experiment results.
        
        Args:
            summary: Summary text
        """
        self.results["summary"] = summary
    
    def generate_summary(self) -> str:
        """
        Generate a summary of the experiment.
        
        This is a helper method that can be used by subclasses to generate
        a standard summary format.
        
        Returns:
            Summary text
        """
        summary = f"# {self.__class__.__name__} Summary\n\n"
        summary += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add description
        if hasattr(self, 'description'):
            summary += f"## Description\n\n{self.description}\n\n"
        
        # Add key metrics
        if self.results["metrics"]:
            summary += "## Key Metrics\n\n"
            for metric, value in self.results["metrics"].items():
                summary += f"- **{metric}**: {value}\n"
            summary += "\n"
        
        # Add key findings
        if hasattr(self, 'findings') and self.findings:
            summary += "## Key Findings\n\n"
            for finding in self.findings:
                summary += f"- {finding}\n"
            summary += "\n"
        
        # Add visualizations
        if self.results["visualizations"]:
            summary += "## Visualizations\n\n"
            for viz_name, viz_path in self.results["visualizations"].items():
                summary += f"![{viz_name}]({os.path.relpath(viz_path, self.output_dir)})\n\n"
        
        return summary
    

    def get_personalities(self):
        """
        Get personalities from the experiment configuration.
        
        Returns:
            Dictionary mapping personality names to their descriptions
        """
        self.logger.info("Retrieving personalities from configuration")
        
        # Result dictionary
        personality_dict = {}
        
        # Get personalities from experiment config
        if 'personalities' in self.config and isinstance(self.config['personalities'], list):
            personalities_list = self.config['personalities']
            
            # Check if we have a list of dictionaries with name and description
            if personalities_list and isinstance(personalities_list[0], dict):
                for personality in personalities_list:
                    if isinstance(personality, dict) and "name" in personality and "description" in personality:
                        personality_dict[personality["name"]] = personality["description"]
                        self.logger.debug(f"Added personality: {personality['name']}")
            
            # Or if we have a list of strings (just names)
            elif personalities_list and isinstance(personalities_list[0], str):
                self.logger.info("Found list of personality names, searching for definitions...")
                
                # Try to find personality definitions in the controller's configuration
                if hasattr(self.controller, 'config') and 'personalities' in self.controller.config:
                    global_personalities = self.controller.config['personalities']
                    
                    # Check if global personalities is a list of dictionaries
                    if isinstance(global_personalities, list) and global_personalities and isinstance(global_personalities[0], dict):
                        # Create a lookup map for faster access
                        global_dict = {}
                        for p in global_personalities:
                            if isinstance(p, dict) and 'name' in p and 'description' in p:
                                global_dict[p['name']] = p['description']
                        
                        # Map names to descriptions
                        for name in personalities_list:
                            if name in global_dict:
                                personality_dict[name] = global_dict[name]
                                self.logger.debug(f"Added personality from global config: {name}")
                            else:
                                self.logger.warning(f"Personality '{name}' not found in global config")
        
        self.logger.info(f"Retrieved {len(personality_dict)} personalities")
        return personality_dict