import os
import sys
import yaml
import json
import logging
import traceback
import datetime
import importlib 
import torch
from typing import Dict, List, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging configuration
def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Configure a detailed logger that captures both console and file output.
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Name of the current experiment for log file naming
    
    Returns:
        Configured logger instance
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ExperimentController:
    """
    Main controller class that orchestrates personality field experiments.
    
    This class is responsible for:
    1. Loading and managing the language model
    2. Setting up and coordinating experiments
    3. Managing results storage and retrieval
    4. Generating reports
    """
    
    def __init__(self, config: Dict[str, Any], separate_configs: Dict[str, Dict[str, Any]], 
             experiment_id: str, output_dir: str):
        """
        Initialize the experiment controller with configuration.
        
        Args:
            config: Merged configuration dictionary
            separate_configs: Dictionary of separate config dictionaries (core, model, experiment)
            experiment_id: Unique identifier for this experiment run
            output_dir: Directory to store experiment outputs
        """
        # Set up base directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(output_dir, "results")
        self.logs_dir = os.path.join(output_dir, "logs")
        
        # Create necessary directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up logging
        self.experiment_id = experiment_id
        self.logger = setup_logging(self.logs_dir, experiment_id)
        self.logger.info(f"Initializing ExperimentController for experiment: {experiment_id}")
        
        # Store configurations
        self.config = config
        self.separate_configs = separate_configs
        self.output_dir = output_dir
        
        # Log configuration details
        self.logger.info(f"Using merged configuration with {len(self.config)} top-level keys")
        self.logger.debug(f"Configuration keys: {list(self.config.keys())}")
        
        if 'personalities' in self.config:
            self.logger.info(f"Found {len(self.config['personalities'])} personalities in merged config")
        else:
            # Check separate configs for personalities
            for config_type, config_dict in self.separate_configs.items():
                if 'personalities' in config_dict:
                    self.logger.info(f"Found {len(config_dict['personalities'])} personalities in {config_type} config")
                    break
            else:
                self.logger.warning("No personalities found in any configuration!")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.instrumentor = None
        self.analyzer = None
        self.visualizer = None
        self.data_repo = None
        
        # Track experiment state
        self.current_experiment = None
        self.experiment_results = {}
        
        self.logger.info("ExperimentController initialized successfully")
    
    def load_model(self) -> None:
        """
        Load the language model and tokenizer based on configuration.
        
        This method initializes the LLM that will be used for experiments.
        """
        self.logger.info(f"Loading model: {self.config['model']['name']}")
        
        try:
            # Load tokenizer
            self.logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model']['name'],
                trust_remote_code=True
            )
            self.logger.debug("Tokenizer loaded successfully")
            
            # Determine data type
            dtype = getattr(torch, self.config['model'].get('dtype', 'float16'))
            self.logger.debug(f"Using data type: {dtype}")
            
            # Load model
            self.logger.debug("Loading model (this may take some time)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model']['name'],
                torch_dtype=dtype,
                device_map=self.config['model'].get('device', 'auto'),
                trust_remote_code=True
            )
            
            # Log model information
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded successfully: {total_params:,} parameters")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                self.logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                
        except Exception as e:
            self.logger.error("Failed to load model")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_instrumentor(self, instrumentor_class) -> None:
        """
        Set up the model instrumentor for capturing internal model states.
        
        Args:
            instrumentor_class: Class to use for model instrumentation
        """
        self.logger.info("Setting up model instrumentor")
        
        if self.model is None:
            self.logger.error("Cannot set up instrumentor: Model not loaded")
            raise RuntimeError("Model must be loaded before setting up instrumentor")
                
        try:
            # Transform configuration to match ModelInstrumentor's expected format
            instrumentation_config = self.config.get('instrumentation', {})
            instrumentation_compatible = {
                'layers': [],
                'capture': [{'type': 'activations'}, {'type': 'attention_patterns'}]
            }
            
            # Convert attention_layers patterns to the expected format
            attn_layers = instrumentation_config.get('attention_layers', {})
            if attn_layers.get('enabled', True):
                for pattern in attn_layers.get('patterns', []):
                    instrumentation_compatible['layers'].append({
                        'pattern': pattern,
                        'sampling': attn_layers.get('sampling_rate', 1)
                    })
            
            # Convert mlp_layers patterns to the expected format
            mlp_layers = instrumentation_config.get('mlp_layers', {})
            if mlp_layers.get('enabled', True):
                for pattern in mlp_layers.get('patterns', []):
                    instrumentation_compatible['layers'].append({
                        'pattern': pattern,
                        'sampling': mlp_layers.get('sampling_rate', 1)
                    })
            
            # Create instrumentor instance with the transformed config
            self.instrumentor = instrumentor_class(
                model=self.model,
                tokenizer=self.tokenizer,
                config=instrumentation_compatible,
                logger=self.logger.getChild('instrumentor')
            )
            
            # Configure instrumentation
            self.instrumentor.register_hooks()
            self.logger.info("Model instrumentor setup successfully")
            
        except Exception as e:
            self.logger.error("Failed to set up model instrumentor")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_analyzer(self, analyzer_class) -> None:
        """
        Set up the activation analyzer.
        
        Args:
            analyzer_class: Class to use for activation analysis
        """
        self.logger.info("Setting up activation analyzer")
        
        try:
            self.analyzer = analyzer_class(
                config=self.config.get('analysis', {}),
                logger=self.logger.getChild('analyzer')
            )
            self.logger.info("Activation analyzer setup successfully")
            
        except Exception as e:
            self.logger.error("Failed to set up activation analyzer")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_visualizer(self, visualizer_class) -> None:
        """
        Set up the visualization engine.
        
        Args:
            visualizer_class: Class to use for visualization
        """
        self.logger.info("Setting up visualization engine")
        
        # Change this line to use experiment_id instead of experiment_name
        viz_output_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)
        
        try:
            self.visualizer = visualizer_class(
                output_dir=viz_output_dir,
                config=self.config.get('visualization', {}),
                logger=self.logger.getChild('visualizer')
            )
            self.logger.info("Visualization engine setup successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to set up visualization engine: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_data_repository(self, repo_class) -> None:
        """
        Set up the data repository for storing experiment results.
        
        Args:
            repo_class: Class to use for data repository
        """
        self.logger.info("Setting up data repository")
        
        data_dir = os.path.join(self.results_dir, self.experiment_name, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            self.data_repo = repo_class(
                base_path=data_dir,
                logger=self.logger.getChild('data_repo')
            )
            self.logger.info("Data repository setup successfully")
            
        except Exception as e:
            self.logger.error("Failed to set up data repository")
            self.logger.error(traceback.format_exc())
            raise

    def setup_components(self):
        """
        Set up all required components for the experiment.
        This method serves as a convenience wrapper to set up instrumentor, analyzer, and visualizer.
        """
        self.logger.info("Setting up all experiment components")
        
        try:
            # Determine appropriate classes for this architecture
            architecture_path = self.__module__.split('.')[-2]  # Get architecture from module path
            
            # Import appropriate classes for this architecture
            module_path = f"core.{architecture_path}"
            
            try:
                instrumentor_module = importlib.import_module(f"{module_path}.model_instrumentor")
                instrumentor_class = getattr(instrumentor_module, "ModelInstrumentor")
                
                analyzer_module = importlib.import_module(f"{module_path}.activation_analyzer")
                analyzer_class = getattr(analyzer_module, "ActivationAnalyzer")
                
                visualizer_module = importlib.import_module(f"{module_path}.visualization_engine")
                visualizer_class = getattr(visualizer_module, "VisualizationEngine")
                
                # Set up each component
                self.setup_instrumentor(instrumentor_class)
                self.setup_analyzer(analyzer_class)
                self.setup_visualizer(visualizer_class)
                
                self.logger.info("All components set up successfully")
                
            except (ImportError, AttributeError) as e:
                self.logger.error(f"Failed to import required classes: {str(e)}")
                raise RuntimeError(f"Missing required component classes: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error setting up components: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def run_experiment(self, experiment_class, experiment_name: str) -> Dict[str, Any]:
        """
        Run a specific experiment.
        
        Args:
            experiment_class: The experiment class to instantiate and run
            experiment_name: Name of the experiment for results tracking
            
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.current_experiment = experiment_name
        
        # Create experiment output directory
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Get experiment-specific config - USE THE SEPARATE CONFIGS VERSION
        experiment_config = self.separate_configs.get('experiment', {})
        
        # Debug log to see what's in the config
        self.logger.debug(f"Using experiment config with keys: {list(experiment_config.keys())}")
        
        try:
            # Create experiment instance
            experiment = experiment_class(
                controller=self,
                config=experiment_config,
                output_dir=experiment_dir,
                logger=self.logger.getChild(experiment_name)
            )
            
            # Run experiment
            self.logger.info(f"Setting up experiment: {experiment_name}")
            experiment.setup()
        
            
            self.logger.info(f"Running experiment: {experiment_name}")
            experiment.run()
            
            self.logger.info(f"Analyzing results for experiment: {experiment_name}")
            results = experiment.analyze_results()
            
            self.logger.info(f"Generating report for experiment: {experiment_name}")
            experiment.generate_report()
            
            # Store results
            self.experiment_results[experiment_name] = results
            if self.data_repo is not None:
                self.data_repo.save_experiment_data(experiment_name, results)
                    
            self.logger.info(f"Experiment {experiment_name} completed successfully")
            return results
                
        except Exception as e:
            self.logger.error(f"Error during experiment {experiment_name}")
            self.logger.error(traceback.format_exc())
            raise
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get the results of a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Check in-memory cache first
        if experiment_name in self.experiment_results:
            return self.experiment_results[experiment_name]
        
        # Try to load from data repository
        if self.data_repo is not None:
            try:
                results = self.data_repo.load_experiment_data(experiment_name)
                self.experiment_results[experiment_name] = results
                return results
            except Exception as e:
                self.logger.error(f"Failed to load results for experiment {experiment_name}")
                self.logger.error(traceback.format_exc())
                
        self.logger.warning(f"No results found for experiment {experiment_name}")
        return None
        
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive report of all experiments.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating comprehensive experiment report")
        
        # Use experiment_id instead of experiment_name
        report_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"comprehensive_report_{timestamp}.md")
        
        try:
            with open(report_path, 'w') as f:
                # Use experiment_id instead of experiment_name
                f.write(f"# Comprehensive Report: {self.experiment_id}\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write experiment summaries
                for experiment_name in self.experiment_results:
                    f.write(f"## Experiment: {experiment_name}\n\n")
                    
                    results = self.experiment_results[experiment_name]
                    if 'summary' in results:
                        f.write(results['summary'])
                    else:
                        f.write("No summary available\n\n")
                    
                    # Add key metrics
                    if 'metrics' in results:
                        f.write("### Key Metrics\n\n")
                        for metric, value in results['metrics'].items():
                            f.write(f"- **{metric}**: {value}\n")
                        f.write("\n")
                    
                    # Add visualizations
                    if 'visualizations' in results:
                        f.write("### Visualizations\n\n")
                        for viz_name, viz_path in results['visualizations'].items():
                            f.write(f"![{viz_name}]({viz_path})\n\n")
                
                # Write overall conclusions
                f.write("## Overall Conclusions\n\n")
                f.write("Analysis of quantum field-like properties in personality emergence:\n\n")
                
                # Add patterns found
                f.write("### Patterns Found\n\n")
                # This would be populated based on cross-experiment analysis
                
                # Add implications for quantum field theory
                f.write("### Implications for Quantum Field Theory of Consciousness\n\n")
                # This would be populated based on analysis
            
            self.logger.info(f"Comprehensive report generated at {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error("Failed to generate comprehensive report")
            self.logger.error(traceback.format_exc())
            raise
            
    def cleanup(self) -> None:
        """
        Clean up resources used by the experiment.
        """
        self.logger.info("Cleaning up experiment resources")
        
        try:
            # Remove instrumentation hooks
            if self.instrumentor is not None:
                self.instrumentor.remove_hooks()
                
            # Release model from memory if needed
            if self.model is not None and hasattr(self, 'release_model'):
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
                
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error("Error during cleanup")
            self.logger.error(traceback.format_exc())