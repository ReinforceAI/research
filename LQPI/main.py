#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Field Properties in Language Model Personality
-----------------------------------------------------

This script serves as the main entry point for experiments investigating
quantum field-like properties in language model personality emergence.

The experiments test whether personalities in language models exhibit properties
consistent with quantum field theory, including field-mediated organization,
quantum-like transitions, topological protection, dimensional organization,
and non-linear self-interaction.

Usage:
    python main.py --experiment personality_mapping --model phi2
    python main.py --experiment transition_dynamics --model llama3 
    python main.py --experiment personality_mapping --model phi4mini
    python main.py --experiment personality_mapping --model gemma3

"""

import os
import sys
import argparse
import logging
import yaml
import json
import datetime
import traceback
import importlib
from typing import Dict, Any, Optional, Type, Tuple, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure global logger
logger = logging.getLogger("qf_personality")


def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """
    Configure detailed logging system with console and file output.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qf_personality_{timestamp}.log")
    
    # Map string log level to actual level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set level for our specific logger
    logger.setLevel(numeric_level)
    
    logger.info(f"Logging initialized at level {log_level}")
    logger.info(f"Log file: {log_file}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for quantum field personality experiments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Quantum Field Properties in Language Model Personality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment", 
        type=str,
        required=True,
        help="Name of the experiment to run (e.g., personality_mapping)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., phi2, llama3-2, mistral_8b, phi4)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["transformer", "mixture_of_experts", "mixture_of_lora"],
        help="Override the detected model architecture"
    )
    
    parser.add_argument(
        "--core_config",
        type=str,
        default="config/core_config.yaml",
        help="Path to core configuration file"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration (overrides auto-detection)"
    )
    
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to experiment configuration (overrides auto-detection)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to store results"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip_existing", 
        action="store_true",
        help="Skip experiments with existing results"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without running the experiment"
    )
    
    return parser.parse_args()


def detect_model_architecture(model_name: str) -> str:
    """
    Detect the architecture type based on the model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        String representing the architecture type
    """
    # Model to architecture mapping
    transformer_models = ["phi2", "phi4mini","llama3", "llama2", "gemma3"]
    mixture_of_experts_models = ["mistral", "mixtral", "deepseek"]
    mixture_of_lora_models = ["phi4multimodal"]
    
    model_lower = model_name.lower()
    
    if any(model in model_lower for model in transformer_models):
        architecture = "transformer"
    elif any(model in model_lower for model in mixture_of_experts_models):
        architecture = "mixture_of_experts"
    elif any(model in model_lower for model in mixture_of_lora_models):
        architecture = "mixture_of_lora"
    else:
        # Default to transformer if unknown
        logger.warning(f"Unknown model architecture for {model_name}, defaulting to transformer")
        architecture = "transformer"
    
    logger.info(f"Detected architecture for {model_name}: {architecture}")
    return architecture


def get_configuration_paths(args: argparse.Namespace) -> Tuple[Dict[str, str], str]:
    """
    Determine the paths to configuration files based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple containing (dictionary with paths to configuration files, detected architecture)
    """
    # Detect architecture if not specified
    architecture = args.architecture
    if not architecture:
        architecture = detect_model_architecture(args.model)
    
    logger.info(f"Using architecture: {architecture}")
    
    # Build configuration paths
    config_paths = {
        "core": args.core_config
    }
    
    # Determine model configuration path
    if args.model_config:
        config_paths["model"] = args.model_config
        logger.info(f"Using specified model config: {args.model_config}")
    else:
        # Try architecture-specific model config
        arch_model_path = f"config/models/{architecture}/{args.model}_config.yaml"
        
        # Fallback to model.yaml if model_config.yaml doesn't exist
        if not os.path.exists(arch_model_path):
            arch_model_alt_path = f"config/models/{architecture}/{args.model}.yaml"
            if os.path.exists(arch_model_alt_path):
                arch_model_path = arch_model_alt_path
                logger.info(f"Using alternative model config naming format: {arch_model_alt_path}")
        
        config_paths["model"] = arch_model_path
        logger.info(f"Using auto-detected model config: {arch_model_path}")
    
    # Determine experiment configuration path
    if args.experiment_config:
        config_paths["experiment"] = args.experiment_config
        logger.info(f"Using specified experiment config: {args.experiment_config}")
    else:
        # Try architecture-specific experiment config
        arch_experiment_path = f"config/experiments/{architecture}/{args.experiment}.yaml"
        config_paths["experiment"] = arch_experiment_path
        logger.info(f"Using auto-detected experiment config: {arch_experiment_path}")
    
    # Validate that all configuration files exist
    for config_type, path in config_paths.items():
        if not os.path.exists(path):
            error_msg = f"Configuration file not found: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    return config_paths, architecture


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file with proper error handling.
    
    Args:
        file_path: Path to YAML configuration file
        
    Returns:
        Loaded configuration as dictionary
    
    Raises:
        ValueError: If the file cannot be read or parsed
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.debug(f"Successfully loaded config from {file_path}")
            return config
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML in {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error reading config file {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise ValueError(error_msg)


def deep_merge(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, with values from update_dict taking precedence.
    
    Args:
        base_dict: Base dictionary to update (modified in place)
        update_dict: Dictionary with values to update
        
    Returns:
        Updated base_dict
    """
    for key, value in update_dict.items():
        if (key in base_dict and 
            isinstance(base_dict[key], dict) and 
            isinstance(value, dict)):
            # Recursively merge nested dictionaries
            deep_merge(base_dict[key], value)
        else:
            # Otherwise, update or add the key-value pair
            base_dict[key] = value
    
    return base_dict


def load_and_merge_configs(config_paths: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Load and merge configuration files in the correct order of precedence.
    
    Args:
        config_paths: Dictionary with paths to configuration files
        
    Returns:
        Tuple containing (merged configuration dictionary, separate configs dictionary)
    """
    configs = {}
    
    # Load individual configurations
    for config_type, path in config_paths.items():
        try:
            configs[config_type] = load_yaml_config(path)
            logger.info(f"Loaded {config_type} configuration from {path}")
        except Exception as e:
            logger.error(f"Error loading {config_type} configuration from {path}")
            logger.error(traceback.format_exc())
            raise
    
    # Create merged configuration with proper precedence
    # The order of precedence is: core < model < experiment
    merged_config = {}
    
    # Start with core config
    if "core" in configs:
        merged_config.update(configs["core"])
    
    # Apply model config
    if "model" in configs:
        deep_merge(merged_config, configs["model"])
    
    # Apply experiment config
    if "experiment" in configs:
        deep_merge(merged_config, configs["experiment"])
    
    return merged_config, configs


def load_controller_class(architecture: str) -> Type:
    """
    Dynamically import the appropriate controller class based on architecture.
    
    Args:
        architecture: Model architecture (transformer, mixture_of_experts, mixture_of_lora)
        
    Returns:
        Controller class
    """
    try:
        module_path = f"core.{architecture}.experiment_controller"
        module = importlib.import_module(module_path)
        controller_class = getattr(module, "ExperimentController")
        logger.info(f"Loaded controller class: {module_path}.ExperimentController")
        return controller_class
    except (ImportError, AttributeError) as e:
        error_msg = f"Failed to load controller class for {architecture} architecture: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise ImportError(error_msg)


def load_experiment_class(experiment_name: str, architecture: str) -> Type:
    """
    Dynamically import the appropriate experiment class based on architecture.
    
    Args:
        experiment_name: Name of the experiment (e.g., personality_mapping)
        architecture: Model architecture (transformer, mixture_of_experts, mixture_of_lora)
        
    Returns:
        Experiment class
    """
    # Format the class name (e.g., personality_mapping -> PersonalityMappingExperiment)
    class_name = ''.join(word.capitalize() for word in experiment_name.split('_')) + 'Experiment'
    
    try:
        # Try architecture-specific implementation first
        module_path = f"experiments.{architecture}.{experiment_name}"
        module = importlib.import_module(module_path)
        experiment_class = getattr(module, class_name)
        logger.info(f"Loaded architecture-specific experiment class: {module_path}.{class_name}")
        return experiment_class
    except (ImportError, AttributeError) as e:
        logger.warning(f"Architecture-specific experiment not found: {str(e)}")
        
        # Try base implementation as fallback
        try:
            module_path = f"experiments.{experiment_name}"
            module = importlib.import_module(module_path)
            experiment_class = getattr(module, class_name)
            logger.info(f"Loaded generic experiment class: {module_path}.{class_name}")
            return experiment_class
        except (ImportError, AttributeError) as e:
            error_msg = f"Failed to load any implementation of experiment class for {experiment_name}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise ImportError(error_msg)


def find_available_experiments(architecture: str) -> List[str]:
    """
    Find all available experiments for the given architecture.
    
    Args:
        architecture: Model architecture
        
    Returns:
        List of available experiment names
    """
    # Basic experiment types
    experiment_types = [
        "personality_mapping", 
        "transition_dynamics", 
        "topological_protection", 
        "dimensional_analysis",
        "nonlinear_interaction"
    ]
    
    available_experiments = []
    
    logger.info(f"Searching for available experiments for {architecture} architecture")
    
    for exp_type in experiment_types:
        # Try architecture-specific implementation
        try:
            module_path = f"experiments.{architecture}.{exp_type}"
            importlib.import_module(module_path)
            available_experiments.append(exp_type)
            logger.info(f"Found architecture-specific experiment: {exp_type}")
            continue
        except ImportError:
            logger.debug(f"No architecture-specific implementation for {exp_type}")
        
        # Try generic implementation as fallback
        try:
            module_path = f"experiments.{exp_type}"
            importlib.import_module(module_path)
            available_experiments.append(exp_type)
            logger.info(f"Found generic experiment: {exp_type}")
        except ImportError:
            logger.debug(f"No implementation found for {exp_type}")
    
    return available_experiments


def run_experiment(
    experiment_class: Type,
    experiment_name: str,
    controller: Any,
    architecture: str
) -> bool:
    """
    Run a specific experiment with detailed error handling.
    
    Args:
        experiment_class: The experiment class to instantiate
        experiment_name: Name of the experiment
        controller: Controller instance
        architecture: Model architecture
        
    Returns:
        True if experiment ran successfully, False otherwise
    """
    logger.info(f"=== Starting experiment: {experiment_name} with {architecture} architecture ===")
    
    try:
        # Run the experiment through the controller
        results = controller.run_experiment(
            experiment_class, 
            experiment_name
        )
        
        logger.info(f"Experiment {experiment_name} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running experiment {experiment_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main() -> None:
    """
    Main entry point for quantum field personality experiments.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads and merges configurations
    4. Sets up the experiment controller and components
    5. Runs the specified experiment(s)
    6. Generates reports and cleans up resources
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up logging
        log_dir = os.path.join(args.output_dir, "logs")
        setup_logging(log_dir, args.log_level)
        
        logger.info("Starting Quantum Field Personality Experiments")
        logger.info(f"Experiment: {args.experiment}")
        logger.info(f"Model: {args.model}")
        
        # Get configuration paths and detect architecture
        config_paths, architecture = get_configuration_paths(args)
        
        # Load and merge configurations
        merged_config, separate_configs = load_and_merge_configs(config_paths)
        
        # Dry run option
        if args.dry_run:
            logger.info("Dry run completed successfully - configuration validated")
            logger.info(f"Would run experiment: {args.experiment} with model: {args.model}")
            return
        
        # Create timestamp and architecture-specific directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{args.experiment}_{args.model}_{timestamp}"

        # Create architecture directory if it doesn't exist
        architecture_dir = os.path.join(args.output_dir, architecture)
        os.makedirs(architecture_dir, exist_ok=True)

        # Create model directory if it doesn't exist
        model_dir = os.path.join(architecture_dir, args.model)
        os.makedirs(model_dir, exist_ok=True)

        # Create results directory within the model directory
        results_dir = os.path.join(model_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create the experiment directory inside results
        experiment_dir = os.path.join(results_dir, f"experiment_run_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        logger.info(f"Results will be stored in: {experiment_dir}")
        
        # Load architecture-specific controller class
        controller_class = load_controller_class(architecture)
        
        # Create controller instance
        controller = controller_class(
            config=merged_config,
            separate_configs=separate_configs,
            experiment_id=experiment_id,
            output_dir=experiment_dir
        )
        
        # Set up model and components
        try:
            logger.info("Loading model and setting up components")
            controller.load_model()
            controller.setup_components()
        except Exception as e:
            logger.error(f"Error setting up model and components: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Failed to set up experiment environment") from e
        
        # Determine experiments to run
        if args.experiment.lower() == "all":
            experiments_to_run = find_available_experiments(architecture)
            if not experiments_to_run:
                logger.error(f"No available experiments found for {architecture} architecture")
                return
            logger.info(f"Found {len(experiments_to_run)} available experiments")
        else:
            experiments_to_run = [args.experiment]
        
        # Run experiments
        successful_experiments = []
        failed_experiments = []
        
        for experiment_name in experiments_to_run:
            # Skip existing results if requested
            if args.skip_existing:
                experiment_result_path = os.path.join(experiment_dir, experiment_name, "results")
                if os.path.exists(experiment_result_path):
                    logger.info(f"Skipping experiment {experiment_name} - results already exist")
                    continue
            
            # Load experiment class
            try:
                experiment_class = load_experiment_class(experiment_name, architecture)
            except ImportError as e:
                logger.error(f"Could not load experiment class for {experiment_name}: {str(e)}")
                failed_experiments.append(experiment_name)
                continue
            
            # Run the experiment
            success = run_experiment(
                experiment_class,
                experiment_name,
                controller,
                architecture
            )
            
            if success:
                successful_experiments.append(experiment_name)
            else:
                failed_experiments.append(experiment_name)
        
        # Generate comprehensive report
        if successful_experiments:
            logger.info("Generating comprehensive report")
            try:
                report_path = controller.generate_comprehensive_report()
                logger.info(f"Comprehensive report generated at {report_path}")
            except Exception as e:
                logger.error(f"Error generating comprehensive report: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Summary
        logger.info("=== Experiment Run Summary ===")
        logger.info(f"Architecture: {architecture}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Total experiments: {len(experiments_to_run)}")
        logger.info(f"Successful: {len(successful_experiments)}")
        logger.info(f"Failed: {len(failed_experiments)}")
        
        if successful_experiments:
            logger.info(f"Successful experiments: {', '.join(successful_experiments)}")
        
        if failed_experiments:
            logger.info(f"Failed experiments: {', '.join(failed_experiments)}")
        
        # Clean up
        logger.info("Cleaning up resources")
        controller.cleanup()
        
        logger.info("Quantum Field Personality Experiments completed")
        
    except Exception as e:
        logger.critical(f"Unhandled exception in main function: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()