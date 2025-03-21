# utils/data_utils.py
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

def process_activations(activations: np.ndarray) -> np.ndarray:
    """
    Process raw activations into a format suitable for analysis.
    
    Args:
        activations: Raw activation data
        
    Returns:
        Processed activations
    """
    # Ensure 2D shape
    if len(activations.shape) > 2:
        # If [batch, seq, hidden], average over sequence dimension
        if len(activations.shape) == 3:
            activations = np.mean(activations, axis=1)
        else:
            # Reshape to 2D in other cases
            activations = activations.reshape(activations.shape[0], -1)
    
    return activations

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return vector / norm

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity [-1, 1]
    """
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return float(np.dot(vec1_norm, vec2_norm))

def find_golden_ratio_patterns(values: np.ndarray, tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """
    Find patterns related to the golden ratio in a sequence of values.
    
    Args:
        values: Array of values to analyze
        tolerance: Tolerance for matching the golden ratio
        
    Returns:
        List of found patterns with indices and ratios
    """
    if len(values) < 2:
        return []
    
    # Golden ratio and its inverse
    phi = (1 + np.sqrt(5)) / 2  # ≈ 1.618...
    phi_inverse = 1 / phi  # ≈ 0.618...
    
    # Compute ratios between consecutive values
    ratios = values[:-1] / values[1:]
    
    # Find patterns
    patterns = []
    for i, ratio in enumerate(ratios):
        if (abs(ratio - phi) < tolerance or 
            abs(ratio - phi_inverse) < tolerance or
            abs(ratio - phi**2) < tolerance or
            abs(ratio - 1/phi**2) < tolerance):
            
            patterns.append({
                "indices": f"{i}:{i+1}",
                "ratio": float(ratio),
                "type": "phi" if abs(ratio - phi) < tolerance else 
                       "1/phi" if abs(ratio - phi_inverse) < tolerance else
                       "phi²" if abs(ratio - phi**2) < tolerance else "1/phi²"
            })
    
    return patterns

def detect_state_jumps(sequence: np.ndarray, threshold: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detect jumps or discontinuities in a sequence of values.
    
    Args:
        sequence: Sequence of values
        threshold: Threshold for jump detection (in standard deviations)
        
    Returns:
        List of detected jumps with positions and magnitudes
    """
    if len(sequence) < 3:
        return []
    
    # Compute differences
    diffs = np.diff(sequence)
    
    # Compute statistics
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # Detect jumps
    jump_threshold = mean_diff + threshold * std_diff
    jumps = []
    
    for i, diff in enumerate(diffs):
        if diff > jump_threshold:
            jumps.append({
                "position": i,
                "magnitude": float(diff),
                "z_score": float((diff - mean_diff) / std_diff)
            })
    
    return jumps

def compute_dimensional_metrics(activations: np.ndarray) -> Dict[str, Any]:
    """
    Compute dimensional metrics for activation data.
    
    Args:
        activations: Activation data [samples, features]
        
    Returns:
        Dictionary of dimensional metrics
    """
    from sklearn.decomposition import PCA
    
    # Ensure activations is 2D
    activations = process_activations(activations)
    
    # Compute PCA
    pca = PCA()
    pca.fit(activations)
    
    # Extract eigenvalues
    eigenvalues = pca.explained_variance_
    
    # Compute metrics
    metrics = {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "effective_dimensions": (1.0 / np.sum(pca.explained_variance_ratio_**2)),
    }
    
    # Find golden ratio patterns
    metrics["golden_ratio_patterns"] = find_golden_ratio_patterns(eigenvalues)
    
    return metrics