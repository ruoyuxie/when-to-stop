import torch
import numpy as np
import random
import yaml
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate
import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
  
def initialize_model_and_tokenizer(model_name: Optional[str] = None):
    """Initialize model and tokenizer once to be reused."""
    if model_name is None:
        model_name = config.MODEL_CONFIG["model_name"]
    print(f"Initializing model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.MODEL_CONFIG["trust_remote_code"],
        torch_dtype=config.MODEL_CONFIG["torch_dtype"],
        device_map=config.MODEL_CONFIG["device_map"],
        attn_implementation=config.MODEL_CONFIG["attn_implementation"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.MODEL_CONFIG["trust_remote_code"]
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer


def prepare_sufficiency_input(query: str, context: str) -> str:
    """Format input for sufficiency checking."""
    return f"Please provide a response to the query based only on the given context:\n\n[QUERY]: {query}\n\n[CONTEXT]: {context}"

def get_suffix_prompt() -> str:
    return "\n\nYour answer:\n\n"


def setup_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
       

def get_output_dir(config):
    output_dir = config.PATH_CONFIG["output_dir"]
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    model_name_short = config.MODEL_CONFIG["model_name"].split("/")[-1]
    model_output_dir = os.path.join(output_dir, f"{timestamp}_{model_name_short}")
    os.makedirs(model_output_dir, exist_ok=True)    
    return model_output_dir

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results: Dict, output_dir: str, filename: str) -> None:
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def collect_features(probe, data: List, batch_size: int = 32) -> Dict:
    """Collect features with batching"""
    features = {}
    
    for i in tqdm(range(0, len(data), batch_size), desc="Collecting features"):
        batch = data[i:i + batch_size]
        batch_features = []
        
        for query, context, _, _, _ in batch:
            head_features = probe.collect_head_activations(query, context)
            batch_features.append(head_features)
        
        # Process batch features
        for head_features in batch_features:
            for head_key, activation in head_features.items():
                if head_key not in features:
                    features[head_key] = []
                features[head_key].append(activation)
    
    # Stack features
    return {
        head_key: np.stack(activation_list)
        for head_key, activation_list in features.items()
    }

def scale_features(features: np.ndarray) -> np.ndarray:
    """Scale features to zero mean and unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (features - mean) / std

def process_batch(batch: List, probe) -> Dict:
    """Process a batch of examples"""
    batch_features = {}
    
    for query, context, _, _, _ in batch:
        try:
            features = probe.collect_head_activations(query, context)
            for head_key, activation in features.items():
                if head_key not in batch_features:
                    batch_features[head_key] = []
                batch_features[head_key].append(activation)
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue
    
    return batch_features