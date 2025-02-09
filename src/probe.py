# probe.py

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from probe_visualization import create_heatmap, analyze_results
import os
from final_classifier import FinalClassifier
from evaluation import SufficiencyEvaluator
from activation_manager import ActivationManager
from utils import setup_random_seeds

@dataclass
class HeadInfo:
    """Store information about an attention head's performance"""
    layer: int
    head: int
    validation_f1: float

class SufficiencyProbe:
    def __init__(self, 
                 config: Dict,
                 model: Optional[AutoModelForCausalLM] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 device: Optional[torch.device] = None):
        self.config = config
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Use provided model/tokenizer or initialize new ones
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device if device is None else device

        else:
            # Initialize new model and tokenizer if not provided
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_CONFIG["model_name"],
                trust_remote_code=config.MODEL_CONFIG["trust_remote_code"]
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_CONFIG["model_name"],
                trust_remote_code=config.MODEL_CONFIG["trust_remote_code"],
                torch_dtype=config.MODEL_CONFIG["torch_dtype"],
                device_map=config.MODEL_CONFIG["device_map"],
            )
            print(f"Model loaded: {config.MODEL_CONFIG['model_name']}")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
            self.device = self.model.device if device is None else device
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        self.activation_manager = ActivationManager(self.model, self.tokenizer, self.device)
        self.selected_heads = []
        self.is_trained = False
        # Initialize associated components
        self.classifier = FinalClassifier(config)
        self.evaluator = SufficiencyEvaluator(config)
        
    def train_and_select_heads(self, train_data: List, val_data: List, num_heads: int = 5):
        """Main training function utilizing pre-computed features"""
        torch.cuda.empty_cache()
        try:
            print("\nStage 1: Head Selection...\n")
            
            # Reorganize pre-computed features
            train_features = defaultdict(list)
            val_features = defaultdict(list)
            
            # Extract features from the new data structure
            for _, is_sufficient, _, _, _, features_dict,_ in train_data:
                for head_key, feature_vector in features_dict.items():
                    train_features[head_key].append(feature_vector)
            
            for _, is_sufficient, _, _, _, features_dict,_ in val_data:
                for head_key, feature_vector in features_dict.items():
                    val_features[head_key].append(feature_vector)
            
            # Convert lists to numpy arrays
            train_features = {k: np.array(v) for k, v in train_features.items()}
            val_features = {k: np.array(v) for k, v in val_features.items()}
            
            # Evaluate each head's performance
            head_performances = []
            self.all_head_accuracies = {}
            
            for head_key in tqdm(train_features.keys(), desc="Evaluating heads"):
                try:
                    # Features are already in numpy array format
                    X_train = train_features[head_key]
                    X_val = val_features[head_key]
                    y_train = np.array([x[1] for x in train_data])  # is_sufficient
                    y_val = np.array([x[1] for x in val_data])
                    
                    # Get head performance
                    val_f1 = self.classifier.evaluate_single_head(X_train, y_train, X_val, y_val)
                    
                    layer, head = map(int, head_key.split('_')[1::2])
                    head_info = HeadInfo(layer, head, val_f1)
                    head_performances.append(head_info)
                    
                    self.all_head_accuracies[head_key] = {
                        "layer": layer,
                        "head": head,
                        "validation_f1": round(float(val_f1), 3)
                    }
                    
                except Exception as e:
                    print(f"Error evaluating head {head_key}: {str(e)}")
                    continue
            
            # Select top performing heads
            self.selected_heads = sorted(
                head_performances,
                key=lambda x: x.validation_f1,
                reverse=True
            )[:num_heads]
            
            if not self.selected_heads:
                raise ValueError("No heads were selected during training")
            
            
            # Analyze probe head results
            create_heatmap(self.all_head_accuracies, os.path.join(self.config.PATH_CONFIG["model_output_dir"], "head_heatmap.png"))
            analyze_results(self.all_head_accuracies)
            
            
            # Stage 2: Train final classifier with selected heads
            print("\nStage 2: Training Final Classifier Based on Selected Heads...\n")
            self.classifier.train_final_classifier(
                train_features,
                val_features,
                train_data,
                val_data,
                self.selected_heads
            )
            
            self.is_trained = True
    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.is_trained = False
            raise
            
    def evaluate_sufficiency(self, features: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        """Evaluate sufficiency using precomputed features"""
        if not self.is_trained:
            raise AttributeError("Probe not trained yet")
        
        # Extract features for selected heads
        selected_features = []
        for head in self.selected_heads:
            head_key = f"layer_{head.layer}_head_{head.head}"
            if head_key in features:
                selected_features.append(features[head_key])
                
        if not selected_features:
            raise ValueError("No features found for selected heads")
            
        # Concatenate features
        feature_vector = np.concatenate(selected_features)
 
        # Get prediction probabilities
        pred_probs = self.classifier.predict_proba(feature_vector)
        is_sufficient = pred_probs[1] >= self.evaluator.threshold
        confidence = pred_probs[1]  # Always use P(sufficient) for confidence
     
        return is_sufficient, confidence

    def save(self, save_dir: str):
        """Enhanced save method that preserves all necessary state."""
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            'selected_heads': [
                {"layer": head.layer, "head": head.head, "validation_f1": head.validation_f1}
                for head in self.selected_heads
            ],
            'classifier_state': self.classifier.state_dict()
        }
        save_path = os.path.join(save_dir, "probe_state.pt")
        torch.save(state_dict, save_path)

    def initialize_probe(self, probe_dir: str, config: Dict):
        """Initialize the probe with proper configuration."""
        if not os.path.exists(probe_dir):
            raise FileNotFoundError(f"Probe directory not found: {probe_dir}")
            
        try:
            probe = SufficiencyProbe(
                config=config,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            save_path = os.path.join(probe_dir, "probe_state.pt")
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"No saved probe found at {save_path}")
            
            state_dict = torch.load(save_path)
            
            # Restore selected heads
            probe.selected_heads = [
                HeadInfo(head["layer"], head["head"], head["validation_f1"])
                for head in state_dict['selected_heads']
            ]
            
            # Load classifier state
            probe.classifier.load_state_dict(state_dict['classifier_state'])
            probe.is_trained = True
            
            print(f"Loaded probe with {len(probe.selected_heads)} selected heads")
            return probe
            
        except Exception as e:
            print(f"Detailed error while loading probe: {str(e)}")
            raise RuntimeError(f"Failed to initialize probe: {str(e)}")
        


def initialize_probe(self, probe_dir: str, config: Dict):
    """Initialize the probe with proper configuration."""
    if not os.path.exists(probe_dir):
        raise FileNotFoundError(f"Probe directory not found: {probe_dir}")
        
    try:
        probe = SufficiencyProbe(
            config=config,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        save_path = os.path.join(probe_dir, "probe_state.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No saved probe found at {save_path}")
        
        state_dict = torch.load(save_path)
        
        probe.selected_heads = [
            HeadInfo(head["layer"], head["head"], head["validation_f1"])
            for head in state_dict['selected_heads']
        ]
        
        probe.classifier.load_state_dict(state_dict['classifier_state'])
        probe.is_trained = True
        
        print(f"Loaded probe from {len(probe.selected_heads)} selected heads")
        return probe
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize probe: {str(e)}")
