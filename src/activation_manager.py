from dataclasses import dataclass
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from utils import prepare_sufficiency_input

@dataclass
class ContextActivations:
    """Store pre-computed activations and token information"""
    full_context: str
    model_token_ids: torch.Tensor
    attention_activations: Dict[str, torch.Tensor]  # store raw attention tensors
    gold_location: Optional["GoldLocation"]

@dataclass
class GoldLocation:
    """Store gold location information based on supporting sentences"""
    start_idx: int  
    end_idx: int     
    text: str
    percentage: float      

class ActivationManager:
    def __init__(self, model, model_tokenizer, device):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.device = device
        self.attention_activations = {}     
            
        # Detect model type
        self.model_type = self._detect_model_type()
        
        # Set model-specific configurations
        if self.model_type == "phi":
            self.hidden_size = self.model.config.hidden_size
            self.num_heads = self.model.config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
        elif self.model_type == "qwen2":
            self.hidden_size = self.model.config.hidden_size
            self.num_heads = self.model.config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.kv_heads = self.model.config.num_key_value_heads
        
        self._register_hooks()
    
    def _detect_model_type(self) -> str:
        """Detect the type of model being used."""
        model_name = self.model.__class__.__name__.lower()
        if "phi" in model_name:
            return "phi"
        elif "qwen2" in model_name:
            return "qwen2"
        return "llama"  # default to llama for other models
        
    def find_gold_location(self, full_context: str, gold_information: List[str]) -> Optional[GoldLocation]:
        """Find gold location indices in tokenized sequence using the gold information."""
        try:
            if not gold_information:
                print("Gold information not provided")
                return GoldLocation(
                    start_idx=0,
                    end_idx=100,
                    text="gold_sentence",
                    percentage=0.01
                )
                return None
                
            # Get the gold sentence
            gold_sentence = gold_information.strip()
            if not gold_sentence:
                return None
                
            # Find the character position of gold sentence in full context
            char_start = full_context.find(gold_sentence)
            if char_start == -1:
                return None
            char_end = char_start + len(gold_sentence)
            
            # Get text before gold sentence to find token offset
            text_before = full_context[:char_start]
            tokens_before = self.model_tokenizer(
                text_before,
                add_special_tokens=False,
            )
            
            # Get tokens for the gold sentence
            gold_tokens = self.model_tokenizer(
                gold_sentence,
                add_special_tokens=False,
            )
            
            # Calculate start and end indices in token space
            start_idx = len(tokens_before.input_ids)
            end_idx = start_idx + len(gold_tokens.input_ids)
            
            return GoldLocation(
                start_idx=start_idx,
                end_idx=end_idx,
                text=gold_sentence,
                percentage=round(char_end/len(full_context), 4)
            )
            
        except Exception as e:
            print(f"Error finding gold location: {str(e)}")
            return None
            
    def compute_activations(self, query: str, context: str, gold_information: List[str]) -> ContextActivations:
        """Compute activations with character-level alignment"""                
        self.attention_activations.clear()
        
        try:
            # Prepare full input text
            model_input_text = prepare_sufficiency_input(query, context)
    
            model_inputs = self.model_tokenizer(
                model_input_text,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Find gold location
            gold_location = self.find_gold_location(model_input_text, gold_information)
            if gold_location is None:
                print("Gold location not found")
            
            # Compute activations
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                            
                # Clear outputs immediately as they're not needed
                del outputs
            
            attention_activations = self._process_head_features()
            
            context_activations = ContextActivations(
                full_context=model_input_text,
                model_token_ids=model_inputs.input_ids.cpu(),
                attention_activations=attention_activations,
                gold_location=gold_location,
            )
            del model_inputs, attention_activations
            torch.cuda.empty_cache()

            return context_activations
            
        except Exception as e:
            print(f"Error computing activations: {str(e)}")
            raise

    def _process_head_features(self) -> Dict[str, torch.Tensor]:
        """Process attention activations into head features, preserving position information"""
        head_features = {}
        for layer_name, activation in self.attention_activations.items():
            if activation is not None:
                # No need to access [-1] since we're storing single activations
                head_features[layer_name] = activation[0]   
        return head_features
        
    def get_chunk_features(
            self,
            context_activations: ContextActivations,
            chunk_boundaries: List[int]
        ) -> List[Dict[str, np.ndarray]]:
            # Check if we have token-level chunks (each boundary differs by 1)
            is_token_level = all(
                j - i == 1 
                for i, j in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
            )
            
            chunk_features_list = []
            
            if is_token_level and len(chunk_boundaries) != 1:
                # Initialize empty dictionaries for each chunk
                chunk_features_list = [{} for _ in range(len(chunk_boundaries))]
                
                # Optimized path for token-level chunks
                for layer_name, layer_tensor in context_activations.attention_activations.items():
                    # Calculate cumulative sums for each dimension
                    # Shape: [seq_len, num_heads, head_dim]
                    cumsum = torch.cumsum(layer_tensor, dim=0)
                    
                    # Calculate means efficiently using cumsum
                    # chunk_means[i] = cumsum[i] / (i+1)
                    chunk_means = cumsum / torch.arange(1, cumsum.shape[0] + 1).view(-1, 1, 1).to(cumsum.device)
                    
                    # Convert to numpy and store features for each head
                    chunk_means_np = chunk_means.cpu().numpy()
                    for i in range(len(chunk_boundaries)):
                        for head_idx in range(chunk_means.shape[1]):
                            head_key = f"{layer_name}_head_{head_idx}"
                            chunk_features_list[i][head_key] = chunk_means_np[i, head_idx]
                            
            else:
                for layer_name, layer_tensor in context_activations.attention_activations.items():
                    # Extract layer number for consistent key naming
                    layer_num = layer_name.split('_')[1]
                    
                    # Initialize cumulative sums and counts
                    cum_sum = torch.zeros_like(layer_tensor[0])  # [num_heads, head_dim]
                    
                    for end_idx in chunk_boundaries:
                        # Calculate cumulative sum up to current boundary
                        chunk_sum = layer_tensor[:end_idx].sum(dim=0)  # Sum over sequence length
                        
                        # Calculate mean for current chunk
                        chunk_mean = chunk_sum / end_idx
                        
                        # Create feature dict for this chunk if it doesn't exist
                        if len(chunk_features_list) < len(chunk_boundaries):
                            chunk_features_list.append({})
                            
                        # Store features for each head in the current chunk
                        chunk_idx = chunk_boundaries.index(end_idx)
                        for head_idx in range(chunk_mean.shape[0]):
                            head_key = f"layer_{layer_num}_head_{head_idx}"
                            chunk_features_list[chunk_idx][head_key] = chunk_mean[head_idx].cpu().numpy()
                        
            return chunk_features_list

    def _register_hooks(self):
        """Register forward hooks to capture attention layer outputs"""
        def get_activation(name):
            def hook(module, input, output):
                if self.model_type == "phi" and "qkv_proj" in name:
                    # Handle Phi model QKV projection
                    processed = self._phi_process_activation(output)
                    if processed is not None:
                        layer_num = name.split('.layers.')[1].split('.')[0]
                        hook_name = f"layer_{layer_num}"
                        self.attention_activations[hook_name] = processed
                elif self.model_type == "qwen2" and "self_attn.o_proj" in name:
                    # Handle Qwen2 attention output
                    processed = self._qwen2_process_activation(output)
                    if processed is not None:
                        layer_num = name.split('.layers.')[1].split('.')[0]
                        hook_name = f"layer_{layer_num}"
                        self.attention_activations[hook_name]=processed
                else:
                    # Original implementation for LLaMA and other models
                    activation = output[0] if isinstance(output, tuple) else output
                    if activation is not None:
                        if isinstance(name, str) and name.startswith("layer_"):
                            hook_name = name
                        else:
                            layer_num = name.split('.')[2] if '.layers.' in name else '0'
                            hook_name = f"layer_{layer_num}"
                        processed = self._process_activation(activation)
                        if processed is not None:
                            self.attention_activations[hook_name] = processed
            return hook

        for name, module in self.model.named_modules():
            if self.model_type == "phi":
                # For Phi models, hook into the attention output
                if "self_attn.o_proj" in name:
                    module.register_forward_hook(get_activation(name))
            elif self.model_type == "qwen2":
                # For Qwen2 models, hook into the attention output
                if "self_attn.o_proj" in name:
                    module.register_forward_hook(get_activation(name))
            else:
                # Original implementation for LLaMA and other models
                if ("attn" in name and 
                    not any(x in name for x in ["key", "query", "value", "output"])):
                    layer_num = name.split('.')[2] if '.layers.' in name else '0'
                    hook_name = f"layer_{layer_num}"
                    module.register_forward_hook(get_activation(hook_name))

    def _validate_and_sanitize_activation(self, activation: torch.Tensor, context: str = "") -> Optional[torch.Tensor]:
        """Validate and sanitize activation tensor for numerical stability."""
        try:
            if activation is None:
                print(f"Warning: Received None activation{f' in {context}' if context else ''}")
                return None
                
            # Handle potential inf/nan values
            if torch.isnan(activation).any() or torch.isinf(activation).any():
                # Replace inf/nan with safe values
                activation = torch.nan_to_num(
                    activation,
                    nan=0.0,
                    posinf=torch.finfo(torch.float32).max,
                    neginf=torch.finfo(torch.float32).min
                )
            
            # Clip extremely large values to float32 range
            activation = torch.clamp(
                activation,
                min=torch.finfo(torch.float32).min,
                max=torch.finfo(torch.float32).max
            )
            
            # Safely convert to float32
            activation = activation.to(torch.float32)
            
            return activation
            
        except Exception as e:
            print(f"Error validating activation{f' in {context}' if context else ''}: {str(e)}")
            return None

    def _process_activation(self, activation: torch.Tensor) -> Optional[torch.Tensor]:
        """Process activation tensor into consistent format"""
        try:
            num_heads = self.model.config.num_attention_heads
            
            # Validate and sanitize activation
            activation = self._validate_and_sanitize_activation(activation, "process_activation")
            if activation is None:
                return None

            if len(activation.shape) == 4:
                batch_size, _, seq_len, _ = activation.shape
                processed = activation.permute(0, 2, 1, 3)
            elif len(activation.shape) == 3:
                batch_size, seq_len, hidden_dim = activation.shape
                processed = activation.view(batch_size, seq_len, num_heads, -1)
            else:
                seq_len, dim = activation.shape
                processed = activation.view(1, seq_len, num_heads, -1)
                
            return processed.detach().cpu()
            
        except Exception as e:
            print(f"Error processing activation: {str(e)}")
            return None

    def _phi_process_activation(self, activation: torch.Tensor) -> Optional[torch.Tensor]:
        """Process Phi model attention output activations"""
        try:
            # Validate and sanitize activation
            activation = self._validate_and_sanitize_activation(activation, "phi_process_activation")
            if activation is None:
                return None
            
            # For Phi attention output, shape is [batch_size, seq_len, hidden_size]
            if len(activation.shape) == 3:
                batch_size, seq_len, hidden_dim = activation.shape
                
                # Verify the dimension matches hidden size
                if hidden_dim != self.hidden_size:
                    print(f"Warning: Unexpected hidden dimension. Expected {self.hidden_size}, got {hidden_dim}")
                    return None
                
                # Reshape into attention head format
                processed = activation.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                return processed.detach().cpu()
            
            return None
            
        except Exception as e:
            print(f"Error processing Phi activation: {str(e)}")
            return None

    def _qwen2_process_activation(self, activation: torch.Tensor) -> Optional[torch.Tensor]:
        """Process Qwen2 model attention output activations"""
        try:
            # Validate and sanitize activation
            activation = self._validate_and_sanitize_activation(activation, "qwen2_process_activation")
            if activation is None:
                return None
                
            # For Qwen2 attention output, shape is [batch_size, seq_len, hidden_size]
            if len(activation.shape) == 3:
                batch_size, seq_len, hidden_dim = activation.shape
                
                # Verify the dimension matches hidden size
                if hidden_dim != self.hidden_size:
                    print(f"Warning: Unexpected hidden dimension. Expected {self.hidden_size}, got {hidden_dim}")
                    return None
                    
                # Reshape into attention head format, accounting for grouped-query attention
                # Qwen2 uses different number of heads for keys/values vs queries
                if self.num_heads > self.kv_heads:
                    # Handle grouped-query attention case
                    # Each key-value head is shared across num_heads/kv_heads query heads
                    head_ratio = self.num_heads // self.kv_heads
                    processed = activation.view(batch_size, seq_len, self.num_heads, -1)
                    # Repeat the values for each query head in the group
                    processed = processed.repeat_interleave(head_ratio, dim=2)
                else:
                    # Standard case where num_heads equals kv_heads
                    processed = activation.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                return processed.detach().cpu()
            
            return None
            
        except Exception as e:
            print(f"Error processing Qwen2 activation: {str(e)}")
            return None

    def get_head_features_for_eval(self, context_activations: ContextActivations) -> Dict[str, np.ndarray]:
        """Get head features for evaluation during early stopping."""
        head_features = {}
        
        # Process each layer's activations
        for layer_name, layer_tensor in context_activations.attention_activations.items():
            # Validate and sanitize layer tensor
            layer_tensor = self._validate_and_sanitize_activation(
                layer_tensor, 
                f"head_features_eval layer {layer_name}"
            )
            if layer_tensor is None:
                continue
                
            # Calculate mean across sequence length dimension for each head
            head_means = layer_tensor.mean(dim=0)
            
            # Extract layer number
            layer_num = layer_name.split('_')[1]
            
            # Store features for each head
            for head_idx in range(head_means.shape[0]):
                head_key = f"layer_{layer_num}_head_{head_idx}"
                head_features[head_key] = head_means[head_idx].cpu().numpy()
        
        return head_features