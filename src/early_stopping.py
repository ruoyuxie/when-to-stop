# early_stopping.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from activation_manager import ActivationManager, ContextActivations
from probe import initialize_probe, SufficiencyProbe
from chunking_manager import ChunkingManager
from tqdm import tqdm
import config
from utils import prepare_sufficiency_input, get_suffix_prompt, clear_gpu_memory
import gc
import argparse

class EarlyOutputModel:
    def __init__(
        self,
        probe_dir: str,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        probe: Optional[SufficiencyProbe] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize model with pretrained components and probe."""
        if model is not None and tokenizer is not None:
            print("Using provided model and tokenizer...")
            self.model = model
            self.tokenizer = tokenizer
        else:
            if model_name is None:
                model_name = config.MODEL_CONFIG["model_name"]
                    
            # Load base model and tokenizer
            print(f"Loading {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=config.MODEL_CONFIG["trust_remote_code"],
                torch_dtype=torch.float16 if config.MODEL_CONFIG["torch_dtype"] == "float16" else torch.float32,
                device_map=config.MODEL_CONFIG["device_map"],
                attn_implementation=config.MODEL_CONFIG["attn_implementation"],
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=config.MODEL_CONFIG["trust_remote_code"]
            )              
            
        self.device = self.model.device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.eos_token_id = self.tokenizer.eos_token_id
            
        if probe is not None:
            print("Using provided probe...")
            self.probe = probe
        else:
            print("Initializing new probe...")
            self.probe = initialize_probe(self, probe_dir, config)
        self.activation_manager = self.probe.activation_manager
        self.chunking_manager = ChunkingManager(self.tokenizer, self.activation_manager)

    def _full_context_generate(self, query, context, task_name, example, generation_kwargs):
        sufficiency_text = prepare_sufficiency_input(query, context) + get_suffix_prompt()

        sufficiency_inputs = self.tokenizer(
            sufficiency_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        with torch.no_grad():
            # Generate
            outputs = self.model.generate(
                **sufficiency_inputs,
                **generation_kwargs,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )[len(sufficiency_text):]
        
        del sufficiency_inputs, outputs
        clear_gpu_memory()
        return {
            "generated_text": generated_text,
            "sufficient_chunk_idx": 0,
            "chunks_processed": 1,
            "sufficiency_confidence": 0,
            "used_chunk": context,
            "early_stopped": False,
            "total_chunks": 1,
            "pred_percentage": 1,
            "actual_gold_percentage": 1,
            "percentage_diff": 1,
        }
    
    def chunk_generate(self, all_input_ids, context_cache, generation_kwargs):
        suffix_inputs = self.tokenizer(
                get_suffix_prompt(), 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(self.device)
        
        all_input_ids.append(suffix_inputs.input_ids)
        combined_input_ids = torch.cat(all_input_ids, dim=1)
        final_attention_mask = torch.ones((1, combined_input_ids.size(1)), device=self.device)
        
        # Generate
        outputs = self.model.generate(
            input_ids=combined_input_ids,
            attention_mask=final_attention_mask,
            past_key_values=context_cache,
            **generation_kwargs,
        )
        
        generated_text = self.tokenizer.batch_decode(outputs[:, combined_input_ids.size(1):], skip_special_tokens=True)[0]
        return generated_text

    def generate(
        self,
        example: Dict,
        chunk: bool = True,
        max_length: Optional[int] = None,
        threshold: Optional[float] = None,
        **generation_kwargs
    ) -> Dict:
        """Generate response with early stopping and activation caching."""
        query = example["input"]
        context = example["context"]
        task_name = example["task"]

        generation_kwargs = config.GENERATION_CONFIG
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        
        if not chunk:
            return self._full_context_generate(query, context, task_name, example, generation_kwargs)
            
        context_cache = DynamicCache()
        all_input_ids = []
        total_length = 0
        
        chunk_boundaries, context_tokens = self.chunking_manager.create_chunks(
            context_activations=None,
            strategy=config.EARLY_OUTPUT_CONFIG["chunking_strategy"],
            task_name=task_name,
            num_chunks=config.EARLY_OUTPUT_CONFIG["percentage_num"],
            context=context
        )
        
        context_tokens = context_tokens[0]
        gold_percentage = self.activation_manager.find_gold_location(
            prepare_sufficiency_input(query, context),
            example.get("gold_information", [])
        ).percentage
        
        total_chunks = len(chunk_boundaries)
        
        # Process chunks
        for chunk_idx, end_boundary in enumerate(chunk_boundaries):
            start_boundary = 0 if chunk_idx == 0 else chunk_boundaries[chunk_idx - 1]
            current_chunk_tokens = context_tokens[start_boundary:end_boundary]
            current_chunk_text = self.tokenizer.decode(
                current_chunk_tokens,
                add_special_tokens=False,
            )
            
            sufficiency_text = prepare_sufficiency_input(query, current_chunk_text) if chunk_idx == 0 else current_chunk_text
            sufficiency_inputs = self.tokenizer(
                sufficiency_text,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            
            input_ids = sufficiency_inputs.input_ids
            all_input_ids.append(input_ids)
            total_length += input_ids.size(1)
            attention_mask = torch.ones((1, total_length), device=self.device)
            
            # Forward pass 
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=context_cache,
                )
            
            # Check sufficiency
            is_sufficient, confidence = self.probe.evaluate_sufficiency(self.activation_manager.get_head_features_for_eval(
                ContextActivations(
                    full_context=sufficiency_text,
                    model_token_ids=sufficiency_inputs.input_ids,
                    attention_activations=self.activation_manager._process_head_features(),
                    gold_location=None
                )
            ))
            
            if is_sufficient and confidence >= threshold:
                del outputs, sufficiency_inputs, attention_mask
                
                generated_text = self.chunk_generate(all_input_ids, context_cache, generation_kwargs)
                
                pred_percentage = round(end_boundary / len(context_tokens), 4)
                percentage_diff = round((pred_percentage - gold_percentage), 4) if gold_percentage else None
                
                return {
                    "generated_text": generated_text,
                    "sufficient_chunk_idx": chunk_idx,
                    "chunks_processed": chunk_idx + 1,
                    "sufficiency_confidence": float(confidence),
                    "used_chunk": current_chunk_text,
                    "early_stopped": True,
                    "total_chunks": total_chunks,
                    "pred_percentage": pred_percentage,
                    "actual_gold_percentage": gold_percentage,
                    "percentage_diff": percentage_diff,
                }
                       
        generated_text = self.chunk_generate(all_input_ids, context_cache, generation_kwargs)
        
        del all_input_ids, context_cache
        return {
            "generated_text": generated_text,
            "sufficient_chunk_idx": total_chunks,
            "chunks_processed": total_chunks,
            "sufficiency_confidence": 1,
            "used_chunk": context,
            "early_stopped": False,
            "total_chunks": total_chunks,
            "pred_percentage": 1,
            "actual_gold_percentage": gold_percentage,
            "percentage_diff": 1 - gold_percentage,
        }

def process_examples(
    model: EarlyOutputModel,
    examples: List[Dict],
    output_dir: str,
    chunk: bool = True,
    batch_size: int = 2,
    threshold: Optional[float] = None
) -> List[Dict]:
    """Process examples in batches and collect results."""
    results = []

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        batch_results = []
        
        try:
            for example in batch:
                output = model.generate(example=example, chunk=chunk, threshold=threshold)
                torch.cuda.empty_cache() 
                model.activation_manager.attention_activations.clear()
                result = {
                    "query": example["input"],
                    "original_context": example["context"],
                    "used_chunk": output["used_chunk"],
                    "early_stopped": output["early_stopped"],
                    "confidence": output["sufficiency_confidence"],
                    "chunks_processed": output["chunks_processed"],
                    "total_chunks": output["total_chunks"],
                    "generated_text": output["generated_text"],
                    "pred_percentage": output["pred_percentage"],
                    "actual_gold_percentage": output["actual_gold_percentage"],
                    "percentage_diff": output["percentage_diff"],
                    "correct_answers": example.get("answers", []),
                    "task": example.get("task", ""),
                    "threshold": threshold if threshold is not None else config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]
                }
                
                batch_results.append(result)

            results.extend(batch_results)
             
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def main(output_dir=None, model_name=None, model=None, tokenizer=None, probe=None, run_eval=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Base directory containing the model results')
    parser.add_argument('--model_name', type=str, help='Model name to use for early stopping')
    parser.add_argument('--data_dir', type=str, help='Data directory')

    args = parser.parse_args()

    # Use command line argument if provided, otherwise use function argument
    if output_dir is not None:
        args.output_dir = output_dir
    output_dir = args.output_dir if args.output_dir is not None else None
    model_name = args.model_name if args.model_name is not None else config.MODEL_CONFIG["model_name"]
    
    if output_dir is None:
        output_dir = config.PATH_CONFIG["output_dir"]
        
    probe_dir = os.path.join(output_dir, "probe")

    NUM_EXAMPLES = config.EARLY_OUTPUT_CONFIG["number_of_evaluation_examples"]
    
    # Load test data
    test_data_path = os.path.join(output_dir, "data", "test.json")
    print(f"\nLoading test data from {test_data_path}\n")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)[:NUM_EXAMPLES]

    # Initialize model
    print("Initializing EarlyOutputModel...")
    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    if model is not None and tokenizer is not None and probe is not None:
        early_output_model = EarlyOutputModel(
            probe_dir=probe_dir,
            model=model,
            tokenizer=tokenizer,
            probe=probe
        )
    else:
        early_output_model = EarlyOutputModel(probe_dir=probe_dir, model_name=model_name)
        model = early_output_model.model
        tokenizer = early_output_model.tokenizer


    # Process examples for both modes
    thresholds = [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]] 
    
    for chunk in [False, True]:
        mode = "chunked" if chunk else "full_context"
        print(f"\nProcessing with {mode} generation mode...")
        
        if not chunk:
            # For full_context mode, just run once without threshold
            results = process_examples(
                model=early_output_model,
                examples=test_data,
                output_dir=results_dir,
                chunk=chunk,
                batch_size=config.EARLY_OUTPUT_CONFIG.get("batch_size", 1)
            )
            
            results_file = os.path.join(results_dir, f"early_output_results_{mode}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {results_file}")
            
        else:
            # For chunked mode, try different thresholds
            for threshold in thresholds:
                threshold = threshold / 100
                print(f"\nProcessing with threshold: {threshold}")
                results = process_examples(
                    model=early_output_model,
                    examples=test_data,
                    output_dir=results_dir,
                    chunk=chunk,
                    batch_size=config.EARLY_OUTPUT_CONFIG.get("batch_size", 1),
                    threshold=threshold
                )
                
                results_file = os.path.join(results_dir, f"early_output_results_{mode}_{int(threshold*100)}.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {results_file}")

                torch.cuda.empty_cache()
                gc.collect()

    # Clean up early output model while preserving model and tokenizer if needed
    del early_output_model.probe
    del early_output_model.activation_manager
    del early_output_model.chunking_manager
    torch.cuda.empty_cache()
    gc.collect()

    if run_eval:
        import early_stopping_eval
        early_stopping_eval.main(output_dir=output_dir, model=model, tokenizer=tokenizer, model_name=model_name, thresholds=thresholds)
 

if __name__ == "__main__":
    main()