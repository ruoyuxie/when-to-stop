# early_stopping_eval.py

import gc
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import os
import json
from tqdm import tqdm
import config
from prompts import evaluation_prompt
from utils import clear_gpu_memory
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
from openai import OpenAI

           
            
class Evaluator:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: Optional[str] = None,
        use_api_model: bool = False
    ):
        """Initialize evaluator with either provided model or create new one."""
        if model is not None and tokenizer is not None:
            print("Using provided model and tokenizer...")
            self.model = model
            self.tokenizer = tokenizer
            self.should_cleanup = False
            self.device = self.model.device

        elif not use_api_model and model is None and tokenizer is None:
            if model_name is None:
                model_name = config.MODEL_CONFIG["model_name"]
                
            print(f"Loading new model and tokenizer from {model_name}...")
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
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.should_cleanup = True
        
            self.device = self.model.device
        else:
            print("Using API model for evaluation...")
            self.should_cleanup = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


    def evaluate_with_openai(self, client: OpenAI, prompt: str) -> str:
        """Evaluate a single response using OpenAI."""
        MAX_RETRIES = 3
        for _ in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API call failed: {str(e)}, retrying...")
                time.sleep(3)
                continue
                
    @staticmethod
    def substring_exact_match(answer: Union[str, List[str]], prediction: str) -> float:
        """Check if the answer is a substring of the prediction."""
        prediction = prediction[0].lower() if isinstance(prediction, list) else prediction.lower()
        if isinstance(answer, list):
            return float(any(str(ans).lower() in prediction for ans in answer))
        return float(str(answer).lower() in prediction)

    def get_yes_no(self, prompt: str, eval_method: str, use_api_model: bool = False) -> Tuple[str, float, float, Dict]:
        if eval_method == "match":
            return self._get_yes_no_match(prompt, use_api_model)
        else:
            return self._get_yes_no_probability(prompt)

    def _get_yes_no_match(self, prompt: str, use_api_model: bool = False) -> Tuple[str, float, float, Dict]:
        """Handle yes/no determination using the match method"""
        if use_api_model:
            response = self.evaluate_with_openai(self.client, prompt)
        else:
            base_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    **base_inputs,
                    max_new_tokens=10,
                    temperature=config.GENERATION_CONFIG["temperature"],
                    do_sample=config.GENERATION_CONFIG["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True
                )
            
            torch.cuda.empty_cache()
            
            response = self.tokenizer.decode(
                generation_output.sequences[0][base_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        if "yes" in response.lower():
            yes_prob, no_prob = 1.0, 0.0
        elif "no" in response.lower():
            yes_prob, no_prob = 0.0, 1.0
        else:
            yes_prob = no_prob = 0.5
        
        return self._create_output(response, yes_prob, no_prob)

    def _get_yes_no_probability(self, prompt: str) -> Tuple[str, float, float, Dict]:
        """Handle yes/no determination using sequence probabilities"""
        # Create full prompts for YES and NO
        yes_prompt = prompt + " [[YES]]"
        no_prompt = prompt + " [[NO]]"
        
        # Calculate sequence probability for YES
        yes_prob = self._calculate_full_sequence_probability(yes_prompt)
        del yes_prompt
        torch.cuda.empty_cache()
        
        # Calculate sequence probability for NO
        no_prob = self._calculate_full_sequence_probability(no_prompt)
        del no_prompt
        torch.cuda.empty_cache()
        
        # Normalize probabilities
        total_prob = yes_prob + no_prob
        if total_prob > 0:
            yes_prob /= total_prob
            no_prob /= total_prob
        else:
            # Fallback to equal probabilities if both are zero
            yes_prob = no_prob = 0.5
        
        return self._create_output("", yes_prob, no_prob)


    def _calculate_full_sequence_probability(self, text: str, temperature: float = 1.0) -> float:
        """
        Calculate the probability of the entire sequence using average log probabilities 
        and temperature scaling to avoid underflow.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.device)
        input_ids = inputs.input_ids
        
        # Get the logits for the entire sequence
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calculate log probabilities for each token position
            log_probs = []
            answer_start_pos = None
            
            for pos in range(input_ids.shape[1] - 1):  # -1 because we predict next token
                next_token = input_ids[0, pos + 1]
                current_logits = logits[0, pos, :]
                
                # Apply temperature scaling to logits
                scaled_logits = current_logits / 1.5
                
                # Calculate log probabilities
                log_probs_at_pos = F.log_softmax(scaled_logits, dim=0)
                token_log_prob = log_probs_at_pos[next_token].item()
                
                # Check if we're at the start of [[YES]] or [[NO]]
                if self.tokenizer.decode(next_token).strip() == "[[":
                    answer_start_pos = len(log_probs)
                
                log_probs.append(token_log_prob)
            
            # If we found the answer position, only consider probabilities from there
            if answer_start_pos is not None:
                relevant_log_probs = log_probs[answer_start_pos:]
            else:
                # Fallback to last few tokens if we can't find the marker
                relevant_log_probs = log_probs[-5:]
            
            # Calculate average log probability for relevant tokens
            avg_log_prob = sum(relevant_log_probs) / len(relevant_log_probs)
            
            # Convert back to probability space
            sequence_prob = np.exp(avg_log_prob)
            
        del outputs, logits, input_ids, inputs, log_probs
        return sequence_prob

    def _create_output(self, response: str, yes_prob: float, no_prob: float) -> Tuple[str, float, float, Dict]:
        """Create the standardized output tuple and dictionary"""
        output_details = {
            'raw_response': response,
            'yes_probability': yes_prob,
            'no_probability': no_prob,
            'predicted': 'yes' if yes_prob > no_prob else 'no',
            'confidence': max(yes_prob, no_prob),
        }
        return response, yes_prob, no_prob, output_details

    def evaluate_results(self, data: List[Dict], evaluation_prompt_template: str, eval_method: str = "match", use_api_model: bool = False) -> Tuple[List[Dict], Dict]:

        total_score = 0
        updated_data = []
        batch_size = 1  # Process one example at a time
        
        print(f"\nProcessing examples...")
        
        # Process in smaller batches
        for idx in tqdm(range(0, len(data), batch_size)):
            batch = data[idx:idx + batch_size]
            batch_updated = []
            
            for item in batch:
                # Skip already processed items
                if "evaluation_summary" in item:
                    batch_updated.append(item)
                    continue
                
                # Extract necessary information outside try block
                task = item.get("task", "")
                question = item.get("query", "")
                context = item.get("original_context", "")
                correct_answers = item.get("correct_answers", "")
                model_answer = item.get("generated_text", "")
                prompt = None  # Initialize prompt variable
                score = 0.0   # Initialize score
                evaluation_result = None
                
                try:
                    
                    # Process based on task type
                    if task == "nothing": # everything will be evaluated using model-baed approach
                        score = self.substring_exact_match(correct_answers, model_answer)
                        evaluation_result = {
                            'type': 'direct_match',
                            'score': score
                        }
                    else:
                        formatted_answers = ('\n'.join(f"- {ans}" for ans in correct_answers) 
                                        if isinstance(correct_answers, list) 
                                        else correct_answers)
                        
                        prompt = evaluation_prompt_template.format(
                            question=question,
                            context=context,
                            correct_answers=formatted_answers,
                            model_answer=model_answer
                        )
                        
                        # Get evaluation result
                        response, yes_prob, no_prob, details = self.get_yes_no(prompt, eval_method, use_api_model)
                        score = 1.0 if details['predicted'] == 'yes' else 0.0
                        evaluation_result = {
                            'type': 'llm_evaluation',
                            'score': score,
                            **details
                        }
                    
                    total_score += score
                    
                    # Update item with evaluation results
                    if 'evaluation_results' not in item:
                        item['evaluation_results'] = {}
                    item['evaluation_results'][f'{eval_method}_results'] = evaluation_result
                    
                    # Clear variables that might exist
                    if 'prompt' in locals():
                        del prompt
                    if 'response' in locals():
                        del response
                    if 'yes_prob' in locals():
                        del yes_prob
                    if 'no_prob' in locals():
                        del no_prob
                    if 'details' in locals():
                        del details
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing item {idx}: {str(e)}")
                    print(f"Task: {task}")  # Add more context about the failing item
                    
                    # Create a failure evaluation result
                    evaluation_result = {
                        'type': 'error',
                        'score': 0.0,
                        'error_message': str(e)
                    }
                    
                    # Cleanup on error
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                finally:
                    # Always clean up, whether successful or not
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                batch_updated.append(item)
            
            # Extend the main list with batch results
            updated_data.extend(batch_updated)
            
            # Clear batch data and run garbage collection
            del batch_updated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Run garbage collection every 10 batches to clean up any cyclical references
            if (idx + 1) % 10 == 0:
                gc.collect()
                
            # Optional: Save intermediate results
            if (idx + 1) % 100 == 0:
                avg_score = total_score / len([d for d in updated_data if 'evaluation_summary' not in d])
                # print(f"\nIntermediate average score after {idx + 1} examples: {avg_score:.4f}")
        
        # Calculate final average score
        avg_score = total_score / len([d for d in updated_data if 'evaluation_summary' not in d])
        
        # Prepare summary statistics
        summary_stats = {
            'total_samples': len([d for d in updated_data if 'evaluation_summary' not in d]),
            'average_score': avg_score
        }
        
        return updated_data, summary_stats

def analyze_performance(output_dir: str, chunked_results: Dict[str, List[Dict]], full_context_results: List[Dict]) -> Dict:
    """Analyze and compare performance between early stopping and full_context generation."""
    print("\n=== Performance Analysis ===")
    
    # Load probe results
    with open(os.path.join(output_dir, "probe_results.json"), 'r') as f:
        probe_results = json.load(f)
    
    # Initialize dictionaries for different thresholds
    stats_by_threshold = {}
    accuracy_by_threshold = {}
    task_accuracies = {"full_context": {}, **{f"chunked_{t}": {} for t in [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]]}}
    
    full_context_eval_summary = next(r for r in full_context_results if "evaluation_summary" in r)["evaluation_summary"]
    full_context_data = [r for r in full_context_results if "evaluation_summary" not in r]
    
    # Calculate task accuracies for full_context mode
    method = config.EARLY_OUTPUT_CONFIG['eval_method']
    for task in set(item.get("task", "") for item in full_context_data):
        if task:
            task_data = [item for item in full_context_data if item.get("task") == task]
            scores = [item["evaluation_results"][f"{method}_results"]["score"] for item in task_data]
            task_accuracies["full_context"][task] = {
                "accuracy": float(np.mean(scores) * 100),
                "count": len(task_data)
            }
    
    # Process results for each threshold
    for mode, results in chunked_results.items():
        if mode == "full_context":
            continue
            
        eval_summary = next(r for r in results if "evaluation_summary" in r)["evaluation_summary"]
        data = [r for r in results if "evaluation_summary" not in r]
        
        threshold = int(mode.split("_")[1])
        threshold_str = f"chunked_{threshold}"
        
        # Calculate statistics for this threshold
        total_examples = len(data)
        early_stopped = sum(1 for r in data if r.get("early_stopped", False))
        early_stop_rate = early_stopped / total_examples * 100
        
        avg_chunks = np.mean([r.get("chunks_processed", 0) for r in data])
        avg_total_chunks = np.mean([r.get("total_chunks", 0) for r in data])
        chunk_reduction_factor = avg_total_chunks / avg_chunks if avg_chunks > 0 else 1
        
        # Store statistics
        stats_by_threshold[threshold_str] = {
            "total_examples": total_examples,
            "early_stopping_rate": early_stop_rate,
            "avg_chunks_processed": float(avg_chunks),
            "avg_total_chunks": float(avg_total_chunks),
            "context_reduction_factor": float(chunk_reduction_factor)
        }
        
        # Store accuracy information
        accuracy_by_threshold[threshold_str] = {
            "accuracy": float(eval_summary["method_scores"][method]["average_score"] * 100),
            "total_samples": eval_summary["method_scores"][method]["total_samples"]
        }
        
        # Calculate task accuracies
        for task in set(item.get("task", "") for item in data):
            if task:
                task_data = [item for item in data if item.get("task") == task]
                scores = [item["evaluation_results"][f"{method}_results"]["score"] for item in task_data]
                task_accuracies[threshold_str][task] = {
                    "accuracy": float(np.mean(scores) * 100),
                    "count": len(task_data)
                }
    
    # Print new focused comparison table with accuracy gain
    print("\nKey Metrics Comparison:")
    headers = ["Threshold", "Accuracy Gain", "Context Reduction"]
    full_context_accuracy = full_context_eval_summary['method_scores'][method]['average_score']*100
    rows = [["full_context", 
             "0.00%",
             "1.00x"]]
    
    for threshold in [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]]:
        threshold_str = f"chunked_{threshold}"
        accuracy_gain = accuracy_by_threshold[threshold_str]['accuracy'] - full_context_accuracy
        rows.append([
            f"{threshold}%",
            f"{accuracy_gain:+.2f}%",  # Using + sign to show gains/losses
            f"{stats_by_threshold[threshold_str]['context_reduction_factor']:.2f}x"
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Print detailed efficiency metrics
    print("\nDetailed Efficiency Metrics:")
    threshold = config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]
    threshold_str = f"chunked_{threshold}"
    efficiency_metrics = [
        ["Full Context Generation", ""],
        ["Context Reduction", "1.00x"],
        ["Early Stopping Results (Threshold {}%)".format(threshold), ""],
        ["Early Stopping Rate", f"{stats_by_threshold[threshold_str]['early_stopping_rate']:.2f}%"],
        ["Average Chunks Processed", f"{stats_by_threshold[threshold_str]['avg_chunks_processed']:.2f}/{stats_by_threshold[threshold_str]['avg_total_chunks']:.2f}"],
        ["Context Reduction Factor", f"{stats_by_threshold[threshold_str]['context_reduction_factor']:.2f}x"]
    ]
    print(tabulate(efficiency_metrics, headers=["Metric", "Value"], tablefmt="grid"))
       
    # Print task-specific accuracy breakdown
    print("\nTask-Specific Accuracy Breakdown:")
    task_headers = ["Task", "Count", "full_context"] + [f"T{t}%" for t in [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]]]
    task_rows = []
    
    all_tasks = set()
    for accuracies in task_accuracies.values():
        all_tasks.update(accuracies.keys())
    
    for task in sorted(all_tasks):
        row = [task]
        # Get count (should be same for all thresholds)
        count = task_accuracies["full_context"].get(task, {"count": 0})["count"]
        row.append(str(count))
        # Add full_context accuracy
        row.append(f"{task_accuracies['full_context'].get(task, {'accuracy': 0})['accuracy']:.2f}%")
        # Add accuracies for each threshold
        for threshold in [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]]:
            threshold_str = f"chunked_{threshold}"
            row.append(f"{task_accuracies[threshold_str].get(task, {'accuracy': 0})['accuracy']:.2f}%")
        task_rows.append(row)
    
    # Add average row
    avg_row = ["Average"]
    avg_row.append(str(sum(int(r[1]) for r in task_rows)))
    avg_row.append(f"{full_context_accuracy:.2f}%")
    for threshold in [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]]:
        threshold_str = f"chunked_{threshold}"
        avg_row.append(f"{accuracy_by_threshold[threshold_str]['accuracy']:.2f}%")
    task_rows.append(avg_row)
    
    print(tabulate(task_rows, headers=task_headers, tablefmt="grid"))
    
    # Prepare the complete analysis results
    analysis_results = {
        "statistics": {
            "by_threshold": stats_by_threshold,
        },
        "accuracy": {
            "overall": {
                "full_context_accuracy": float(full_context_eval_summary["method_scores"][method]["average_score"] * 100)
            },
            "by_threshold": accuracy_by_threshold,
            "by_task": task_accuracies
        },
        "probe_results": {
            "f1_before_tuning": float(probe_results["test_results"]["overall"]["f1"]),
        }
    }
    
    return analysis_results

def create_evaluator(model=None, tokenizer=None, model_name=None, use_api_model=False):
    """Create an evaluator instance with the appropriate configuration."""
    if model is not None and tokenizer is not None and not use_api_model:
        return Evaluator(model=model, tokenizer=tokenizer)
    elif not use_api_model and model is None and tokenizer is None:
        return Evaluator(model_name=model_name if model_name is not None else config.MODEL_CONFIG["model_name"])
    elif use_api_model:
        return Evaluator(use_api_model=use_api_model)
    else:
        raise ValueError("Invalid combination of parameters for evaluator creation")

def main(output_dir=None, model=None, tokenizer=None, model_name=None, thresholds=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Base directory containing the model results')
    parser.add_argument('--model_name', type=str, help='Model name to use for early stopping')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    args = parser.parse_args()

    # Use command line argument if provided, otherwise use function argument
    if output_dir is not None:
        args.output_dir = output_dir
    output_dir = args.output_dir if args.output_dir is not None else None
    
    if model_name is not None:
        args.model_name = model_name
    model_name = args.model_name if args.model_name is not None else config.MODEL_CONFIG["eval_model"]
    
    use_api_model = config.MODEL_CONFIG["eval_model"] == "api"

    # Keep the original default if neither is provided
    if output_dir is None:
        output_dir = config.PATH_CONFIG["output_dir"]
        
    print(f"\nBase directory: {output_dir}")
    results_dir = os.path.join(output_dir, "test_results")
    eval_method = config.EARLY_OUTPUT_CONFIG['eval_method']
    
    thresholds = [config.EARLY_OUTPUT_CONFIG["min_sufficient_confidence"]] 
    # Define thresholds and file patterns
    files = {
        'full_context': os.path.join(results_dir, "early_output_results_full_context.json"),
        **{f'chunked_{t}': os.path.join(results_dir, f"early_output_results_chunked_{t}.json") 
           for t in thresholds}
    }
    
    # Load and process all results
    all_results = {}
    for mode, file_path in files.items():
        try:
            with open(file_path, 'r') as f:
                all_results[mode] = json.load(f)
            print(f"Successfully loaded {mode} results from {file_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
            raise
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from {file_path}")
            raise
    
    if not all_results:
        print("No results files found or could be loaded. Exiting...")
        return
    
    # Create evaluator
    try:
        evaluator = create_evaluator(model, tokenizer, model_name, use_api_model)
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        return
    
    # Run evaluations for all modes
    for mode, results in all_results.items():
        print(f"\nEvaluating {mode} results...")
        try:
            updated_data, summary_stats = evaluator.evaluate_results(
                results, evaluation_prompt, eval_method=eval_method, use_api_model=use_api_model
            )
            all_results[mode] = updated_data
            
            # Update evaluation summary
            if not any('evaluation_summary' in item for item in updated_data):
                all_results[mode].append({
                    'evaluation_summary': {
                        'method_scores': {eval_method: summary_stats}
                    }
                })
        except Exception as e:
            print(f"Error evaluating {mode} results: {str(e)}")
            continue
    
    try:
        # Run performance analysis
        analysis_results = analyze_performance(output_dir, all_results, all_results['full_context'])
        
        # Save results
        for mode, results in all_results.items():
            with open(files[mode], 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save analysis results
        analysis_path = os.path.join(results_dir, "final_performance_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print("\nAll evaluations completed and results saved!")
        print(f"Analysis results saved to: {analysis_path}")
        
    except Exception as e:
        print(f"Error in analysis or saving results: {str(e)}")
    finally:
        if evaluator.should_cleanup:
            clear_gpu_memory()
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()