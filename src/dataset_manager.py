# dataset_manager.py

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import json
import os
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import save_results
from chunking_manager import ChunkingManager
from activation_manager import ContextActivations
import torch

@dataclass
class TaskSplits:
    """Store train, validation, and test data for a specific task"""
    task_name: str
    # Tuple structure: (query, is_sufficient, chunk_text, answers, task_name, chunk_features, gold_percentage)
    train_data: List[Tuple[str, bool, str, List[str], str, Dict[str, np.ndarray], float]]
    val_data: List[Tuple[str, bool, str, List[str], str, Dict[str, np.ndarray], float]]
    test_data: List[Tuple[str, bool, str, List[str], str, Dict[str, np.ndarray], float]]

class DatasetManager:
    def __init__(self, data_dir: str, probe = None):
        """Initialize DatasetManager with data directory and probe instance"""
        self.data_dir = data_dir
        self.probe = probe
        self.task_splits: Dict[str, TaskSplits] = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.task_names = None

        # Initialize chunking manager
        self.chunking_manager = ChunkingManager(
            tokenizer=self.probe.tokenizer,
            activation_manager=self.probe.activation_manager
        )

    def _create_chunks_from_examples(
        self, 
        examples: List[Dict], 
        task_name: str,
        batch_size: int = 4
    ) -> List[Tuple[str, str, Dict[str, np.ndarray], bool, str]]:
        """Create chunks from examples with efficient batch processing"""
        chunking_config = self.probe.config.DATA_CONFIG["chunking"]
        strategy = chunking_config["strategy"]
        chunked_data = []
        
        # Process examples in batches
        for i in tqdm(range(0, len(examples), batch_size), desc=f"Processing {task_name} data batches"):
            batch = examples[i:min(i + batch_size, len(examples))]
            
            # Prepare batch data
            batch_queries = [entry['input'] for entry in batch]
            batch_contexts = [entry['context'] for entry in batch]
            batch_answers = [entry.get('answers', []) for entry in batch]
            batch_gold_info = [entry.get('gold_information', []) for entry in batch]
            
            # Process each example in the batch
            batch_results = []
            for idx in range(len(batch)):
                # Get full context activations for current example
                context_activations = self.probe.activation_manager.compute_activations(
                    query=batch_queries[idx],
                    context=batch_contexts[idx],
                    gold_information=batch_gold_info[idx]
                )
                
                # Create chunks using chunking manager
                chunk_boundaries, _ = self.chunking_manager.create_chunks(
                    context_activations=context_activations,
                    strategy=strategy,
                    task_name=task_name,
                    num_chunks=chunking_config["percentage"]["percentage_num"] if strategy == "percentage" else None
                )
                
                chunks = self.chunking_manager._process_chunks(context_activations, chunk_boundaries)

                # Process each chunk in the current example
                for _, chunk_end, chunk_features, chunk_text in chunks:
                    is_sufficient = self._check_chunk_sufficiency(
                        chunk_end,
                        context_activations
                    )
                    batch_results.append((
                        batch_queries[idx],
                        is_sufficient,
                        chunk_text,
                        batch_answers[idx],
                        task_name,
                        chunk_features,
                        context_activations.gold_location.percentage if context_activations.gold_location else 0.0
                    ))

                del context_activations, chunk_boundaries, chunks
                torch.cuda.empty_cache()

            # Extend chunked_data with batch results
            chunked_data.extend(batch_results)
            del batch_results
            torch.cuda.empty_cache()
        
        return chunked_data

    def _process_batch_chunks(
        self,
        context_activations_batch: List[ContextActivations],
        chunk_boundaries_batch: List[List[int]],
        batch_queries: List[str],
        batch_answers: List[List[str]],
        task_name: str
        ) -> List[Tuple[str, bool, str, List[str], str, Dict[str, np.ndarray], float]]:
        """Helper method to process chunks for a batch of examples"""
        batch_results = []
        
        for idx, (context_activations, chunk_boundaries) in enumerate(zip(context_activations_batch, chunk_boundaries_batch)):
            chunks = self.chunking_manager._process_chunks(context_activations, chunk_boundaries)
            
            for _, chunk_end, chunk_features, chunk_text in chunks:
                is_sufficient = self._check_chunk_sufficiency(
                    chunk_end,
                    context_activations
                )
                batch_results.append((
                    batch_queries[idx],
                    is_sufficient,
                    chunk_text,
                    batch_answers[idx],
                    task_name,
                    chunk_features,
                    context_activations.gold_location.percentage if context_activations.gold_location else 0.0
                ))
                
        return batch_results

    def prepare_splits(
        self,
        model_output_dir: str,
        train_tasks: Set[str] = None,
        eval_tasks: Set[str] = None,
        development_ratio: float = 0.8,
        samples_per_task: int = 100,
        batch_size: int = 4 
    ) -> Tuple[List, List, List]:
        """Prepare train, validation, and test splits with batch processing"""
        print("\nChunking and getting activations...")
        original_test_data = []
        original_train_data = []
        original_val_data = []
        self.task_names = sorted(list(train_tasks & eval_tasks))
        
        for task_name in self.task_names:
            json_path = os.path.join(self.data_dir, f"{task_name}.json")
            if not os.path.exists(json_path):
                print(f"Warning: Task file not found: {json_path}")
                continue
            torch.cuda.empty_cache()

            print(f"\nProcessing task: {task_name}")
            
            raw_examples = self._load_raw_examples(json_path, samples_per_task)
            if not raw_examples:
                continue
                
            dev_examples, test_examples = self._split_examples(raw_examples, development_ratio)
            train_examples, val_examples = self._split_examples(dev_examples, 0.8)
            
            # Process each split with batching
            train_data = self._create_chunks_from_examples(train_examples, task_name, batch_size)
            val_data = self._create_chunks_from_examples(val_examples, task_name, batch_size)
            test_data = self._create_chunks_from_examples(test_examples, task_name, batch_size)
            
            # Update original data
            for example in test_examples:
                example["task"] = task_name
            original_test_data.extend(test_examples)
            for example in train_examples:
                example["task"] = task_name
            original_train_data.extend(train_examples)
            for example in val_examples:
                example["task"] = task_name
            original_val_data.extend(val_examples)
            
            self._remove_duplicates(train_data, val_data, test_data)
            
            self.task_splits[task_name] = TaskSplits(
                task_name=task_name,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )
    
        self._combine_randomize_keep_split_balanced()
        
        if train_tasks:
            self.train_data = [x for x in self.train_data if x[4] in train_tasks]
            self.val_data = [x for x in self.val_data if x[4] in train_tasks]
            original_train_data = [x for x in original_train_data if x["task"] in train_tasks]
            original_val_data = [x for x in original_val_data if x["task"] in train_tasks]
        if eval_tasks:
            self.test_data = [x for x in self.test_data if x[4] in eval_tasks]
            original_test_data = [x for x in original_test_data if x["task"] in eval_tasks]
        
        statistics = self._print_split_statistics()
        data_output_dir = os.path.join(model_output_dir, "data")
        self.save_original_data(data_output_dir, "train.json", original_train_data)
        self.save_original_data(data_output_dir, "val.json", original_val_data)
        self.save_original_data(data_output_dir, "test.json", original_test_data)
        
        # Save chunked data
        self.save_chunked_data(data_output_dir, original_train_data, self.train_data, "train")
        self.save_chunked_data(data_output_dir, original_val_data, self.val_data, "val")
        self.save_chunked_data(data_output_dir, original_test_data, self.test_data, "test")
        
        return self.train_data, self.val_data, self.test_data, statistics



    def save_chunked_data(self, data_output_dir: str, original_data: List[Dict], chunked_data: List[Tuple], split_name: str):

        # Create mapping from query to gold information in original data
        query_to_gold = {
            example['input']: example.get('gold_information', [])
            for example in original_data
        }
        
        # Process chunked data into new format
        processed_chunks = []
        for chunk in chunked_data:
            query, is_sufficient, chunk_text, answers, task, _, gold_percentage = chunk
            
            processed_chunk = {
                "query": query,
                "context": chunk_text,
                "answers": answers,
                "label": is_sufficient,
                "task": task,
                "gold_information": query_to_gold.get(query, []),
                "gold_location": gold_percentage
            }
            processed_chunks.append(processed_chunk)
        
        # Save processed chunks
        output_path = os.path.join(data_output_dir, f"chunked_{split_name}.json")
        save_results(processed_chunks, data_output_dir, f"chunked_{split_name}.json")
        print(f"Saved chunked data to: {output_path}")
        
    def save_original_data(self, model_output_dir: str, filename: str, data: List[Dict]):
        """Save original data to file"""
        save_results(data, model_output_dir, filename)
        print(f"Saved original data to: {model_output_dir}/{filename}")
 
    def _check_chunk_sufficiency(
        self,
        chunk_length: int,
        context_activations: 'ContextActivations'
    ) -> bool:
        if not context_activations.gold_location:
            print("No gold location found")
            return False
        # Check if we've reached the end of gold information
        return chunk_length >= context_activations.gold_location.end_idx
 
    def _remove_duplicates(self, train_data, val_data, test_data):
        """Remove duplicate examples between splits based on query and chunk_text"""
        train_keys = {(x[0], x[2]) for x in train_data}  # (query, chunk_text)
        val_data[:] = [x for x in val_data if (x[0], x[2]) not in train_keys]
        
        combined_keys = train_keys | {(x[0], x[2]) for x in val_data}
        test_data[:] = [x for x in test_data if (x[0], x[2]) not in combined_keys]

    def _print_split_statistics(self):
        """Print and return detailed statistics about the dataset splits"""
        print("\nDataset Split Statistics:")
        
        statistics = {}  # Store all statistics here
        
        splits = {
            "Training": self.train_data,
            "Validation": self.val_data,
            "Test": self.test_data
        }
        
        for split_name, split_data in splits.items():
            if not split_data:
                print(f"\n{split_name} set: Empty")
                continue
                
            total_examples = len(split_data)
            positive_examples = sum(1 for x in split_data if x[1])
            
            print(f"\n{split_name} set:")
            print(f"Total examples: {total_examples}")
            print(f"Positive examples: {positive_examples} ({(positive_examples/total_examples)*100:.1f}%)")
            print(f"Negative examples: {total_examples-positive_examples} ({((total_examples-positive_examples)/total_examples)*100:.1f}%)")
            
            # Store statistics
            statistics[split_name.lower()] = {
                "total": total_examples,
                "positive": positive_examples,
                "negative": total_examples - positive_examples
            }
            
            # Calculate per-task statistics
            task_stats = defaultdict(lambda: {"total": 0, "positive": 0})
            for example in split_data:
                task_name = example[4]
                task_stats[task_name]["total"] += 1
                if example[1]:
                    task_stats[task_name]["positive"] += 1
            
            print("\nPer-task breakdown:")
            statistics[f"{split_name.lower()}_tasks"] = {}
            
            for task_name, stats in task_stats.items():
                total = stats["total"]
                positive = stats["positive"]
                print(f"{task_name}:")
                print(f"  Total: {total}")
                print(f"  Positive: {positive} ({(positive/total)*100:.1f}%)")
                print(f"  Negative: {total-positive} ({((total-positive)/total)*100:.1f}%)")
                
                statistics[f"{split_name.lower()}_tasks"][task_name] = {
                    "total": total,
                    "positive": positive,
                    "negative": total - positive
                }
        
        return statistics

    def _calculate_chunk_stats(self, data: List[Tuple]) -> Dict[str, float]:
        """Calculate statistics for a given set of chunks"""
        if not data:
            return {
                "avg_tokens": 0,
                "std_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        # Updated to use chunk_text at index 2
        token_lengths = []
        for example in data:
            chunk_text = example[2]
            tokens = self.probe.tokenizer.encode(chunk_text, add_special_tokens=False)
            token_lengths.append(len(tokens))
            
        token_lengths = np.array(token_lengths)
        
        return {
            "avg_tokens": float(np.mean(token_lengths)),
            "std_tokens": float(np.std(token_lengths)),
            "min_tokens": float(np.min(token_lengths)),
            "max_tokens": float(np.max(token_lengths))
        }

    def get_task_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of examples across tasks and splits"""
        distribution = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int)
        }
        
        # Updated to use task_name at index 4
        for example in self.train_data:
            distribution["train"][example[4]] += 1
        for example in self.val_data:
            distribution["val"][example[4]] += 1
        for example in self.test_data:
            distribution["test"][example[4]] += 1
        
        return dict(distribution)

    def _combine_randomize_keep_split_balanced(self):
        """Combine splits from all tasks with balanced representation"""
        if not self.task_splits:
            raise ValueError("No task splits available")
            
        min_train = min(len(splits.train_data) for splits in self.task_splits.values())
        min_val = min(len(splits.val_data) for splits in self.task_splits.values())
        min_test = min(len(splits.test_data) for splits in self.task_splits.values())
        
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        for splits in self.task_splits.values():
            self.train_data.extend(random.sample(splits.train_data, min_train))
            self.val_data.extend(random.sample(splits.val_data, min_val))
            self.test_data.extend(splits.test_data[:min_test])
        
        random.shuffle(self.train_data)
        random.shuffle(self.val_data)

    def _load_raw_examples(self, json_path: str, n_samples: int) -> List[Dict]:
        """Load and sample raw examples from JSON file"""
        try:
            with open(json_path, 'r') as f:
                examples = json.load(f)
            
            if len(examples) > n_samples:
                return random.sample(examples, n_samples)
            return examples
            
        except Exception as e:
            print(f"Error loading {json_path}: {str(e)}")
            return []

    def _split_examples(
        self,
        examples: List[Dict],
        split_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split examples into two parts based on ratio"""
        split_point = int(len(examples) * split_ratio)
        random.shuffle(examples)
        return examples[:split_point], examples[split_point:]

    def get_chunk_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics about chunk sizes across splits"""
        stats = {
            "train": self._calculate_chunk_stats(self.train_data),
            "val": self._calculate_chunk_stats(self.val_data),
            "test": self._calculate_chunk_stats(self.test_data)
        }
        return stats