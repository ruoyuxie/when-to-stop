# train_probe.py

import argparse
import time
import os
from datetime import datetime
from probe import SufficiencyProbe
from dataset_manager import DatasetManager
from utils import setup_random_seeds, load_config, save_results, get_output_dir
import config
import json
import early_stopping

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate sufficiency probe")
    parser.add_argument("--model_name", type=str,
                      help="Model name or path")
    parser.add_argument("--output_dir", type=str,
                      help="Output directory")
    parser.add_argument("--data_dir", type=str,
                        help="Data directory")
    return parser.parse_args()

def update_config(args):
    # Override with command line arguments if provided
    if args.data_dir:
        config.PATH_CONFIG["data_dir"] = args.data_dir
    if args.model_name:
        config.MODEL_CONFIG["model_name"] = args.model_name
    if args.output_dir:
        config.PATH_CONFIG["output_dir"] = args.output_dir
    
def main():
    args = parse_args()
    
    # Update config based on arguments
    update_config(args)
    
    setup_random_seeds(config.TRAINING_CONFIG["seed"])
    
    model_output_dir = get_output_dir(config) if args.output_dir is None else args.output_dir
    config.PATH_CONFIG["model_output_dir"]= model_output_dir

    print(f"\nData directory: {config.PATH_CONFIG['data_dir']}\n")
    print(f"Output directory: {model_output_dir}\n")

    probe = SufficiencyProbe(config)

    start_time = time.time()
    
    print("Starting probe training and evaluation")

    dataset_manager = DatasetManager(config.PATH_CONFIG["data_dir"],probe=probe)
    
    # Prepare data splits using config
    train_data, val_data, test_data, statistics = dataset_manager.prepare_splits(
        model_output_dir,
        train_tasks=config.TRAINING_CONFIG["train_tasks"],
        eval_tasks=config.TRAINING_CONFIG["eval_tasks"],
        development_ratio=config.TRAINING_CONFIG["dev_ratio"],
        batch_size=config.TRAINING_CONFIG["batch_size"],
        samples_per_task=config.TRAINING_CONFIG["samples_per_task"], 
    )

    # Train and evaluate
    print("\nProbing...")
    probe.train_and_select_heads(
        train_data, 
        val_data, 
    )
        
    # Save probe and test dataaset
    probe_save_dir = os.path.join(model_output_dir, "probe")
    probe.save(probe_save_dir)
    print(f"Saved probe to: {probe_save_dir}")

    print("\nEvaluating final classifier...")
    test_results, _ = probe.evaluator.evaluate_sufficiency_performance(probe, test_data)
    
    # print the results
    print("\nTest results:")
    print(json.dumps(test_results, indent=2))

    # Save all results
    final_results = {
        "execution_time_seconds": time.time() - start_time,
        "samples_per_task": config.TRAINING_CONFIG["samples_per_task"],  
        "num_heads": config.TRAINING_CONFIG["num_heads"],
        "batch_size": config.TRAINING_CONFIG["batch_size"],
        "test_results": test_results,
        "data_statistics": statistics,
        "classifier_comparison": probe.classifier.training_metrics,
        "head_accuracies": probe.all_head_accuracies,
    }
        
    save_results(final_results, model_output_dir, "probe_results.json")
    print(f"Saved all results to: {model_output_dir}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    del train_data, val_data, test_data, dataset_manager, probe, final_results, test_results, statistics

    # Run and evaluate with early stopping
    print("\nRunning early stopping...\n")
    early_stopping.main(output_dir=model_output_dir, model_name=config.MODEL_CONFIG["model_name"])

if __name__ == "__main__":
    main()
    
