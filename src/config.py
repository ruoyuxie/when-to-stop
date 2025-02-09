# config.py:

# Input data and output directories
PATH_CONFIG = {
    "data_dir": "<path_to_data_dir>",  # Data directory - will be overridden by command line arguments if provided
    "output_dir": "<path_to_output_dir>",  # Output directory - will be overridden by command line arguments if provided
}

# Main model configuration
MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",  # Model name - will be overridden by command line arguments if provided
    "trust_remote_code": True,
    "torch_dtype": "float16",
    "device_map": "auto",
    "eval_model": "api", # 'api' or any models, e.g. "meta-llama/Llama-3.3-70B-Instruct". For API, make sure you have OpenAI API key set as environment variable `OPENAI_API_KEY`. Note that using smaller models like 1B, 3B will be faster but less accurate in evaluation.
}

# Probe training and evaluation configuration
TRAINING_CONFIG = {
    "samples_per_task": 600, # Number of samples per task, 600 for the full dataset, reduce it for faster testing
    "num_heads": 5,
    "batch_size": 1, # increase it for faster training
    "dev_ratio": 0.8,
    "seed": 42,
    "train_tasks": {"multihop_qa","singlehop_qa","multihop_kv","code"}, # Specify the tasks to train on
    "eval_tasks": {"multihop_qa","singlehop_qa","code","multihop_kv"}, # Specify the tasks to evaluate on
}

# Final classifier configuration
CLASSIFIER_CONFIG = {
    "target_precision": 0.9,
    "default_threshold": 0.5,
    "num_top_classifiers": 4,
    "visualize": True,
    "metrics": ["accuracy", "f1", "precision", "recall", "auc_roc"]
}

# Evaluation configuration for the final classifier
EVALUATION_CONFIG = {
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc"
    ],
    "target_precisions": [0.90, 0.95, 0.98]
}

# Chunking strategy for the input data
DATA_CONFIG = {
    "chunking": {
        "strategy": "percentage",  # Options: "percentage", "sentence", "token"
        "percentage": {
            "percentage_num": 10, # Percentage of the input data to chunk 
        }
    }
}

# Evaluate the probe on the test set for early stopping
EARLY_OUTPUT_CONFIG = {
    "min_sufficient_confidence": 95, # Minimum confidence to consider context sufficient in percentage
    "chunking_strategy": DATA_CONFIG["chunking"]["strategy"],
    "percentage_num": DATA_CONFIG["chunking"]["percentage"]["percentage_num"],
    "number_of_evaluation_examples": 1000,  # Number of examples to evaluate
    "batch_size": 1,
    "eval_method": "match", # match or prob; match: excat match, prob: use probability

}

# Generation configuration
GENERATION_CONFIG = {
    "temperature": 0.000001,
    "max_new_tokens": 128,
    "do_sample": True,
}

