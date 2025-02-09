# probe_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from typing import Dict
import numpy as np
import argparse
from collections import defaultdict

def set_style():
    """Set consistent style parameters for visualizations"""
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        'font.size': 17,
        'axes.labelsize': 21,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 17
    })

def create_heatmap(accuracies: Dict[str, dict], save_path: str):
    """
    Create and save a heatmap visualization of probe accuracies across attention heads
    with enhanced color distinction.
    """
    set_style()
    
    layers = set()
    heads = set()
    
    for key, info in accuracies.items():
        layers.add(info['layer'])
        heads.add(info['head'])
    
    layers = sorted(list(layers))
    heads = sorted(list(heads))
    
    if not layers or not heads:
        raise ValueError("No valid layer or head information found in the f1 data")
    
    print(f"Found {len(layers)} layers and {len(heads)} heads per layer")
    
    acc_matrix = np.zeros((len(layers), len(heads)))
    
    for key, info in accuracies.items():
        layer_idx = layers.index(info['layer'])
        head_idx = heads.index(info['head'])
        acc_matrix[layer_idx, head_idx] = info['validation_f1']
    
    # Sort each row in descending order
    acc_matrix = np.sort(acc_matrix, axis=1)[:, ::-1]
    
    plt.figure(figsize=(min(2 * len(heads), 15), min(2 * len(layers), 10)))
    
    sns.heatmap(
        acc_matrix,
        cmap='YlGnBu',
        xticklabels='',
        yticklabels=range(1, len(layers) + 1),
        cbar_kws={'format': '%.2f'},
        center=None,   
        robust=True   
    )
    
    plt.xlabel('Head (Sorted)')
    plt.ylabel('Layer')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Successfully created and saved enhanced heatmap to: {save_path}")

def create_confidence_trends(confidence_data: Dict, save_path: str, num_chunks: int = 10):
    """Create and save confidence trends visualization."""
    set_style()
    plt.figure(figsize=(12, 8))
    
    # Process and organize data by task
    task_confidences = defaultdict(lambda: defaultdict(list))
    
    # Task name mapping
    task_name_mapping = {
        'code': 'Code Understanding',
        'multihop_kv': 'Key-Value Retrieval',
        'singlehop_qa': 'Single-Hop QA',
        'multihop_qa': 'Multi-Hop QA'
    }
    
    for query, measurements in confidence_data.items():
        sorted_measurements = sorted(measurements, key=lambda x: x['context_length'])
        if len(sorted_measurements) == num_chunks:
            original_task_name = measurements[0].get('task_name', measurements[0].get('task', 'unknown'))
            mapped_task_name = task_name_mapping.get(original_task_name, original_task_name)
            key = f"{mapped_task_name}_{len(task_confidences[mapped_task_name])}"
            task_confidences[mapped_task_name][key] = [m['confidence'] for m in sorted_measurements]
    
    chunks = range(num_chunks)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(task_confidences)))
    
    for (task_name, task_data), color in zip(task_confidences.items(), colors):
        task_scores = np.array(list(task_data.values()))
        if len(task_scores) > 0:
            task_mean = np.mean(task_scores, axis=0)
            task_25th = np.percentile(task_scores, 25, axis=0)
            task_75th = np.percentile(task_scores, 75, axis=0)
            
            plt.plot(chunks, task_mean, 
                    color=color, 
                    linewidth=2.5, 
                    label=task_name,
                    zorder=3)
            
            plt.fill_between(chunks, 
                           task_25th,
                           task_75th,
                           color=color, 
                           alpha=0.2,
                           zorder=2)
    
    plt.xlabel('Chunk')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(chunks, range(1, num_chunks + 1))
    plt.yticks()
    
    # Find the best location for the legend
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.legend(loc='best')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_distribution(confidence_data: Dict, save_path: str, num_chunks: int = 10):
    """Create and save confidence distribution visualization."""
    set_style()
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    chunk_scores = [[] for _ in range(num_chunks)]
    for query, measurements in confidence_data.items():
        sorted_measurements = sorted(measurements, key=lambda x: x['context_length'])
        if len(sorted_measurements) == num_chunks:
            for i, measurement in enumerate(sorted_measurements):
                chunk_scores[i].append(measurement['confidence'])
    
    colors = sns.color_palette("viridis", n_colors=num_chunks)
    bp = plt.boxplot(chunk_scores, 
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(alpha=0.8))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Chunk')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(range(1, num_chunks + 1), range(1, num_chunks + 1))
    plt.yticks()
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_results(accuracies: Dict[str, dict]):
    """Analyze and print insights about the results."""
    top_heads = sorted(
        accuracies.items(),
        key=lambda x: x[1]['validation_f1'],
        reverse=True
    )[:5]
    
    print("\nTop 5 attention heads for detecting information sufficiency:")
    for head_id, info in top_heads:
        print(f"Layer {info['layer']}, Head {info['head']}: {info['validation_f1']:.3f}")
    
    layer_stats = {}
    for info in accuracies.values():
        layer = info['layer']
        acc = info['validation_f1']
        if layer not in layer_stats:
            layer_stats[layer] = []
        layer_stats[layer].append(acc)
    
    print("\nLayer-wise performance:")
    for layer in sorted(layer_stats.keys()):
        mean_acc = np.mean(layer_stats[layer])
        max_acc = np.max(layer_stats[layer])
        print(f"Layer {layer}: Mean = {mean_acc:.3f}, Max = {max_acc:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize sufficiency probe results')
    parser.add_argument('--results_path', type=str, default='/usr/project/xtmp/rx55/projects/early-stop/results/short_version_pipeline_code_8d_1-18/01-18_03-10_Llama-3.2-1B-Instruct/probe_results.json')
    parser.add_argument('--confidence_data_path', type=str, default='/usr/project/xtmp/rx55/projects/early-stop/results/short_version_test/01-26_19-22_Llama-3.2-1B-Instruct/test_confidence_data.json')
    parser.add_argument('--output_dir', type=str, default='/usr/project/xtmp/rx55/projects/early-stop/results/short_version_test/01-26_16-51_Llama-3.2-1B-Instruct/visualization_test')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load and process probe results
        with open(args.results_path, 'r') as f:
            accuracies = json.load(f)["head_accuracies"]
            
        if not accuracies:
            raise ValueError("Empty F1 data loaded")
            
        # Create heatmap visualization
        vis_path = os.path.join(args.output_dir, "head_heatmap.png")
        create_heatmap(accuracies, vis_path)
        analyze_results(accuracies)
        
        # Load and process confidence data
        with open(args.confidence_data_path, 'r') as f:
            confidence_data = json.load(f)
        
        # Create confidence visualizations
        trends_path = os.path.join(args.output_dir, "confidence_trends.png")
        dist_path = os.path.join(args.output_dir, "confidence_distribution.png")
        
        create_confidence_trends(confidence_data, trends_path)
        create_confidence_distribution(confidence_data, dist_path)
        
    except Exception as e:
        logging.error(f"Error processing results: {str(e)}")
        raise

if __name__ == "__main__":
    main()