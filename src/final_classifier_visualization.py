# final_classifier_visualization.py:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from typing import Dict

def plot_auc_scores(auc_scores: Dict, output_path: str) -> None:
    """Plot AUC scores for all classifiers."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        'font.size': 17,
        'axes.labelsize': 21,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 17,
        'axes.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    classifiers = list(auc_scores.keys())
    auc_values = list(auc_scores.values())
    
    bars = ax.bar(range(len(classifiers)), auc_values, 
                  color=sns.color_palette("husl", len(classifiers)))
    
    ax.set_ylabel('AUC Score', labelpad=15)
    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels([clf.replace('_', ' ').title() for clf in classifiers], 
                       rotation=30, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_results(cv_results: Dict, output_path: str) -> None:
    """Plot cross-validation results with error bars."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({
        'font.size': 17,
        'axes.labelsize': 21,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 17,
        'axes.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    classifiers = list(cv_results.keys())
    mean_scores = [cv_results[clf]['mean_score'] for clf in classifiers]
    std_scores = [cv_results[clf]['std_score'] for clf in classifiers]
    
    # Create bar plot
    bars = ax.bar(range(len(classifiers)), mean_scores, yerr=std_scores, 
                capsize=5, color=sns.color_palette("husl", len(classifiers)))
    
    # Customize plot
    ax.set_ylabel('Mean CV Score', labelpad=15)
    
    # Set x-ticks and labels
    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels([clf.replace('_', ' ').title() for clf in classifiers], 
                    rotation=30, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add grid and adjust layout
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

