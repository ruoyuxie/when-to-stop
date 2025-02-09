# evaluation.py

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from collections import defaultdict
from tqdm import tqdm
import json
import os

class SufficiencyEvaluator:
    def __init__(self, config: Dict):
        """Initialize evaluator"""
        self.results = {}
        self.threshold = config.CLASSIFIER_CONFIG["default_threshold"]
        self.target_precision = config.CLASSIFIER_CONFIG["target_precision"]
 
    def evaluate_sufficiency_performance(self, probe, test_data: List[Tuple]) -> Dict:
        """Evaluate using updated data structure"""
        results = {
            "overall": {
                "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "confidence_stats": {"mean": 0.0, "std": 0.0}
            },
            "per_task": {},
            "confidence_data": {}
        }
        
        task_predictions = defaultdict(list)
        task_confidences = defaultdict(list)
        task_labels = defaultdict(list)
        confidence_by_query = defaultdict(list)
        
        for example in tqdm(test_data, desc="Evaluating"):
            query, is_sufficient, chunk_text, _, task_name, features,_ = example
            try:
                pred, conf = probe.evaluate_sufficiency(features)
                
                # Track metrics
                task_predictions[task_name].append(pred)
                task_confidences[task_name].append(conf)
                task_labels[task_name].append(is_sufficient)
                
                confidence_by_query[query].append({
                    'confidence': conf,
                    'context_length': len(chunk_text),
                    'is_sufficient': is_sufficient,
                    'task_name': task_name
                })
                
            except Exception as e:
                print(f"Error evaluating example: {str(e)}")
                continue
        
        # Calculate per-task metrics
        for task_name in task_predictions:
            task_preds = np.array(task_predictions[task_name])
            task_confs = np.array(task_confidences[task_name])
            task_labs = np.array(task_labels[task_name])
            
            if len(task_preds) > 0:
                results["per_task"][task_name] = {
                    "f1": round(f1_score(task_labs, task_preds), 3),
                    "accuracy": round(np.mean(task_preds == task_labs), 3),
                    "precision": round(precision_score(task_labs, task_preds), 3),
                    "recall": round(recall_score(task_labs, task_preds), 3),
                    "confidence_stats": {
                        "mean": round(np.mean(task_confs), 3),
                        "std": round(np.std(task_confs), 3)
                    },
                    "num_examples": len(task_preds)
                }
        
        # Calculate overall metrics
        all_preds = []
        all_confs = []
        all_labels = []
        for task in task_predictions:
            all_preds.extend(task_predictions[task])
            all_confs.extend(task_confidences[task])
            all_labels.extend(task_labels[task])
        
        if all_preds:
            results["overall"].update({
                "f1": round(f1_score(all_labels, all_preds), 3),
                "accuracy": round(np.mean(np.array(all_preds) == np.array(all_labels)), 3),
                "precision": round(precision_score(all_labels, all_preds), 3),
                "recall": round(recall_score(all_labels, all_preds), 3),
                "confidence_stats": {
                    "mean": round(np.mean(all_confs), 3),
                    "std": round(np.std(all_confs), 3)
                }
            })
        
        self.results = results
        return results, confidence_by_query
    

    def _find_optimal_threshold(self, precisions: np.ndarray, recalls: np.ndarray,
                              thresholds: np.ndarray, target_precision: float) -> float:
        """Find threshold that achieves target precision while maximizing recall"""
        max_recall = 0
        optimal_threshold = 1.0
        
        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= target_precision and r > max_recall:
                max_recall = r
                optimal_threshold = t
        
        return optimal_threshold

    def tune_thresholds(self, probe, val_data: List[Tuple]) -> Dict:
        """Tune classification thresholds using precomputed features"""
        print("Tuning classification thresholds...")
        
        y_true = []
        y_scores = []
        
        # Collect predictions
        for _, is_sufficient, _, _, _, features,_ in tqdm(val_data, desc="Collecting predictions"):
            try:
                selected_features = []
                for head in probe.selected_heads:
                    head_key = f"layer_{head.layer}_head_{head.head}"
                    if head_key in features:
                        selected_features.append(features[head_key])
                        
                if not selected_features:
                    raise ValueError("No valid features for threshold tuning")
                
                # Aggregate features
                X_val = np.concatenate(selected_features)
                pred = probe.classifier.predict_proba(X_val) 
                prob_positive = pred[1]
                y_scores.append(prob_positive)
                y_true.append(is_sufficient)
            except Exception as e:
                print(f"Error during threshold tuning: {str(e)}")
                continue
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Find threshold maximizing F1
        thresholds = np.linspace(0, 1, 200)
        best_f1 = 0
        best_f1_threshold = 0.5
        f1_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = threshold
                f1_metrics = {
                    "f1": round(f1, 3),
                    "threshold": round(threshold, 3),
                    "precision": round(precision_score(y_true, y_pred), 3),
                    "recall": round(recall_score(y_true, y_pred), 3)
                }
        
        # Find thresholds for target precisions
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
        precision_metrics = {}
        
        for target_precision in [0.90, 0.95, 0.98]:
            optimal_threshold = self._find_optimal_threshold(
                precisions, recalls, pr_thresholds, target_precision
            )
            
            y_pred = (y_scores >= optimal_threshold).astype(int)
            precision_metrics[f"p{int(target_precision * 100)}"] = {
                "f1": round(f1_score(y_true, y_pred), 3),
                "threshold": round(optimal_threshold, 3),
                "precision": round(precision_score(y_true, y_pred), 3),
                "recall": round(recall_score(y_true, y_pred), 3),
            }
        
        tuning_results = {
            "optimal_f1": f1_metrics,
            "precision_targeted": precision_metrics
        }
        
        # Update default threshold
        self.threshold = best_f1_threshold
        
        return tuning_results