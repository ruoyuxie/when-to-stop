# final_classifier.py
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import logging
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import StratifiedKFold
from final_classifier_visualization import plot_auc_scores, plot_cv_results

class FinalClassifier:
    def __init__(self, config: Dict):
        """Initialize classifier configurations"""
        self.classifiers = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2,
                eval_metric='auc'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                class_weight='balanced',
                metric='auc',
                verbose=-1,
                objective='binary'
            ),
            'catboost': ctb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                border_count=128,
                bagging_temperature=1,
                random_strength=1,
                verbose=False,
                eval_metric='AUC'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features='sqrt'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                cache_size=2000
            ),
            'logistic': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=2000,
                class_weight='balanced'
            )
        }
        
        self.head_selection_probe = LogisticRegression(
            max_iter=2000,
            class_weight='balanced'
        )
        self.METRIC_FUNCTIONS = {'f1': f1_score, 'accuracy': accuracy_score, 
                               'precision': precision_score, 'recall': recall_score, 
                               'auc_roc': roc_auc_score}
        
        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.ensemble = None
        if config is not None and hasattr(config, 'CLASSIFIER_CONFIG'):
            self.threshold = config.CLASSIFIER_CONFIG.get("default_threshold", 0.5)
            self.n_cv_folds = config.CLASSIFIER_CONFIG.get("n_cv_folds", 5)
        else:
            self.threshold = 0.5
            self.n_cv_folds = 5
        
        self.config = config
   
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features using numpy operations"""
        if features.size == 0:
            return features
            
        # Handle missing values
        features = np.nan_to_num(features, nan=np.nanmean(features))
        
        # Handle outliers using IQR method
        Q1 = np.percentile(features, 25, axis=0)
        Q3 = np.percentile(features, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return np.clip(features, lower_bound, upper_bound)


    def cross_validate_classifiers(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform stratified cross-validation for all classifiers"""
        logging.info("Starting cross-validation process...")
        
        cv_scores = {name: {'cv_scores': [], 'mean_score': 0.0} 
                    for name in self.classifiers.keys()}
        
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logging.info(f"Processing fold {fold}/{self.n_cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Preprocess and scale the fold data
            X_train_fold = self.preprocess_features(X_train_fold)
            X_val_fold = self.preprocess_features(X_val_fold)
            
            X_train_scaled = self.scaler.fit_transform(X_train_fold)
            X_val_scaled = self.scaler.transform(X_val_fold)
            
            # Evaluate each classifier
            for name, clf in self.classifiers.items():
                try:
                    clf.fit(X_train_scaled, y_train_fold)
                    y_pred = clf.predict(X_val_scaled)
                    score = self.METRIC_FUNCTIONS['accuracy'](y_val_fold, y_pred)
                    cv_scores[name]['cv_scores'].append(score)
                except Exception as e:
                    logging.warning(f"Error in fold {fold} for classifier {name}: {str(e)}")
                    continue
        
        # Calculate mean scores and store results
        for name in cv_scores:
            if cv_scores[name]['cv_scores']:
                cv_scores[name]['mean_score'] = np.mean(cv_scores[name]['cv_scores'])
                cv_scores[name]['std_score'] = np.std(cv_scores[name]['cv_scores'])
                logging.info(f"{name}: Mean CV Score = {cv_scores[name]['mean_score']:.3f} "
                           f"(±{cv_scores[name]['std_score']:.3f})")
        
        return cv_scores

    def train_final_classifier(self, train_features: Dict, val_features: Dict, 
                            train_data: List, val_data: List, selected_heads: List) -> None:
        """Enhanced training pipeline with preprocessing, cross-validation and visualization"""
        # Extract features
        X_train = []
        X_val = []
        
        for _, _, _, _, _, features, _ in train_data:
            selected_features = []
            for head in selected_heads:
                head_key = f"layer_{head.layer}_head_{head.head}"
                if head_key in features:
                    selected_features.append(features[head_key])
            if selected_features:
                X_train.append(np.concatenate(selected_features))
                
        for _, _, _, _, _, features, _ in val_data:
            selected_features = []
            for head in selected_heads:
                head_key = f"layer_{head.layer}_head_{head.head}"
                if head_key in features:
                    selected_features.append(features[head_key])
            if selected_features:
                X_val.append(np.concatenate(selected_features))
        
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        
        # Get labels
        y_train = np.array([x[1] for x in train_data[:len(X_train)]])
        y_val = np.array([x[1] for x in val_data[:len(X_val)]])
        
        logging.info("Starting training pipeline...")
        
        # Step 1: Preprocess features
        X_train_processed = self.preprocess_features(X_train)
        X_val_processed = self.preprocess_features(X_val)
        
        # Step 2: Perform cross-validation
        cv_results = self.cross_validate_classifiers(X_train_processed, y_train)
        
        # Collect AUC scores for each classifier
        auc_scores = {}
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        X_val_scaled = self.scaler.transform(X_val_processed)
        
        for name, clf in self.classifiers.items():
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_val_scaled)[:, 1]
            auc_scores[name] = roc_auc_score(y_val, y_prob)
        
        # Plot CV results and save in the output directory
        plot_cv_results(cv_results, os.path.join(self.config.PATH_CONFIG["model_output_dir"], 'classifier_cv_results.png'))
        
        # Step 3: Select top performing classifiers
        top_classifiers = sorted(
            [(name, scores['mean_score']) for name, scores in cv_results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:self.config.CLASSIFIER_CONFIG["num_top_classifiers"]]
        
        logging.info("Selected top classifiers: %s", 
                    ", ".join([f"{name} ({score:.3f})" for name, score in top_classifiers]))
        
        # Step 4: Create weighted ensemble
        weights = [1.0 + i/10 for i in range(len(top_classifiers)-1, -1, -1)]
        
        self.ensemble = VotingClassifier(
            estimators=[(name, self.classifiers[name]) for name, _ in top_classifiers],
            voting='soft',
            weights=weights
        )
        
        # Step 5: Final training on preprocessed data
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Step 6: Evaluate and visualize performance
        X_val_scaled = self.scaler.transform(X_val_processed)
        val_pred = self.ensemble.predict(X_val_scaled)
        val_proba = self.ensemble.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        val_score = accuracy_score(y_val, val_pred)
        auc_score = roc_auc_score(y_val, val_proba)
        
        # Plot ROC curve
        plot_auc_scores(auc_scores, os.path.join(self.config.PATH_CONFIG["model_output_dir"], 'classifier_auc_scores.png'))
        
        # Compare individual classifier performance on validation set
        classifier_metrics = {}
        for name, clf in self.classifiers.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_val_scaled)
            y_prob = clf.predict_proba(X_val_scaled)[:, 1]
            
            classifier_metrics[name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_prob)
            }
        
        # Add ensemble metrics
        classifier_metrics['ensemble'] = {
            'accuracy': val_score,
            'auc': auc_score
        }
        
        logging.info(f"Final validation accuracy: {val_score:.3f}")
        logging.info(f"Final validation AUC: {auc_score:.3f}")
        
        # Store training metrics
        self.training_metrics = {
            'ensemble_members': [name for name, _ in top_classifiers],
            'cross_validation_results': cv_results,
            "auc_scores": auc_scores,
            'final_validation_score': val_score,
            'final_auc_score': auc_score,
            'classifier_metrics': classifier_metrics
        } 
                
    def predict(self, features: np.ndarray) -> bool:
        """Make binary prediction with preprocessing"""
        if not self.ensemble:
            raise ValueError("Classifier not trained yet")
        
        # Preprocess and scale input features
        features_processed = self.preprocess_features(features.reshape(1, -1))
        features_scaled = self.scaler.transform(features_processed)
        return bool(self.ensemble.predict(features_scaled)[0])

    def predict_proba(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict probability scores with preprocessing"""
        if not self.ensemble:
            raise ValueError("Classifier not trained yet")
        
        # Preprocess and scale input features
        features_processed = self.preprocess_features(features.reshape(1, -1))
        features_scaled = self.scaler.transform(features_processed)
        probs = self.ensemble.predict_proba(features_scaled)[0]
        return tuple(probs)

    def _get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from stacked ensemble"""
        # Get level 1 predictions
        level1_preds = np.zeros((X.shape[0], len(self.classifiers)))
        for i, (name, model) in enumerate(self.level1_models):
            base_name = name.rsplit('_', 1)[0]  # Remove seed suffix
            idx = list(self.classifiers.keys()).index(base_name)
            level1_preds[:, idx] += model.predict_proba(X)[:, 1] / 5  # Average over seeds

        # Get final predictions from meta-learner
        return self.meta_learner.predict_proba(level1_preds)[:, 1]
 
    def evaluate_single_head(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate single head performance"""
        self.head_selection_probe.fit(X_train, y_train)
        y_pred = self.head_selection_probe.predict(X_val)
        return f1_score(y_val, y_pred)

    def _save_comparison_plots(self, results: Dict, output_dir: str) -> None:
        """Save comparison plots for visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        metrics = list(next(iter(results.values())).keys())
        
        for classifier_name, metric_values in results.items():
            for metric in metrics:
                plot_data.append({
                    'Classifier': classifier_name,
                    'Metric': metric.upper(),
                    'Score': metric_values[metric] * 100
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df, x='Classifier', y='Score', hue='Metric')
        plt.xticks(rotation=45)
        plt.title('Classifier Performance Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classifier_comparison_metrics.png'))
        plt.close()

    def state_dict(self):
        """Return a dictionary containing the classifier state"""
        if not hasattr(self, 'ensemble'):
            raise ValueError("Classifier not trained yet")
                
        return {
            'ensemble': pickle.dumps(self.ensemble),
            'scaler': pickle.dumps(self.scaler),
            'threshold': self.threshold,
            'training_metrics': self.training_metrics
        }

    def load_state_dict(self, state_dict):
        """Load classifier state from a dictionary"""
        try:
            # Load ensemble
            self.ensemble = pickle.loads(state_dict['ensemble'])
            
            # Load scaler and other attributes
            self.scaler = pickle.loads(state_dict['scaler'])
            self.threshold = state_dict['threshold']
            self.training_metrics = state_dict.get('training_metrics', {})
            
        except Exception as e:
            raise ValueError(f"Error loading classifier state: {str(e)}")
