"""
Emotion Analysis Module
Wraps the emotion-based sentiment analysis ensemble
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Use local emotion prediction module
from .simple_predict import EmotionPredictor

# Global emotion configuration
# NOTE: These specific labels only apply because we fine-tuned all three models 
# (GPT-2, RoBERTa, DeBERTa) on these exact emotion categories.
# If using different pre-trained models, these labels would need to be updated.
EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Wrapper for the emotion analysis ensemble
    """
    
    def __init__(self, viz_dir=None, 
                 checkpoint_dir=None,
                 device=None):
        self.viz_dir = viz_dir or Path("outputs/visualizations/emotions")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            # Use the new models directory
            self.checkpoint_dir = Path(__file__).parent.parent.parent / "models"
        
        # Device setup
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Using device: {self.device}")
        
        # Model status
        self.models_loaded = False
        self.predictor = None
        
        # Emotion labels
        self.emotion_labels = EMOTION_LABELS
        
    def check_models_exist(self):
        """Check if model files exist"""
        models = {
            'gpt2': self.checkpoint_dir / 'gpt2' / 'final_model.pt',
            'roberta': self.checkpoint_dir / 'roberta' / 'final_model.pt',
            'deberta': self.checkpoint_dir / 'deberta' / 'final_model.pt'
        }
        
        status = {}
        for model_name, model_path in models.items():
            exists = model_path.exists()
            status[model_name] = exists
            if not exists:
                logger.warning(f"Model file not found: {model_path}")
        
        return status
    
    def load_models(self, force_reload = False):
        """Load all emotion models"""
        if self.models_loaded and not force_reload:
            logger.info("Models already loaded")
            return True
        
        try:
            # Check if models exist
            model_status = self.check_models_exist()
            if not all(model_status.values()):
                missing = [k for k, v in model_status.items() if not v]
                logger.error(f"Missing model files: {missing}")
                logger.error("Please ensure models are downloaded to the checkpoints directory")
                return False
            
            # Use the already imported predictor
            
            logger.info("Loading emotion models...")
            self.predictor = EmotionPredictor()
            
            # Load models
            self.predictor.load_models()
            self.models_loaded = True
            logger.info("All emotion models loaded successfully")
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import emotion modules: {e}")
            logger.error("Ensure you're in the correct environment with all dependencies")
            return False
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def predict_emotions(self, texts, 
                        batch_size=24):
        """
        Predict emotions for a list of texts
        
        Args:
            texts: List of review texts
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions
        """
        if not self.models_loaded:
            if not self.load_models():
                logger.error("Cannot predict without loaded models")
                return pd.DataFrame()
        
        results = []
        
        # Add model info to progress description
        num_models = len(self.predictor.models) if hasattr(self.predictor, 'models') else 3
        
        # Process in batches with clearer progress
        print(f"🎭 Running emotion analysis using {num_models} models on {len(texts)} reviews...")
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc=f"Predicting emotions ({num_models} models)",
                     unit="batch"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Use batch prediction for MASSIVE speedup
                batch_preds = self.predictor.predict_batch(batch_texts, ensemble_only=False)
                
                for pred in batch_preds:
                    if pred and 'ensemble' in pred:
                        result = {
                            'text': pred['text'][:100] + '...' if len(pred['text']) > 100 else pred['text'],
                            'emotion': pred['ensemble']['emotion'],
                            'confidence': pred['ensemble']['confidence']
                        }
                        
                        # Add ensemble-level information
                        if 'individual_predictions' in pred['ensemble']:
                            result['individual_predictions'] = pred['ensemble']['individual_predictions']
                        if 'individual_confidences' in pred['ensemble']:
                            result['individual_confidences'] = pred['ensemble']['individual_confidences']
                        if 'disagreement_penalty' in pred['ensemble']:
                            result['disagreement_penalty'] = pred['ensemble']['disagreement_penalty']
                        if 'prediction_agreement' in pred['ensemble']:
                            result['prediction_agreement'] = pred['ensemble']['prediction_agreement']
                        
                        # Add raw probabilities if available (for entropy calculation)
                        if 'probabilities' in pred['ensemble']:
                            result['raw_probabilities'] = pred['ensemble']['probabilities']
                        
                        # Add individual model results if available
                        if 'individual_models' in pred:
                            for model, model_pred in pred['individual_models'].items():
                                result[f'{model}_emotion'] = model_pred['emotion']
                                result[f'{model}_confidence'] = model_pred['confidence']
                                # Also store individual probabilities for detailed analysis
                                if 'probabilities' in model_pred:
                                    result[f'{model}_probabilities'] = model_pred['probabilities']
                        
                        results.append(result)
                    else:
                        # Fallback result
                        results.append({
                            'text': pred.get('text', '')[:100] + '...' if len(pred.get('text', '')) > 100 else pred.get('text', ''),
                            'emotion': 'unknown',
                            'confidence': 0.0
                        })
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Add empty results for failed batch
                for text in batch_texts:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'emotion': 'unknown',
                        'confidence': 0.0
                    })
        
        return pd.DataFrame(results)
    
    def analyze_reviews(self, df, 
                       text_column='review',
                       sample_size=None):
        """
        Analyze emotions in reviews
        
        Args:
            df: DataFrame with reviews
            text_column: Column name containing review text
            sample_size: Optional sample size for large datasets
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting emotion analysis on {len(df)} reviews")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} reviews from {len(df)}")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        # Get texts
        texts = df_sample[text_column].tolist()
        
        # Get individual model predictions (which includes ensemble data)
        predictions_df = self.predict_emotions(texts)
        
        if len(predictions_df) == 0:
            logger.error("No predictions generated")
            return {}
        
        # Analyze results
        emotion_counts = predictions_df['emotion'].value_counts().to_dict()
        emotion_percentages = {
            emotion: count / len(predictions_df) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        # Calculate average confidence by emotion
        avg_confidence = predictions_df.groupby('emotion')['confidence'].mean().to_dict()
        
        # Get high confidence predictions
        high_conf_mask = predictions_df['confidence'] > 0.8
        high_conf_emotions = predictions_df[high_conf_mask]['emotion'].value_counts().to_dict()
        
        # Model disagreement analysis using already calculated predictions
        disagreement_stats = self._analyze_disagreements_from_predictions(predictions_df)
        
        results = {
            'total_analyzed': len(predictions_df),
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'avg_confidence_by_emotion': avg_confidence,
            'overall_avg_confidence': predictions_df['confidence'].mean(),
            'high_confidence_emotions': high_conf_emotions,
            'high_confidence_ratio': high_conf_mask.sum() / len(predictions_df),
            'predictions_df': predictions_df,
            'disagreement_stats': disagreement_stats
        }
        
        # Add raw probabilities if available
        if 'raw_probabilities' in predictions_df.columns:
            results['raw_probabilities'] = predictions_df['raw_probabilities'].tolist()
        
        # Save detailed results
        self._save_predictions(predictions_df, df_sample)
        
        # Create visualizations
        self.visualize_results(results, game_name="Game")
        
        return results
    
    def _analyze_disagreements_from_predictions(self, predictions_df):
        """Analyze model disagreements using existing predictions"""
        try:
            if len(predictions_df) == 0:
                return {}
            
            # Calculate disagreement rate using ALL predictions, not a sample
            disagreements = 0
            for _, row in predictions_df.iterrows():
                if 'gpt2_emotion' in row and 'roberta_emotion' in row and 'deberta_emotion' in row:
                    emotions = [row['gpt2_emotion'], row['roberta_emotion'], row['deberta_emotion']]
                    if len(set(emotions)) > 1:
                        disagreements += 1
            
            disagreement_rate = disagreements / len(predictions_df) if len(predictions_df) > 0 else 0
            
            return {
                'sample_size': len(predictions_df),  # Now using ALL predictions
                'disagreement_rate': disagreement_rate,
                'disagreement_count': disagreements
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze disagreements: {e}")
            return {}
    
    def visualize_results(self, results, game_name = "Game"):
        """Create visualizations for emotion analysis"""
        if not results or 'emotion_counts' not in results:
            logger.warning("No results to visualize")
            return
        
        # 1. Emotion Distribution Pie Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart with better visibility for small slices
        emotion_counts = results['emotion_counts']
        # Sort emotions by count
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0] for e in sorted_emotions]
        sizes = [e[1] for e in sorted_emotions]
        
        # Define colors for each emotion type
        emotion_colors = {
            'joy': '#FFD700',      # Gold
            'love': '#FF69B4',     # Hot pink
            'surprise': '#FF8C00', # Dark orange
            'anger': '#DC143C',    # Crimson
            'sadness': '#4169E1',  # Royal blue
            'fear': '#8B4513'      # Saddle brown
        }
        colors = [emotion_colors.get(label, '#808080') for label in labels]
        
        # Handle extreme dominance (e.g., 76.5% joy)
        total_predictions = sum(sizes)
        dominant_threshold = 0.7  # If one emotion > 70%, group smaller ones
        
        if sizes and sizes[0] / total_predictions > dominant_threshold:
            # Group smaller emotions as "Others"
            dominant_emotion = labels[0]
            dominant_count = sizes[0]
            other_count = sum(sizes[1:])
            
            if other_count > 0:
                labels = [dominant_emotion, 'Others']
                sizes = [dominant_count, other_count]
                colors = [colors[0], '#CCCCCC']
                explode = [0, 0.1]  # Explode "Others" for visibility
            else:
                labels = [dominant_emotion]
                sizes = [dominant_count]
                colors = [colors[0]]
                explode = [0]
        else:
            # Normal case - explode small slices
            explode = [0.1 if size/sum(sizes) < 0.05 else 0 for size in sizes]
        
        # Custom autopct to show counts
        def autopct_format(pct):
            absolute = int(pct / 100. * total_predictions)
            return f'{pct:.1f}%\n({absolute:,})' if pct > 3 else f'({absolute:,})'
        
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels,
            colors=colors,
            autopct=autopct_format,
            startangle=90,
            explode=explode,
            textprops={'fontsize': 10}
        )
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax1.set_title(f'Emotion Distribution - {game_name}\n'
                     f'Total Predictions: {total_predictions:,}', 
                     fontsize=16, fontweight='bold')
        
        # Bar chart with confidence
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        confidences = [results['avg_confidence_by_emotion'].get(e, 0) for e in emotions]
        
        x = np.arange(len(emotions))
        bars = ax2.bar(x, counts, color=colors[:len(emotions)])
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{conf:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Emotion', fontsize=12)
        ax2.set_ylabel('Number of Reviews', fontsize=12)
        ax2.set_title(f'Emotion Counts with Avg Confidence - {game_name}', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(emotions, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "emotion_distribution.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved emotion visualization to {viz_file}")
        
        # 2. Confidence Distribution Plot
        self._create_confidence_distribution(results, game_name)
        
        # 3. Model Disagreement Analysis (if available)
        if 'disagreement_stats' in results and results['disagreement_stats']:
            self._create_disagreement_analysis(results, game_name)
        
        # 4. Model disagreement analysis using Jensen-Shannon Divergence
        if 'predictions_df' in results:
            self._create_uncertainty_visualizations(results, game_name)
        
        # 5. Create detailed report
        self._create_detailed_report(results, game_name)
    
    def _create_confidence_distribution(self, results, game_name):
        """Create confidence distribution histogram"""
        if 'predictions_df' not in results:
            return
        
        predictions_df = results['predictions_df']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with better binning
        n_bins = np.linspace(0, 1, 21)  # 0 to 1 in 0.05 increments
        n, bins, patches = ax.hist(predictions_df['confidence'], bins=n_bins, 
                                  alpha=0.7, color='skyblue', edgecolor='black')
        
        # Color code bins by confidence level
        for i, patch in enumerate(patches):
            if bins[i] < 0.5:
                patch.set_facecolor('#e74c3c')  # Red for low confidence
            elif bins[i] < 0.8:
                patch.set_facecolor('#f39c12')  # Orange for medium
            else:
                patch.set_facecolor('#27ae60')  # Green for high
        
        # Add statistics
        mean_conf = predictions_df['confidence'].mean()
        median_conf = predictions_df['confidence'].median()
        std_conf = predictions_df['confidence'].std()
        
        ax.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2,
                  label=f'Mean: {mean_conf:.3f}')
        ax.axvline(median_conf, color='blue', linestyle='dashed', linewidth=2,
                  label=f'Median: {median_conf:.3f}')
        
        # Add text box with stats
        textstr = f'μ = {mean_conf:.3f}\nσ = {std_conf:.3f}\nn = {len(predictions_df)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Number of Predictions', fontsize=12)
        ax.set_title(f'Prediction Confidence Distribution - {game_name}', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with clean name
        viz_file = self.viz_dir / "confidence_distribution.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confidence distribution to {viz_file}")
    
    def _create_disagreement_analysis(self, results, game_name):
        """Create model disagreement visualization"""
        disagreement_stats = results.get('disagreement_stats', {})
        if not disagreement_stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of agreement vs disagreement
        sizes = [
            disagreement_stats['sample_size'] - disagreement_stats['disagreement_count'],
            disagreement_stats['disagreement_count']
        ]
        labels = ['Agreement', 'Disagreement']
        colors = ['#2ecc71', '#e74c3c']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Model Agreement Analysis - {game_name}', fontsize=14, fontweight='bold')
        
        # Bar chart showing emotion distribution by confidence level
        if 'predictions_df' in results:
            predictions_df = results['predictions_df']
            
            # Create confidence bins
            predictions_df['conf_bin'] = pd.cut(predictions_df['confidence'], 
                                               bins=[0, 0.6, 0.8, 1.0],
                                               labels=['Low (<0.6)', 'Medium (0.6-0.8)', 'High (>0.8)'])
            
            # Count emotions by confidence bin
            conf_emotion_counts = predictions_df.groupby(['conf_bin', 'emotion']).size().unstack(fill_value=0)
            
            # Create stacked bar chart
            conf_emotion_counts.plot(kind='bar', stacked=True, ax=ax2, colormap='tab10')
            ax2.set_xlabel('Confidence Level', fontsize=12)
            ax2.set_ylabel('Number of Predictions', fontsize=12)
            ax2.set_title('Emotion Distribution by Confidence Level', fontsize=14, fontweight='bold')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
            ax2.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save with clean name
        viz_file = self.viz_dir / "model_agreement_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model agreement analysis to {viz_file}")
    
    def _create_uncertainty_visualizations(self, results, game_name):
        """Create model disagreement visualizations using Jensen-Shannon Divergence"""
        predictions_df = results['predictions_df']
        
        # Calculate multi-way Jensen-Shannon Divergence (JSD) for model disagreement
        # JSD(P1, P2, P3) = (1/n) * [KL(P1||M) + KL(P2||M) + KL(P3||M)]
        # where M = (P1 + P2 + P3) / n (average distribution)
        # JSD measures how much the models disagree on their probability distributions
        
        def kl_divergence(p, q, epsilon=1e-10):
            """Calculate KL divergence between two probability distributions"""
            p = np.clip(p, epsilon, 1.0)  # Avoid log(0)
            q = np.clip(q, epsilon, 1.0)
            return np.sum(p * np.log(p / q))
        
        def jensen_shannon_divergence(distributions):
            """Calculate multi-way Jensen-Shannon Divergence"""
            if len(distributions) < 2:
                return 0.0
            
            # Convert to numpy arrays
            distributions = [np.array(d) for d in distributions]
            n_models = len(distributions)
            
            # Calculate average distribution M
            M = np.mean(distributions, axis=0)
            
            # Calculate KL divergence from each distribution to M
            kl_sum = sum(kl_divergence(P, M) for P in distributions)
            
            # Jensen-Shannon divergence is the average
            jsd = kl_sum / n_models
            
            # Normalize by log(2) to get value roughly in [0, 1] for binary case
            # For multi-way, we use log(n_models) for normalization
            return jsd / np.log(n_models)
        
        # DEBUG: Print all available columns for troubleshooting
        logger.info("=== DEBUGGING MODEL DISAGREEMENT CALCULATION ===")
        logger.info(f"Total columns in predictions_df: {len(predictions_df.columns)}")
        logger.info(f"All columns: {list(predictions_df.columns)}")
        
        # Check if we have individual model probabilities for JSD calculation
        model_columns = [col for col in predictions_df.columns if col.endswith('_probabilities')]
        logger.info(f"Columns ending with '_probabilities': {model_columns}")
        
        # Also check for other probability-related columns
        prob_columns = [col for col in predictions_df.columns if 'prob' in col.lower()]
        logger.info(f"All probability-related columns: {prob_columns}")
        
        # Check for confidence columns
        confidence_columns = [col for col in predictions_df.columns if col.endswith('_confidence')]
        logger.info(f"Confidence columns: {confidence_columns}")
        
        # Sample a few rows to see what data we actually have
        if len(predictions_df) > 0:
            logger.info("=== SAMPLE DATA INSPECTION ===")
            sample_row = predictions_df.iloc[0]
            for col in predictions_df.columns:
                if 'prob' in col.lower() or 'confidence' in col.lower():
                    logger.info(f"Column '{col}': type={type(sample_row[col])}, value={sample_row[col]}")
        
        if len(model_columns) >= 2:
            logger.info(f"✅ Found {len(model_columns)} model probability distributions - proceeding with JSD")
            logger.info(f"Model columns: {model_columns}")
            
            jsd_scores = []
            successful_calculations = 0
            
            for idx, row in predictions_df.iterrows():
                # Extract probability distributions from each model
                model_probs = []
                for col in model_columns:
                    try:
                        # Safely get the value without triggering array boolean evaluation
                        probs = row[col]
                        
                        # Skip if explicitly None
                        if probs is None:
                            continue
                            
                        # Handle pandas NA/null values safely
                        if hasattr(probs, '__len__') and len(probs) == 0:
                            continue
                            
                        # Convert to numpy array and validate
                        probs_array = np.array(probs)
                        
                        # Check if it's a valid probability distribution
                        if probs_array.size > 0 and len(probs_array.shape) == 1 and not np.isnan(probs_array).all():
                            model_probs.append(probs_array)
                            if idx < 3:  # Debug first few rows
                                logger.info(f"Row {idx}, {col}: shape={probs_array.shape}, sample={probs_array[:3]}")
                        else:
                            if idx < 3:
                                logger.warning(f"Row {idx}, {col}: Invalid probability array - shape={probs_array.shape}, all_nan={np.isnan(probs_array).all()}")
                                
                    except (ValueError, TypeError, AttributeError) as e:
                        if idx < 3:
                            logger.warning(f"Row {idx}, {col}: Could not process: {e}, type={type(probs)}")
                    except Exception as e:
                        if idx < 3:
                            logger.error(f"Row {idx}, {col}: Unexpected error: {e}")
                
                # Calculate JSD if we have at least 2 model predictions
                if len(model_probs) >= 2:
                    jsd = jensen_shannon_divergence(model_probs)
                    jsd_scores.append(jsd)
                    successful_calculations += 1
                    if idx < 3:  # Debug first few rows
                        logger.info(f"Row {idx}: JSD = {jsd:.4f} from {len(model_probs)} models")
                else:
                    jsd_scores.append(0.0)  # No disagreement if only one model
                    if idx < 3:
                        logger.info(f"Row {idx}: No JSD calculation (only {len(model_probs)} models)")
            
            predictions_df['model_disagreement'] = jsd_scores
            logger.info(f"✅ Calculated Jensen-Shannon Divergence for {successful_calculations}/{len(predictions_df)} predictions")
            logger.info(f"JSD statistics: mean={np.mean(jsd_scores):.4f}, std={np.std(jsd_scores):.4f}, max={np.max(jsd_scores):.4f}")
            
        else:
            logger.warning("❌ No individual model probabilities found - cannot calculate model disagreement")
            logger.info("Available columns: " + ", ".join([col for col in predictions_df.columns if 'prob' in col.lower()]))
            
            # Fallback: Use confidence variance as a proxy for disagreement
            if len(confidence_columns) >= 2:
                logger.info(f"🔄 Using confidence variance from {len(confidence_columns)} models as disagreement proxy")
                
                disagreement_scores = []
                for idx, row in predictions_df.iterrows():
                    confidences = [row[col] for col in confidence_columns if pd.notna(row[col])]
                    if len(confidences) >= 2:
                        # Use coefficient of variation (std/mean) as disagreement measure
                        cv = np.std(confidences) / (np.mean(confidences) + 1e-10)
                        disagreement_scores.append(cv)
                    else:
                        disagreement_scores.append(0.0)
                
                predictions_df['model_disagreement'] = disagreement_scores
                logger.info("Using confidence coefficient of variation as disagreement proxy")
            else:
                logger.error("❌ Cannot calculate model disagreement - no individual model data found")
                logger.info("This means the emotion predictor is not returning individual model results")
                predictions_df['model_disagreement'] = 0.0
                return
        
        # Create figure with improved layout: 2x1 on left, 1 large on right
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
        
        # Left column: 2x1 compact plots
        ax1 = fig.add_subplot(gs[0, 0])  # Top left
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
        # Right side: Full height for uncertainty analysis
        ax3 = fig.add_subplot(gs[:, 1])  # Right side (full height)
        
        emotions = predictions_df['emotion'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))
        
        # 1. Model Disagreement Distribution (compact)
        ax1.hist(predictions_df['model_disagreement'], bins=20, 
                alpha=0.7, color='darkred', edgecolor='black')
        mean_disagreement = predictions_df['model_disagreement'].mean()
        ax1.axvline(mean_disagreement, color='blue', linestyle='dashed', linewidth=2,
                   label=f'Mean: {mean_disagreement:.3f}')
        ax1.set_xlabel('Model Disagreement (JSD)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Disagreement Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Emotion-wise disagreement boxplot (compact)
        disagreement_by_emotion = []
        emotion_labels = []
        for emotion in emotions:
            mask = predictions_df['emotion'] == emotion
            if mask.sum() > 0:
                disagreement_by_emotion.append(predictions_df[mask]['model_disagreement'])
                emotion_labels.append(f"{emotion}\n({mask.sum()})")
        
        if disagreement_by_emotion:
            bp = ax2.boxplot(disagreement_by_emotion, labels=emotion_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                if i < len(colors):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.7)
        
        ax2.set_ylabel('Model Disagreement (JSD)', fontsize=11)
        ax2.set_title('Disagreement by Emotion', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', labelsize=9)
        
        # 3. Large Right: Model Uncertainty Analysis (Confidence vs Disagreement)
        # Plot with emotion colors, no size complexity
        for i, emotion in enumerate(emotions):
            mask = predictions_df['emotion'] == emotion
            if mask.sum() > 0:
                # Sample for readability if too many points
                sample_size = min(200, mask.sum())
                if sample_size < mask.sum():
                    sample_indices = np.random.choice(predictions_df[mask].index, sample_size, replace=False)
                    sample_mask = predictions_df.index.isin(sample_indices)
                else:
                    sample_mask = mask
                
                ax3.scatter(predictions_df[sample_mask]['confidence'], 
                           predictions_df[sample_mask]['model_disagreement'],
                           alpha=0.6, color=colors[i], label=emotion, s=40)
                
                # Add individual emotion trend lines
                try:
                    emotion_data = predictions_df[mask]
                    if len(emotion_data) > 2:  # Need at least 3 points for trend
                        z_emotion = np.polyfit(emotion_data['confidence'], emotion_data['model_disagreement'], 1)
                        p_emotion = np.poly1d(z_emotion)
                        x_emotion = np.linspace(emotion_data['confidence'].min(), emotion_data['confidence'].max(), 50)
                        ax3.plot(x_emotion, p_emotion(x_emotion), color=colors[i], alpha=0.5, linewidth=2, linestyle='-')
                except:
                    pass
        
        # Add overall trend line
        try:
            z = np.polyfit(predictions_df['confidence'], predictions_df['model_disagreement'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(predictions_df['confidence'].min(), predictions_df['confidence'].max(), 100)
            ax3.plot(x_trend, p(x_trend), "black", alpha=0.8, linewidth=3, linestyle='--', label='Overall Trend')
            
            # Add correlation stats
            corr = predictions_df['confidence'].corr(predictions_df['model_disagreement'])
            
            stats_text = f'Overall Correlation: {corr:.3f}'
            
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top')
        except:
            pass
        
        ax3.set_xlabel('Ensemble Confidence Score', fontsize=14)
        ax3.set_ylabel('Model Disagreement (Jensen-Shannon Divergence)', fontsize=14)
        ax3.set_title('Model Uncertainty Analysis\n(Confidence vs Disagreement by Emotion)', fontsize=16, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        
        plt.suptitle(f'Model Disagreement Analysis (Jensen-Shannon Divergence) - {game_name}', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save with updated name
        viz_file = self.viz_dir / "model_disagreement_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model disagreement analysis to {viz_file}")
    
    def _create_detailed_report(self, results, game_name):
        """Create detailed emotion analysis report"""
        report_file = self.viz_dir / "emotion_analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"EMOTION ANALYSIS REPORT\n")
            f.write(f"Game: {game_name}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total reviews analyzed: {results['total_analyzed']}\n")
            f.write(f"Overall average confidence: {results['overall_avg_confidence']:.3f}\n")
            f.write(f"High confidence predictions: {results['high_confidence_ratio']:.1%}\n\n")
            
            f.write("EMOTION DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for emotion, percentage in sorted(results['emotion_percentages'].items(), 
                                            key=lambda x: x[1], reverse=True):
                count = results['emotion_counts'][emotion]
                avg_conf = results['avg_confidence_by_emotion'].get(emotion, 0)
                f.write(f"{emotion:10s}: {percentage:5.1f}% ({count:4d} reviews, avg conf: {avg_conf:.3f})\n")
            
            f.write("\nMODEL AGREEMENT\n")
            f.write("-" * 40 + "\n")
            disagree = results.get('disagreement_stats', {})
            if disagree:
                f.write(f"Sample size: {disagree.get('sample_size', 0)}\n")
                f.write(f"Disagreement rate: {disagree.get('disagreement_rate', 0):.1%}\n")
                f.write(f"Reviews with disagreement: {disagree.get('disagreement_count', 0)}\n")
            else:
                f.write("No disagreement analysis available\n")
            
            f.write("\nEMOTION INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Dominant emotion
            dominant = max(results['emotion_percentages'].items(), key=lambda x: x[1])
            f.write(f"Dominant emotion: {dominant[0]} ({dominant[1]:.1f}%)\n")
            
            # Emotional valence
            positive_emotions = ['joy', 'love', 'surprise']
            negative_emotions = ['sadness', 'anger', 'fear']
            
            positive_pct = sum(results['emotion_percentages'].get(e, 0) for e in positive_emotions)
            negative_pct = sum(results['emotion_percentages'].get(e, 0) for e in negative_emotions)
            
            f.write(f"\nEmotional Valence:\n")
            f.write(f"  Positive emotions: {positive_pct:.1f}%\n")
            f.write(f"  Negative emotions: {negative_pct:.1f}%\n")
            f.write(f"  Valence ratio: {positive_pct/negative_pct:.2f}:1\n" if negative_pct > 0 else "")
        
        logger.info(f"Saved emotion report to {report_file}")
    
    def _save_predictions(self, predictions_df, original_df):
        """Save predictions to CSV"""
        # Combine with original data
        combined_df = pd.concat([
            original_df.reset_index(drop=True),
            predictions_df[['emotion', 'confidence']].reset_index(drop=True)
        ], axis=1)
        
        # Save with clean name
        pred_file = self.viz_dir / "emotion_predictions.csv"
        combined_df.to_csv(pred_file, index=False)
        logger.info(f"Saved predictions to {pred_file}")