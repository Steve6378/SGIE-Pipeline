#!/usr/bin/env python3
"""
Simple Emotion Prediction Interface - Standalone Version for Final Pipeline
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # Suppress pipeline warnings

import sys
import torch
import warnings

# Suppress the HuggingFace pipeline GPU warning
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")
import torch.nn.functional as F
import yaml
from pathlib import Path
import json

# Import transformers and set verbosity
import transformers
transformers.logging.set_verbosity_error()

# Import trainers from local pipeline_utils
from .trainers.gpt2_trainer import GPT2Trainer
from .trainers.roberta_trainer import RobertaTrainer  
from .trainers.deberta_trainer import DeBERTaTrainer

class EmotionPredictor:
    def __init__(self, config_path=None):
        """Initialize the emotion prediction system"""
        # Use default config if none provided
        if config_path is None:
            config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        
        # Fallback config if file doesn't exist
        if not Path(config_path).exists():
            self.config = {
                'emotions': ['joy', 'love', 'surprise', 'anger', 'sadness', 'fear'],
                'ensemble': {'penalty_weight': 0.1}
            }
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = self.config['emotions']
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(self.emotions)}
        self.id_to_emotion = {i: emotion for i, emotion in enumerate(self.emotions)}
        self.penalty_weight = self.config['ensemble']['penalty_weight']
        
        # Load models lazily
        self.models = {}
        self.models_loaded = False
        
    def load_models(self, models_to_load=None):
        """Load trained model checkpoints"""
        if self.models_loaded:
            return
            
        if models_to_load is None:
            models_to_load = ['gpt2', 'roberta', 'deberta']
        
        trainers = {
            'gpt2': GPT2Trainer,
            'roberta': RobertaTrainer, 
            'deberta': DeBERTaTrainer
        }
        
        print("Loading trained models...")
        
        for model_name in models_to_load:
            if model_name in trainers:
                try:
                    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
                    trainer = trainers[model_name](str(config_path))
                    
                    # Look in local models directory
                    checkpoint_paths = [
                        Path(__file__).parent.parent.parent / f'models/{model_name}/final_model.pt'
                    ]
                    
                    checkpoint_path = None
                    for path in checkpoint_paths:
                        if path.exists():
                            checkpoint_path = path
                            break
                    
                    if checkpoint_path:
                        trainer.load_checkpoint(checkpoint_path)
                        trainer.model.eval()
                        self.models[model_name] = trainer
                        print(f"{model_name} loaded from {checkpoint_path}")
                    else:
                        print(f"Warning: Checkpoint not found for {model_name}")
                        
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
        
        self.models_loaded = True
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def predict_single_model(self, text, model_name):
        """Get prediction from a single model"""
        if model_name not in self.models:
            return None
            
        trainer = self.models[model_name]
        
        with torch.no_grad():
            # Efficient tokenization without max padding
            encoding = trainer.tokenizer(
                text, max_length=512, padding=True, 
                truncation=True, return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get prediction
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1).values.item()
            prediction = torch.argmax(probabilities, dim=-1).item()
            
            return {
                'emotion': self.id_to_emotion[prediction],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0]
            }
    
    def predict_batch_single_model(self, texts, model_name):
        """Get predictions from a single model for a batch of texts"""
        if model_name not in self.models:
            return []
            
        trainer = self.models[model_name]
        
        with torch.no_grad():
            # Batch tokenization - much faster
            encodings = trainer.tokenizer(
                texts, max_length=512, padding=True, 
                truncation=True, return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Batch prediction - major speedup
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            confidences = torch.max(probabilities, dim=-1).values.cpu().numpy()
            predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()
            
            results = []
            for i in range(len(texts)):
                results.append({
                    'emotion': self.id_to_emotion[predictions[i]],
                    'confidence': float(confidences[i]),
                    'probabilities': probabilities[i].cpu().numpy()
                })
            
            return results
    
    def predict_ensemble(self, text):
        """Get ensemble prediction with disagreement penalty"""
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble prediction")
            return None
        
        # Get predictions from all available models
        model_results = {}
        for model_name in self.models:
            result = self.predict_single_model(text, model_name)
            if result:
                model_results[model_name] = result
        
        if len(model_results) < 2:
            print("Not enough model predictions for ensemble")
            return None
        
        # Calculate ensemble with disagreement penalty
        model_names = list(model_results.keys())
        probs = [torch.tensor(model_results[model]['probabilities']) for model in model_names]
        
        # Pad with zeros if we have less than 3 models
        while len(probs) < 3:
            probs.append(torch.zeros_like(probs[0]))
        
        p1, p2, p3 = probs[:3]
        
        # Calculate mean probabilities
        p_mean = (p1 + p2 + p3) / 3.0
        
        # Calculate disagreement penalty
        penalty = self.penalty_weight * (
            torch.pow(p1 - p2, 2) + 
            torch.pow(p2 - p3, 2) + 
            torch.pow(p3 - p1, 2)
        ) / 3.0
        
        # Apply penalty and ensure non-negative
        p_final = F.relu(p_mean - penalty)
        p_final = p_final / (p_final.sum() + 1e-8)
        
        # Get final prediction
        prediction = torch.argmax(p_final).item()
        confidence = torch.max(p_final).item()
        penalty_magnitude = penalty.sum().item()
        
        # Calculate disagreement metrics
        predictions = [self.emotion_to_id[model_results[model]['emotion']] for model in model_names]
        confidences = [model_results[model]['confidence'] for model in model_names]
        
        # Prediction agreement
        majority_pred = max(set(predictions), key=predictions.count)
        agreement = predictions.count(majority_pred) / len(predictions)
        
        # Confidence variance
        conf_variance = torch.var(torch.tensor(confidences)).item()
        
        return {
            'emotion': self.id_to_emotion[prediction],
            'confidence': confidence,
            'individual_predictions': {model: model_results[model]['emotion'] for model in model_names},
            'individual_confidences': {model: model_results[model]['confidence'] for model in model_names},
            'disagreement_penalty': penalty_magnitude,
            'prediction_agreement': agreement,
            'confidence_variance': conf_variance,
            'uncertainty': 'high' if agreement < 0.7 else 'low'
        }
    
    def predict(self, text, ensemble_only=False, detailed=False):
        """Main prediction interface"""
        if not self.models_loaded:
            self.load_models()
        
        if not self.models:
            print("Error: No models loaded! Please check model paths.")
            return None
        
        results = {'text': text}
        
        if not ensemble_only:
            # Get individual model predictions
            results['individual_models'] = {}
            for model_name in self.models:
                pred = self.predict_single_model(text, model_name)
                if pred:
                    results['individual_models'][model_name] = pred
        
        # Get ensemble prediction
        if len(self.models) >= 2:
            ensemble_result = self.predict_ensemble(text)
            if ensemble_result:
                results['ensemble'] = ensemble_result
        
        return results
    
    def predict_batch(self, texts, ensemble_only=False):
        """Batch prediction interface - MUCH faster for multiple texts"""
        if not self.models_loaded:
            self.load_models()
        
        if not self.models:
            print("Error: No models loaded! Please check model paths.")
            return []
        
        batch_results = []
        
        # Get individual model predictions for all texts at once
        individual_predictions = {}
        if not ensemble_only:
            for model_name in self.models:
                individual_predictions[model_name] = self.predict_batch_single_model(texts, model_name)
        
        # Process results for each text
        for i, text in enumerate(texts):
            result = {'text': text}
            
            if not ensemble_only and individual_predictions:
                result['individual_models'] = {}
                for model_name in self.models:
                    if i < len(individual_predictions[model_name]):
                        result['individual_models'][model_name] = individual_predictions[model_name][i]
            
            # Calculate ensemble from individual predictions
            if len(self.models) >= 2 and not ensemble_only:
                ensemble_result = self._calculate_ensemble_from_individual(result.get('individual_models', {}))
                if ensemble_result:
                    result['ensemble'] = ensemble_result
            
            batch_results.append(result)
        
        return batch_results
    
    def _calculate_ensemble_from_individual(self, individual_models):
        """Calculate ensemble from individual model results"""
        if len(individual_models) < 2:
            return None
        
        model_names = list(individual_models.keys())
        probs = []
        for model_name in model_names:
            if 'probabilities' in individual_models[model_name]:
                probs.append(torch.tensor(individual_models[model_name]['probabilities']))
        
        if len(probs) < 2:
            return None
        
        # Pad with zeros if we have less than 3 models
        while len(probs) < 3:
            probs.append(torch.zeros_like(probs[0]))
        
        p1, p2, p3 = probs[:3]
        
        # Calculate mean probabilities
        p_mean = (p1 + p2 + p3) / 3.0
        
        # Calculate disagreement penalty
        penalty = self.penalty_weight * (
            torch.pow(p1 - p2, 2) + 
            torch.pow(p2 - p3, 2) + 
            torch.pow(p3 - p1, 2)
        ) / 3.0
        
        # Apply penalty and ensure non-negative
        p_final = F.relu(p_mean - penalty)
        p_final = p_final / (p_final.sum() + 1e-8)
        
        # Get final prediction
        prediction = torch.argmax(p_final).item()
        confidence = torch.max(p_final).item()
        penalty_magnitude = penalty.sum().item()
        
        # Calculate disagreement metrics
        predictions = [self.emotion_to_id[individual_models[model]['emotion']] for model in model_names if 'emotion' in individual_models[model]]
        confidences = [individual_models[model]['confidence'] for model in model_names if 'confidence' in individual_models[model]]
        
        # Prediction agreement
        majority_pred = max(set(predictions), key=predictions.count) if predictions else 0
        agreement = predictions.count(majority_pred) / len(predictions) if predictions else 0
        
        # Confidence variance
        conf_variance = torch.var(torch.tensor(confidences)).item() if confidences else 0
        
        return {
            'emotion': self.id_to_emotion[prediction],
            'confidence': confidence,
            'individual_predictions': {model: individual_models[model]['emotion'] for model in model_names if 'emotion' in individual_models[model]},
            'individual_confidences': {model: individual_models[model]['confidence'] for model in model_names if 'confidence' in individual_models[model]},
            'disagreement_penalty': penalty_magnitude,
            'prediction_agreement': agreement,
            'confidence_variance': conf_variance,
            'uncertainty': 'high' if agreement < 0.7 else 'low',
            'probabilities': p_final.numpy()
        }