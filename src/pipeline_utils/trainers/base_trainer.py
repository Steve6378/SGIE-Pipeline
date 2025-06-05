"""
Base trainer with dual-scheduler approach for optimal fine-tuning:

1. Linear Warmup + Decay Scheduler:
   - Provides stable warmup and smooth learning rate decay
   - Steps every training step for consistent exploration

2. ReduceLROnPlateau Scheduler:
   - Aggressively reduces LR by 0.5x after 3 validation plateau steps
   - Pushes learning rate down to 1e-8 for maximum fine-tuning
   - Steps based on validation loss performance

This combination gives the best of both worlds: stable training with aggressive fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import yaml
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config_path, model_type):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = model_type
        self.model_config = self.config['models'][model_type]
        self.training_config = self.config['training']
        self.num_classes = len(self.config['emotions'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / model_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def build_model(self):
        pass
    
    def setup_training(self, train_loader):
        # Account for gradient accumulation in scheduler steps
        num_training_steps = (len(train_loader) // self.model_config['gradient_accumulation_steps']) * self.model_config['num_epochs']
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_config['learning_rate']),
            weight_decay=float(self.model_config['weight_decay'])
        )
        
        # Dual-scheduler approach: Best of both worlds
        # 1. Linear warmup + decay for stable training and exploration
        self.linear_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.model_config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        # Initialize scheduler state to avoid warning
        self.linear_scheduler._step_count = 1
        
        # 2. ReduceLROnPlateau for aggressive fine-tuning when validation plateaus
        # Reduces LR by 0.5x after 3 validation failures, down to 1e-8
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-8
        )
        
        if self.training_config['fp16'] and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training {self.model_type}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.model_config['gradient_accumulation_steps']
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.model_config['gradient_accumulation_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.linear_scheduler.step()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.model_config['gradient_accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % self.model_config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config['max_grad_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.linear_scheduler.step()
            
            total_loss += loss.item() * self.model_config['gradient_accumulation_steps']
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress display with current lr
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}",
                'lr': f"{current_lr:.1e}",
                'patience': f"{self.patience_counter}/{self.training_config['early_stopping_patience']}"
            })
            
            # Periodic validation evaluation during training
            eval_steps = self.training_config.get('eval_steps', 100)
            if val_loader is not None and (batch_idx + 1) % eval_steps == 0:
                print(f"\n--- Validation at step {batch_idx + 1} ---")
                val_metrics = self.evaluate_subset(val_loader, max_batches=40)  # Eval ~1000 samples
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
                
                # Step plateau scheduler based on validation loss
                old_lr = self.optimizer.param_groups[0]['lr']
                self.plateau_scheduler.step(val_metrics['loss'])
                new_lr = self.optimizer.param_groups[0]['lr']
                
                # If plateau scheduler reduced LR, update linear scheduler's base LRs
                if new_lr < old_lr:
                    print(f"Plateau scheduler reduced LR from {old_lr:.1e} to {new_lr:.1e}")
                    # Update linear scheduler's base learning rates
                    ratio = new_lr / old_lr
                    self.linear_scheduler.base_lrs = [g * ratio for g in self.linear_scheduler.base_lrs]
                
                # Check for early stopping during training
                if self.should_stop_early(val_metrics):
                    print(f"Early stopping triggered at step {batch_idx + 1}")
                    return {
                        'loss': total_loss / (batch_idx + 1),
                        'accuracy': correct_predictions / total_predictions,
                        'early_stopped': True
                    }
                
                self.model.train()  # Switch back to training mode
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_predictions
        }
    
    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating {self.model_type}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        return {
            'loss': total_loss / len(eval_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def evaluate_subset(self, eval_loader, max_batches=20):
        """Fast evaluation on subset for periodic validation during training"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        batches_processed = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if batches_processed >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                batches_processed += 1
        
        if batches_processed == 0:
            return {'loss': float('inf'), 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / batches_processed,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader, val_loader):
        self.setup_training(train_loader)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        val_metrics = None  # Initialize to handle early stopping case
        
        for epoch in range(self.model_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.model_config['num_epochs']}")
            
            train_metrics = self.train_epoch(train_loader, val_loader)
            
            # Check if early stopping was triggered during the epoch
            if train_metrics.get('early_stopped', False):
                print("Early stopping triggered during epoch - ending training")
                break
            val_metrics = self.evaluate(val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Learning Rate: {current_lr:.1e}, Patience: {self.patience_counter}/{self.training_config['early_stopping_patience']}")
            
            # Step plateau scheduler at end of epoch based on validation loss
            old_lr = self.optimizer.param_groups[0]['lr']
            self.plateau_scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # If plateau scheduler reduced LR, update linear scheduler's base LRs
            if new_lr < old_lr:
                print(f"Plateau scheduler reduced LR from {old_lr:.1e} to {new_lr:.1e}")
                # Update linear scheduler's base learning rates
                ratio = new_lr / old_lr
                self.linear_scheduler.base_lrs = [g * ratio for g in self.linear_scheduler.base_lrs]
            
            if self.should_save_checkpoint(val_metrics):
                self.save_checkpoint(epoch, val_metrics)
            
            if self.should_stop_early(val_metrics):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Always save final checkpoint regardless of performance
        print("Saving final checkpoint...")
        if val_metrics is None:
            # If we never got to do a full validation, do one now
            val_metrics = self.evaluate(val_loader)
        self.save_checkpoint(epoch, val_metrics, checkpoint_type='final')
        
        return history
    
    def should_save_checkpoint(self, val_metrics):
        metric = self.training_config['early_stopping_metric']
        
        if metric == 'val_loss':
            return val_metrics['loss'] < self.best_val_loss
        elif metric == 'val_acc':
            return val_metrics['accuracy'] > self.best_val_acc
        else:
            return val_metrics['f1'] > self.best_val_f1
    
    def should_stop_early(self, val_metrics):
        metric = self.training_config['early_stopping_metric']
        patience = self.training_config['early_stopping_patience']
        
        if metric == 'val_loss':
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        elif metric == 'val_acc':
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        return self.patience_counter >= patience
    
    def save_checkpoint(self, epoch, metrics, checkpoint_type='best'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'linear_scheduler_state_dict': self.linear_scheduler.state_dict(),
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.model_config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'patience_counter': self.patience_counter
        }
        
        checkpoint_path = self.checkpoint_dir / f'{checkpoint_type}_model.pt'
        torch.save(checkpoint, checkpoint_path)
        
        metrics_path = self.checkpoint_dir / f'{checkpoint_type}_metrics.json'
        # Filter metrics to only include JSON-serializable data
        json_metrics = {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Checkpoint ({checkpoint_type}) saved at epoch {epoch + 1}")
    
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # For evaluation, we only need the model weights
        # Skip optimizer, scheduler, and training state
        
        return checkpoint.get('metrics', {})