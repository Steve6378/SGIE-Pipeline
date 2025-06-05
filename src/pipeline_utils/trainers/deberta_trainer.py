import torch
import torch.nn as nn
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, DebertaV2Config
from .base_trainer import BaseTrainer
import math
from tqdm import tqdm
import numpy as np


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer optimized for DeBERTa"""
    def __init__(self, in_dim, out_dim, rank=16, alpha=16):
        super().__init__()
        # Xavier initialization for better gradient flow
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.xavier_uniform_(self.A)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.scaling = alpha / rank
        
    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear, rank=16, alpha=16):
        super().__init__()
        self.linear = linear
        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False
            
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha
        )
        
        # Enable merged inference
        self.merged = False
        
    def forward(self, x):
        if self.merged:
            return self.linear(x)
        return self.linear(x) + self.lora(x)
    
    def merge_lora(self):
        """Merge LoRA weights into the main weights for faster inference"""
        if not self.merged:
            self.linear.weight.data += (self.lora.A @ self.lora.B).T * self.lora.scaling
            self.merged = True
    
    def unmerge_lora(self):
        """Unmerge LoRA weights"""
        if self.merged:
            self.linear.weight.data -= (self.lora.A @ self.lora.B).T * self.lora.scaling
            self.merged = False


class DeBERTaEmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, use_lora=True,
                 lora_rank=16, lora_alpha=16, dropout_rate=0.1,
                 use_disentangled_attention=True):
        super().__init__()
        
        config = DebertaV2Config.from_pretrained(model_name)
        config.num_labels = num_classes
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        
        # DeBERTa specific: enhanced mask decoder
        config.enhanced_mask_decoder = use_disentangled_attention
        
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Additional classifier head with dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        if use_lora:
            self.apply_lora(lora_rank, lora_alpha)
            
    def apply_lora(self, rank, alpha):
        """Apply LoRA to DeBERTa's disentangled attention layers"""
        lora_modules = []
        lora_count = 0
        
        for name, module in self.deberta.named_modules():
            if isinstance(module, nn.Linear):
                # Target specific layers in DeBERTa
                if any(key in name for key in ['query_proj', 'key_proj', 'value_proj', 
                                                'pos_query_proj', 'pos_key_proj',
                                                'dense', 'intermediate.dense', 'output.dense']):
                    if 'classifier' not in name and 'pooler' not in name:
                        # print(f"Applying LoRA to: {name}")  # Commented out per user request
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = self.deberta
                        for part in parent_name.split('.'):
                            if part:
                                parent = getattr(parent, part)
                        
                        lora_layer = LinearWithLoRA(module, rank, alpha)
                        setattr(parent, child_name, lora_layer)
                        lora_modules.append(lora_layer)
                        lora_count += 1
        
        self.lora_modules = lora_modules
        print(f"Applied LoRA to {lora_count} layers")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def merge_lora_weights(self):
        """Merge all LoRA weights for faster inference"""
        if hasattr(self, 'lora_modules'):
            for module in self.lora_modules:
                module.merge_lora()
    
    def unmerge_lora_weights(self):
        """Unmerge all LoRA weights"""
        if hasattr(self, 'lora_modules'):
            for module in self.lora_modules:
                module.unmerge_lora()


class DeBERTaTrainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, 'deberta')
        self.tokenizer = None
        self.build_model()
        
    def build_model(self):
        model_name = self.model_config['name']
        
        print(f"Initializing DeBERTa-v3 tokenizer: {model_name}")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        
        # DeBERTa specific configurations
        use_lora = self.model_config.get('use_lora', True)
        lora_rank = self.model_config.get('lora_rank', 16)
        lora_alpha = self.model_config.get('lora_alpha', 16)
        dropout_rate = self.model_config.get('dropout_rate', 0.1)
        use_disentangled = self.model_config.get('use_disentangled_attention', True)
        
        print(f"Building DeBERTa-v3 model with LoRA (rank={lora_rank}, alpha={lora_alpha})")
        print(f"Disentangled attention: {use_disentangled}")
        
        self.model = DeBERTaEmotionClassifier(
            model_name,
            self.num_classes,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate,
            use_disentangled_attention=use_disentangled
        )
        
        # LoRA parameters are already set up correctly in LinearWithLoRA
        # Just ensure classifier remains trainable
        for name, param in self.model.deberta.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
        
        if self.training_config['gradient_checkpointing']:
            self.model.deberta.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"DeBERTa-v3 Model loaded: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize_dataset(self, texts, max_length=512, show_progress=True):
        """Tokenize dataset with DeBERTa tokenizer"""
        if show_progress:
            print(f"Tokenizing {len(texts)} samples with DeBERTa...")
            iterator = tqdm(texts, desc="Tokenizing", unit="samples")
        else:
            iterator = texts
        
        all_input_ids = []
        all_attention_masks = []
        
        # Process in batches for efficiency
        batch_size = 100
        batch_texts = []
        
        for i, text in enumerate(iterator):
            batch_texts.append(str(text))
            
            if len(batch_texts) == batch_size or i == len(texts) - 1:
                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                all_input_ids.append(encodings['input_ids'])
                all_attention_masks.append(encodings['attention_mask'])
                batch_texts = []
        
        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
    
    def analyze_disentangled_attention(self, text):
        """Analyze DeBERTa's disentangled attention mechanism"""
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Get model outputs with attention weights
            outputs = self.model.deberta.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
        
        # Extract attention patterns
        attention_weights = outputs.attentions
        
        # DeBERTa has both content and position attention
        # Average across layers and heads
        content_attention = torch.stack([attn[:, :, :, :] for attn in attention_weights]).mean(dim=(0, 1))
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'tokens': tokens,
            'attention_weights': content_attention.cpu().numpy(),
            'num_layers': len(attention_weights),
            'attention_shape': attention_weights[0].shape
        }
    
    def get_emotion_embeddings(self, texts, batch_size=16):
        """Extract emotion-aware embeddings using DeBERTa's enhanced representations"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model.deberta.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use pooled output for classification
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(pooled_output.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def train_epoch(self, train_loader, val_loader=None):
        """Override train_epoch to add DeBERTa-specific optimizations"""
        # Enable training mode optimizations
        if hasattr(self.model, 'unmerge_lora_weights'):
            self.model.unmerge_lora_weights()
        
        metrics = super().train_epoch(train_loader, val_loader)
        
        return metrics
    
    def evaluate(self, eval_loader):
        """Override evaluate to use merged LoRA weights for faster inference"""
        # Merge LoRA weights for faster inference
        if hasattr(self.model, 'merge_lora_weights'):
            self.model.merge_lora_weights()
        
        metrics = super().evaluate(eval_loader)
        
        # Unmerge after evaluation
        if hasattr(self.model, 'unmerge_lora_weights'):
            self.model.unmerge_lora_weights()
        
        return metrics