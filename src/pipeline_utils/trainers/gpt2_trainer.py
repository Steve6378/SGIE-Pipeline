import torch
import torch.nn as nn
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
from .base_trainer import BaseTrainer
import math
from tqdm import tqdm


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""
    def __init__(self, in_dim, out_dim, rank=16, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.scaling = alpha / rank
        
    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear/Conv1D layer augmented with LoRA"""
    def __init__(self, layer, rank=16, alpha=16):
        super().__init__()
        self.layer = layer
        # Freeze the original layer
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Handle both Linear and Conv1D layers
        if hasattr(layer, 'in_features'):  # nn.Linear
            in_features = layer.in_features
            out_features = layer.out_features
        elif hasattr(layer, 'weight'):  # Conv1D
            # Conv1D weight shape is (in_features, out_features) - opposite of Linear!
            weight_shape = layer.weight.shape
            if len(weight_shape) == 2:
                in_features, out_features = weight_shape  # Swapped for Conv1D
            else:
                raise ValueError(f"Unexpected Conv1D weight shape: {weight_shape}")
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        # Create LoRA layer
            
        self.lora = LoRALayer(
            in_features, 
            out_features, 
            rank=rank, 
            alpha=alpha
        )
        
    def forward(self, x):
        return self.layer(x) + self.lora(x)


class GPT2EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, use_lora=True, 
                 lora_rank=16, lora_alpha=16):
        super().__init__()
        
        config = GPT2Config.from_pretrained(model_name)
        config.num_labels = num_classes
        
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        if self.gpt2.config.pad_token_id is None:
            self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        
        self.gpt2.resize_token_embeddings(self.gpt2.config.vocab_size + 1)
        
        if use_lora:
            self.apply_lora(lora_rank, lora_alpha)
            
    def apply_lora(self, rank, alpha):
        """Apply LoRA to attention and MLP layers"""
        lora_count = 0
        
        # Access the actual transformer layers
        transformer = self.gpt2.transformer
        
        # Apply LoRA to transformer layers
        
        # Apply LoRA to transformer layers
        if hasattr(transformer, 'h'):
            for layer_idx, layer in enumerate(transformer.h):
                # Apply LoRA to attention layers
                if hasattr(layer, 'attn'):
                    attn = layer.attn
                    if hasattr(attn, 'c_attn'):
                        # Apply LoRA to attention input projection
                        attn.c_attn = LinearWithLoRA(attn.c_attn, rank, alpha)
                        lora_count += 1
                    if hasattr(attn, 'c_proj'):
                        # Apply LoRA to attention output projection
                        attn.c_proj = LinearWithLoRA(attn.c_proj, rank, alpha)
                        lora_count += 1
                
                # Apply LoRA to MLP layers
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    if hasattr(mlp, 'c_fc'):
                        # Apply LoRA to MLP input projection
                        mlp.c_fc = LinearWithLoRA(mlp.c_fc, rank, alpha)
                        lora_count += 1
                    if hasattr(mlp, 'c_proj'):
                        # Apply LoRA to MLP output projection
                        mlp.c_proj = LinearWithLoRA(mlp.c_proj, rank, alpha)
                        lora_count += 1
        
        # Total LoRA layers: {lora_count}
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


class GPT2Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, 'gpt2')
        self.tokenizer = None
        self.build_model()
        
    def build_model(self):
        model_name = self.model_config['name']
        
        # Initialize GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Create model with LoRA
        use_lora = self.model_config.get('use_lora', True)
        lora_rank = self.model_config.get('lora_rank', 16)
        lora_alpha = self.model_config.get('lora_alpha', 16)
        
        # Build GPT-2 model with LoRA
        self.model = GPT2EmotionClassifier(
            model_name, 
            self.num_classes,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
        # LoRA parameters are already set up correctly in LinearWithLoRA
        # Just ensure classifier remains trainable
        for name, param in self.model.gpt2.named_parameters():
            if 'classifier' in name or 'score' in name:
                param.requires_grad = True
        
        if self.training_config['gradient_checkpointing']:
            self.model.gpt2.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"GPT-2 Model loaded: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize_dataset(self, texts, max_length=512, show_progress=True):
        """Tokenize a dataset with detailed progress tracking"""
        if show_progress:
            print(f"Tokenizing {len(texts)} samples...")
            iterator = tqdm(texts, desc="Tokenizing")
        else:
            iterator = texts
        
        all_input_ids = []
        all_attention_masks = []
        
        for text in iterator:
            encoding = self.tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            all_input_ids.append(encoding['input_ids'].squeeze())
            all_attention_masks.append(encoding['attention_mask'].squeeze())
        
        return {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_masks)
        }
    
    def compute_metrics(self, eval_pred):
        """Compute detailed metrics including per-class performance"""
        predictions, labels = eval_pred
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
        
        # Basic metrics from parent class
        metrics = super().evaluate({'predictions': predictions, 'labels': labels})
        
        # Add GPT-2 specific metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            labels, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        # Add per-class F1 scores
        for i in range(self.num_classes):
            if str(i) in report:
                metrics[f'class_{i}_f1'] = report[str(i)]['f1-score']
        
        return metrics