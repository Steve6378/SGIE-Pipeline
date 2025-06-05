import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from .base_trainer import BaseTrainer
import math
from tqdm import tqdm


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""
    def __init__(self, in_dim, out_dim, rank=16, alpha=16):
        super().__init__()
        # Initialize A with small random values
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        # Initialize B with zeros
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.scaling = alpha / rank
        
    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear layer augmented with LoRA"""
    def __init__(self, linear, rank=16, alpha=16):
        super().__init__()
        self.linear = linear
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
            
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank=rank, 
            alpha=alpha
        )
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)


class RobertaEmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, use_lora=True,
                 lora_rank=16, lora_alpha=16, dropout_rate=0.1):
        super().__init__()
        
        config = RobertaConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        if use_lora:
            self.apply_lora(lora_rank, lora_alpha)
            
    def apply_lora(self, rank, alpha):
        """Apply LoRA to self-attention and feed-forward layers"""
        lora_count = 0
        for name, module in self.roberta.named_modules():
            if isinstance(module, nn.Linear):
                # Apply LoRA to query, key, value projections and output projections
                if any(key in name for key in ['query', 'key', 'value', 'dense']):
                    # Skip the final classifier layer
                    if 'classifier' not in name:
                        # print(f"Applying LoRA to: {name}")  # Commented out per user request
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = self.roberta
                        for part in parent_name.split('.'):
                            if part:
                                parent = getattr(parent, part)
                        
                        lora_layer = LinearWithLoRA(module, rank, alpha)
                        setattr(parent, child_name, lora_layer)
                        lora_count += 1
        
        print(f"Applied LoRA to {lora_count} layers")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


class RobertaTrainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, 'roberta')
        self.tokenizer = None
        self.build_model()
        
    def build_model(self):
        model_name = self.model_config['name']
        
        print(f"Initializing RoBERTa tokenizer: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # RoBERTa specific configurations
        use_lora = self.model_config.get('use_lora', True)
        lora_rank = self.model_config.get('lora_rank', 16)
        lora_alpha = self.model_config.get('lora_alpha', 16)
        dropout_rate = self.model_config.get('dropout_rate', 0.1)
        
        print(f"Building RoBERTa model with LoRA (rank={lora_rank}, alpha={lora_alpha})")
        self.model = RobertaEmotionClassifier(
            model_name,
            self.num_classes,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate
        )
        
        # LoRA parameters are already set up correctly in LinearWithLoRA
        # Just ensure classifier remains trainable
        for name, param in self.model.roberta.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
        
        if self.training_config['gradient_checkpointing']:
            self.model.roberta.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"RoBERTa Model loaded: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize_dataset(self, texts, max_length=512, show_progress=True):
        """Tokenize dataset with RoBERTa tokenizer"""
        if show_progress:
            print(f"Tokenizing {len(texts)} samples with RoBERTa...")
            iterator = tqdm(texts, desc="Tokenizing")
        else:
            iterator = texts
        
        all_input_ids = []
        all_attention_masks = []
        
        for text in iterator:
            # RoBERTa uses different special tokens
            encoding = self.tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            all_input_ids.append(encoding['input_ids'].squeeze())
            all_attention_masks.append(encoding['attention_mask'].squeeze())
        
        return {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_masks)
        }
    
    def compute_embeddings(self, texts, batch_size=32):
        """Extract embeddings from RoBERTa for analysis"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenize_dataset(batch_texts, show_progress=False)
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model.roberta.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use [CLS] token representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def analyze_attention_weights(self, text):
        """Analyze attention patterns for interpretability"""
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
            outputs = self.model.roberta.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention weights from all layers
        attention_weights = outputs.attentions
        
        # Average attention across all heads and layers
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1))
        
        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'tokens': tokens,
            'attention_weights': avg_attention,
            'layer_attentions': attention_weights
        }