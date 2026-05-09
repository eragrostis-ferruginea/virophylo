import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType


class DNABERT2Wrapper(nn.Module):
    def __init__(self, model_name="models/dnabert2", lora_rank=16,
                 lora_alpha=32, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        self.embed_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value", "dense"],
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        self._alibi_initialized = False

    def init_alibi(self, device):
        if not self._alibi_initialized:
            if hasattr(self.backbone.base_model.model.encoder, 'rebuild_alibi_tensor'):
                self.backbone.base_model.model.encoder.rebuild_alibi_tensor(
                    size=self.backbone.config.alibi_starting_size,
                    device=device
                )
            self._alibi_initialized = True

    def forward(self, sequences, max_length=2048):
        self.init_alibi(self.backbone.device)
        
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(self.backbone.device) for k, v in encoded.items()}

        outputs = self.backbone(**encoded)
        
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs
        
        cls_embeddings = last_hidden_state[:, 0, :]
        per_position_embeddings = last_hidden_state

        return cls_embeddings, per_position_embeddings

    def get_trainable_params(self):
        trainable = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
        return trainable


class NTWrapper(nn.Module):
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-500m-multi-species",
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value", "dense"],
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, sequences, max_length=2048):
        tokenized = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenized = {k: v.to(self.backbone.device) for k, v in tokenized.items()}

        outputs = self.backbone(**tokenized)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        per_position_embeddings = outputs.last_hidden_state

        return cls_embeddings, per_position_embeddings

    def get_trainable_params(self):
        trainable = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
        return trainable
