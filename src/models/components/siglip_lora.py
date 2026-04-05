"""SigLIP vision encoder + LoRA + binary classification head.

Returns (features, logits) to match the interface expected by MIMICLitModule /
FMMIMICLitModule, where features = pooler_output and logits = classifier output.
"""

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class SigLIPLoRA(nn.Module):
    """SigLIP vision encoder with LoRA PEFT and a linear classification head.

    Args:
        model_name: HuggingFace model identifier for SigLIP.
        output_size: Number of output classes (default 2 for binary).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability in LoRA layers.
        fallback_n_blocks: If peft is unavailable, unfreeze the last N transformer blocks.
    """

    def __init__(
        self,
        model_name: str = "StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli",
        output_size: int = 2,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fallback_n_blocks: int = 4,
    ):
        super().__init__()

        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, output_size)

        if PEFT_AVAILABLE:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            self.vision_model = get_peft_model(self.vision_model, config)
            self.vision_model.print_trainable_parameters()
        else:
            print(
                f"WARNING: peft not available. Unfreezing last {fallback_n_blocks} "
                "transformer blocks instead."
            )
            for param in self.vision_model.parameters():
                param.requires_grad = False
            encoder_layers = self.vision_model.vision_model.encoder.layers
            for layer in encoder_layers[-fallback_n_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.vision_model.vision_model.post_layernorm.parameters():
                param.requires_grad = True

        # Classifier head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SigLIPLoRA: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")

    def forward(self, pixel_values: torch.Tensor):
        """
        Returns:
            features: pooler output [B, hidden_size]
            logits:   classifier output [B, output_size]
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        features = outputs.pooler_output  # [B, hidden_size]
        logits = self.classifier(features)
        return features, logits
