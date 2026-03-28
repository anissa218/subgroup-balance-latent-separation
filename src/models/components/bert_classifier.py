import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AutoModel

class cusBERTClassifier(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.model.config.hidden_size
        
        self.classifier = nn.Linear(self.hidden_size, output_size)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):

        input_ids, attention_mask = x[0],x[1]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        logits = self.classifier(pooled_output)

        return pooled_output, logits


if __name__ == "__main__":
    _ = cusBERTClassifier()