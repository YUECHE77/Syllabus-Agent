from transformers import BertModel
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout=0.3):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)

        self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)  # [768, num_class]

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        outputs = outputs.pooler_output  # [batch_size, hidden_size] = [batch_size, 768]
        outputs = self.dropout(outputs)

        outputs = self.classifier(outputs)  # [batch_size, num_class]

        return outputs
