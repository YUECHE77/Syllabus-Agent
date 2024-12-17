from binary_classifier.nets import BertClassifier
from binary_classifier.utils import inference

from transformers import BertTokenizerFast

import torch

pretrain_model_path = r'D:\HuggingFace_models\bert-base-uncased'
model_path = r'D:\CSCI544_project_code\models\CSCI544_Bert_best.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained(pretrain_model_path)

model = BertClassifier(pretrain_model_path, 1, dropout=0.2)
model.load_state_dict(torch.load(model_path))
model.to(device)

query_1 = 'What is the policy on late submissions for assignments?'
query_2 = 'Does the final project allow for group work?'

print(inference(query_1, query_2, model, tokenizer, device))
