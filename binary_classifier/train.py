from binary_classifier.nets import BertClassifier
from binary_classifier.dataset import load_data
from binary_classifier.utils import evaluate_model

from transformers import BertTokenizerFast

import torch
import torch.nn as nn

from tqdm import tqdm

train_path = r'D:\CSCI544_project_code\binary_classifier\dataset\train.csv'
val_path = r'D:\CSCI544_project_code\binary_classifier\dataset\val.csv'
test_path = r'D:\CSCI544_project_code\binary_classifier\dataset\test.csv'

model_path = r'D:\HuggingFace_models\bert-base-uncased'

epoch_num = 20
batch_size = 4
learning_rate = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dropout = 0.2

tokenizer = BertTokenizerFast.from_pretrained(model_path)

train_loader = load_data(train_path, tokenizer, batch_size=batch_size, mode='train')
val_loader = load_data(val_path, tokenizer, batch_size=batch_size, mode='val')
test_loader = load_data(test_path, tokenizer, batch_size=batch_size, mode='test')

model = BertClassifier(model_path, 1, dropout=dropout).to(device)
print(f'Using device: {device}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_func = nn.BCEWithLogitsLoss()

best_acc = 0.0
all_accuracy = []
all_loss = []

for epoch in range(epoch_num):
    total_batches = len(train_loader)

    with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
        for data in train_loader:
            model.train()
            optimizer.zero_grad()

            input_ids, attention_mask, labels = [item.to(device) for item in data]

            outputs = model(input_ids, attention_mask).view(-1)

            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        with torch.no_grad():
            model.eval()

            accuracy, avg_loss = evaluate_model(model, val_loader, loss_func, device)
            all_accuracy.append(accuracy)
            all_loss.append(avg_loss)
            print(f'\nEpoch: {epoch + 1:02d}, Accuracy: {accuracy:.4f}, loss: {avg_loss:.4f}')

            if accuracy > best_acc:
                best_acc = accuracy
                sub_path = f"bert_accuracy_{accuracy}.pth"
                torch.save(model.state_dict(), sub_path)

print('\nFinished Training!!!\n')

with torch.no_grad():
    model.eval()
    accuracy, avg_loss = evaluate_model(model, test_loader, loss_func, device)
    print(f'On the test set:\n Accuracy: {accuracy:.4f}, loss: {avg_loss:.4f}')
