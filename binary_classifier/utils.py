import torch
from utilities.constants import dataset_prompt


def evaluate_model(model, data_loader, loss_func, device):
    total = 0
    correct = 0
    total_loss = 0

    for data in data_loader:
        input_ids, attention_mask, labels = [item.to(device) for item in data]

        outputs = model(input_ids, attention_mask).view(-1)

        pred = (torch.sigmoid(outputs) > 0.5).float()

        correct += (pred == labels).sum().item()

        total += len(labels)

        loss = loss_func(outputs, labels)
        total_loss += loss.item() * len(labels)

    accuracy = correct * 100 / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


def inference(query_1, query_2, bert_model, bert_tokenizer, device):
    prompt = dataset_prompt.format(q_1=query_1, q_2=query_2)

    tokenized_input = bert_tokenizer.encode_plus(prompt, padding='longest', truncation=True, max_length=256,
                                                 return_attention_mask=True, return_tensors='pt')
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask).view(-1)

    if torch.sigmoid(outputs) > 0.5:  # 0 means the user is asking about the same course，1 means otherwise
        return 'No'
    else:
        return 'Yes'
