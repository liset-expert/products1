import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim

def train_bert_model(products_df, num_epochs=5, batch_size=32, learning_rate=1e-5, max_length=128):
    
    """
    Trains a multi-label classification model using BERT for sequence classification,
    predicts labels for the test set, and calculates accuracy, precision, recall, and F1-score.

    Args:
        products_df (DataFrame): DataFrame containing 'Information' and 'extracted_categories' columns.
        num_epochs (int, optional): Number of training epochs. Default is 5.
        batch_size (int, optional): Batch size for training. Default is 32.
        learning_rate (float, optional): Learning rate for optimization. Default is 1e-5.
        max_length (int, optional): Maximum length of input sequences. Default is 128.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, and F1-score.
    """
    
    # Division of data into training and test sets
    train_df, test_df = train_test_split(products_df, test_size=0.2, random_state=42)

    # Category preprocessing
    mlb = MultiLabelBinarizer()
    train_categories = [set(categories) for categories in train_df['extracted_categories']]
    test_categories = [set(categories) for categories in test_df['extracted_categories']]
    encoded_train_categories = mlb.fit_transform(train_categories)
    encoded_test_categories = mlb.transform(test_categories)

    # Text tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_train_texts = tokenizer(list(train_df['Information']), padding=True, truncation=True, return_tensors='pt', max_length=128)
    encoded_test_texts = tokenizer(list(test_df['Information']), padding=True, truncation=True, return_tensors='pt', max_length=128)

    # BERT Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

    # DataLoader Definition
    class CustomDataset(Dataset):
        def __init__(self, encoded_texts, encoded_categories):
            self.encoded_texts = encoded_texts
            self.encoded_categories = encoded_categories

        def __len__(self):
            return len(self.encoded_texts['input_ids'])

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encoded_texts.items()}, self.encoded_categories[idx]

    train_dataset = CustomDataset(encoded_train_texts, encoded_train_categories)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = BCEWithLogitsLoss()

    # Training cycle
    for epoch in range(5):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            targets = targets.to(device)
            outputs = model(**inputs).logits
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            # Print training progress
            print(f"Epoch [{epoch+1}/{5}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    model.eval()
    test_dataset = CustomDataset(encoded_test_texts, encoded_test_categories)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()} 
            labels = labels.to(device)
            outputs = model(**inputs).logits
            predictions = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Metric calculation
    all_predictions_tensor = torch.stack(all_predictions)
    all_labels_tensor = torch.stack(all_labels)

    # Accuracy
    correct = (all_predictions_tensor == all_labels_tensor).sum().item()
    total = all_labels_tensor.numel()
    accuracy = correct / total

    # True Positives, False Positives, True Negatives, False Negatives
    true_positives = ((all_predictions_tensor == 1) & (all_labels_tensor == 1)).sum().item()
    false_positives = ((all_predictions_tensor == 1) & (all_labels_tensor == 0)).sum().item()
    true_negatives = ((all_predictions_tensor == 0) & (all_labels_tensor == 0)).sum().item()
    false_negatives = ((all_predictions_tensor == 0) & (all_labels_tensor == 1)).sum().item()

    # Precision, recall, F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


