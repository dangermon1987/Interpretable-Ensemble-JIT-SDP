import torch
import pandas as pd
import os
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score
from torch.utils.tensorboard import SummaryWriter


class StackedDataset(Dataset):
    def __init__(self, prediction_paths, target_column='is_buggy_commit'):
        """
        Initialize the dataset by loading the predictions from the given paths.

        Arguments:
        - prediction_paths: List of paths to the prediction CSV files.
        - target_column: The column containing the target labels.
        """
        self.prediction_paths = prediction_paths
        self.target_column = target_column
        self.predictions, self.targets = self.load_predictions_and_targets()

    def load_predictions_and_targets(self):
        """Load predictions and target labels from the given CSV files."""
        all_preds = []
        all_targets = []

        # Loop through each prediction file and load the data
        for path in self.prediction_paths:
            if os.path.exists(path):
                pred_data = pd.read_csv(path)


                if 'prob' in pred_data.columns and self.target_column in pred_data.columns:
                    probs = pred_data['prob'].values  # Extract the prediction probabilities
                    targets = pred_data[self.target_column].values  # Extract the target labels
                    all_preds.append(probs)
                    all_targets.append(targets)
                else:
                    raise ValueError(
                        f"Prediction file at {path} does not contain 'prob' and '{self.target_column}' columns.")
            else:
                raise FileNotFoundError(f"Prediction file not found: {path}")

        # Stack all predictions (columns of probabilities from each model) and convert to tensor
        all_preds_tensor = torch.tensor(all_preds).T  # Shape (num_samples, num_models)
        all_targets_tensor = torch.tensor(all_targets[0])  # Assuming targets are consistent across files

        return all_preds_tensor, all_targets_tensor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return the stacked predictions and target for a given index
        return self.predictions[idx], self.targets[idx]


class StackedEnsembleModel(nn.Module):
    def __init__(self, num_models):
        super(StackedEnsembleModel, self).__init__()

        # Linear layer to combine predictions from multiple models
        self.fc_stack = nn.Linear(num_models, 1)  # Single output for binary classification

    def forward(self, inputs):
        """
        Forward pass for the stacked model.

        Arguments:
        - inputs: A tensor of shape (batch_size, num_models), containing stacked predictions.

        Returns:
        - Final prediction (raw output for binary classification).
        """
        output = self.fc_stack(inputs)
        return output


# Function to perform k-fold cross-validation and calculate the evaluation metrics
def cross_validation(prediction_paths, k_folds=5, batch_size=32, num_epochs=10, learning_rate=0.001, patience=3):
    # Initialize dataset
    dataset = StackedDataset(prediction_paths=prediction_paths)

    # Prepare KFold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {
        'f1': [],
        'auc': [],
        'mcc': [],
        'accuracy': [],
        'precision': []
    }

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.predictions)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        # Setup TensorBoard writer for this fold
        writer = SummaryWriter(log_dir=f'./runs/fold_{fold + 1}')

        # Split dataset into training and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoader for training and validation sets
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = StackedEnsembleModel(num_models=len(prediction_paths))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # For binary classification (sigmoid will be applied in evaluation)

        # Early stopping initialization
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        # Training loop
        def train(model, train_loader, optimizer, criterion):
            model.train()
            epoch_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())  # Squeeze to match target shape
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            return epoch_loss / len(train_loader)

        # Validation loop
        def validate(model, val_loader, criterion):
            model.eval()
            epoch_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.float())
                    epoch_loss += loss.item()
                    all_preds.append(outputs)
                    all_targets.append(targets)

            # Flatten the predictions and targets
            all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
            all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

            # Apply sigmoid to get probabilities for binary classification
            all_preds = torch.sigmoid(torch.tensor(all_preds)).numpy()

            # Compute metrics
            f1 = f1_score(all_targets, all_preds.round())
            auc = roc_auc_score(all_targets, all_preds)
            mcc = matthews_corrcoef(all_targets, all_preds.round())
            accuracy = accuracy_score(all_targets, all_preds.round())
            precision = precision_score(all_targets, all_preds.round())

            return epoch_loss / len(val_loader), f1, auc, mcc, accuracy, precision

        # Training and Validation for this fold
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, f1, auc, mcc, accuracy, precision = validate(model, val_loader, criterion)

            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)
            writer.add_scalar('Metrics/AUC', auc, epoch)
            writer.add_scalar('Metrics/MCC', mcc, epoch)
            writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
            writer.add_scalar('Metrics/Precision', precision, epoch)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}: F1 = {f1:.4f}, AUC = {auc:.4f}, MCC = {mcc:.4f}, Accuracy = {accuracy:.4f}, Precision = {precision:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            # Store the metrics for this fold
            fold_results['f1'].append(f1)
            fold_results['auc'].append(auc)
            fold_results['mcc'].append(mcc)
            fold_results['accuracy'].append(accuracy)
            fold_results['precision'].append(precision)

        # Close the writer for this fold
        writer.close()

    # Calculate the average metrics across all folds
    avg_f1 = np.mean(fold_results['f1'])
    avg_auc = np.mean(fold_results['auc'])
    avg_mcc = np.mean(fold_results['mcc'])
    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_precision = np.mean(fold_results['precision'])

    print(f"\nCross-validation completed.")
    print(
        f"Average F1: {avg_f1:.4f}, Average AUC: {avg_auc:.4f}, Average MCC: {avg_mcc:.4f}, Average Accuracy: {avg_accuracy:.4f}, Average Precision: {avg_precision:.4f}")

    return fold_results


# Define paths to the prediction CSV files
prediction_paths = [
    '/workspace/s2156631-thesis/results/expriment_PDG_CFG_CALL_20241105-022928_no_ast_no_manual_512_again/9/',
    'path_to_predictions_model_2.csv',
    'path_to_predictions_model_3.csv'
]

# Perform k-fold cross-validation with early stopping and TensorBoard logging
cross_validation(prediction_paths=prediction_paths, k_folds=5, num_epochs=10, batch_size=32, learning_rate=0.001,
                 patience=3)
