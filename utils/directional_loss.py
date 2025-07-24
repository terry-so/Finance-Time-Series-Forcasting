import torch
import torch.nn as nn
import numpy as np

class DirectionalLoss(nn.Module):
    """
    Directional Loss for time series forecasting.
    Penalizes predictions that have wrong directional movement.
    """
    def __init__(self, alpha=0.5, beta=1.0, base_loss='mse'):
        super(DirectionalLoss, self).__init__()
        self.alpha = alpha  # Weight for base loss
        self.beta = beta    # Weight for directional loss
        self.base_loss = base_loss
        
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss()
        else:
            self.base_criterion = nn.MSELoss()
    
    def forward(self, pred, true, last_values=None):
        """
        Args:
            pred: [batch_size, pred_len, features] - predictions
            true: [batch_size, pred_len, features] - ground truth
            last_values: [batch_size, features] - last known values for direction calculation
        """
        # Base loss (MSE/MAE)
        base_loss = self.base_criterion(pred, true)
        
        # Directional loss
        if last_values is not None:
            # Calculate direction accuracy
            directional_loss = self._calculate_directional_loss(pred, true, last_values)
        else:
            # If no last values, use period-to-period direction
            directional_loss = self._calculate_period_directional_loss(pred, true)
        
        # Combined loss
        total_loss = self.alpha * base_loss + self.beta * directional_loss
        return total_loss
    
    def _calculate_directional_loss(self, pred, true, last_values):
        """Calculate directional loss using last known values"""
        batch_size, pred_len, features = pred.shape
        
        # Expand last_values to match prediction shape
        last_expanded = last_values.unsqueeze(1).expand(-1, pred_len, -1)
        
        # Calculate actual and predicted directions
        true_direction = torch.sign(true - last_expanded)
        pred_direction = torch.sign(pred - last_expanded)
        
        # Directional accuracy (1 if same direction, 0 if opposite)
        direction_correct = (true_direction * pred_direction > 0).float()
        
        # Loss is 1 - accuracy (so we minimize it)
        directional_loss = 1.0 - direction_correct.mean()
        
        return directional_loss
    
    def _calculate_period_directional_loss(self, pred, true):
        """Calculate directional loss using period-to-period changes"""
        if pred.shape[1] < 2:  # Need at least 2 time steps
            return torch.tensor(0.0, device=pred.device)
        
        # Calculate period-to-period changes
        true_diff = true[:, 1:, :] - true[:, :-1, :]
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        
        # Calculate directions
        true_direction = torch.sign(true_diff)
        pred_direction = torch.sign(pred_diff)
        
        # Directional accuracy
        direction_correct = (true_direction * pred_direction > 0).float()
        
        # Loss is 1 - accuracy
        directional_loss = 1.0 - direction_correct.mean()
        
        return directional_loss

class WeightedDirectionalLoss(nn.Module):
    """
    Enhanced directional loss with magnitude weighting
    """
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1, base_loss='mse'):
        super(WeightedDirectionalLoss, self).__init__()
        self.alpha = alpha  # Weight for base loss
        self.beta = beta    # Weight for directional loss
        self.gamma = gamma  # Weight for magnitude penalty
        
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss()
        else:
            self.base_criterion = nn.MSELoss()
    
    def forward(self, pred, true, last_values=None):
        # Base loss
        base_loss = self.base_criterion(pred, true)
        
        # Directional loss with magnitude weighting
        if last_values is not None:
            directional_loss = self._weighted_directional_loss(pred, true, last_values)
        else:
            directional_loss = self._weighted_period_directional_loss(pred, true)
        
        total_loss = self.alpha * base_loss + self.beta * directional_loss
        return total_loss
    
    def _weighted_directional_loss(self, pred, true, last_values):
        batch_size, pred_len, features = pred.shape
        last_expanded = last_values.unsqueeze(1).expand(-1, pred_len, -1)
        
        # Calculate changes and directions
        true_change = true - last_expanded
        pred_change = pred - last_expanded
        
        true_direction = torch.sign(true_change)
        pred_direction = torch.sign(pred_change)
        
        # Weight by magnitude of actual change
        magnitude_weight = torch.abs(true_change)
        magnitude_weight = magnitude_weight / (magnitude_weight.mean() + 1e-8)  # Normalize
        
        # Directional accuracy
        direction_correct = (true_direction * pred_direction > 0).float()
        
        # Weighted directional loss
        weighted_loss = magnitude_weight * (1.0 - direction_correct)
        directional_loss = weighted_loss.mean()
        
        return directional_loss
    
    def _weighted_period_directional_loss(self, pred, true):
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        true_diff = true[:, 1:, :] - true[:, :-1, :]
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        
        true_direction = torch.sign(true_diff)
        pred_direction = torch.sign(pred_diff)
        
        # Weight by magnitude
        magnitude_weight = torch.abs(true_diff)
        magnitude_weight = magnitude_weight / (magnitude_weight.mean() + 1e-8)
        
        direction_correct = (true_direction * pred_direction > 0).float()
        weighted_loss = magnitude_weight * (1.0 - direction_correct)
        
        return weighted_loss.mean()