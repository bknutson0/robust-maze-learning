import datetime
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter

from src.models.base_net import BaseNet
from src.utils.config import DEVICE, LOGGING_LEVEL, Hyperparameters
from src.utils.maze_loading import maze_loaders
from src.utils.model_loading import load_model
from src.utils.seeding import set_seed
from src.utils.testing import is_minimal_path

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def train_epoch(
    hyperparams: Hyperparameters,
    train_loader: DataLoader[TensorDataset],
    train_subset_loader: DataLoader[TensorDataset],
    validation_loader: DataLoader[TensorDataset],
    model: BaseNet,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    writer: SummaryWriter | None,
    epoch: int,
) -> tuple[float, float, float, float]:
    """Train the model for one epoch."""
    model.train()

    for batch_idx, (inputs, solutions) in enumerate(train_loader):
        optimizer.zero_grad()
        frac_epoch = epoch + batch_idx / len(train_loader)
        model.train_step(inputs, solutions, hyperparams, criterion, optimizer, frac_epoch, writer)

    # Compute training loss for the epoch, analogous to validation loss
    # Select random subset of training data with same size as validation data
    train_loss, train_acc = compute_average_loss_and_accuracy(hyperparams, train_subset_loader, model, criterion)
    logger.info(f'Computed training loss/accuracy:   {round(train_loss, 8):.4e}/{round(train_acc, 4):.4f}')

    # Perform validation once at the end of each epoch
    val_loss, val_acc = compute_average_loss_and_accuracy(hyperparams, validation_loader, model, criterion)
    logger.info(f'Computed validation loss/accuracy: {round(val_loss, 8):.4e}/{round(val_acc, 4):.4f}')

    # Update learning rate
    learning_rate_scheduler.step(val_acc)

    if writer is not None:
        writer.add_scalar('loss/train_epoch', train_loss, epoch + 1)
        writer.add_scalar('accuracy/train_epoch', train_acc, epoch + 1)
        writer.add_scalar('loss/validation_epoch', val_loss, epoch + 1)
        writer.add_scalar('accuracy/validation_epoch', val_acc, epoch + 1)

    return train_loss, train_acc, val_loss, val_acc


def compute_average_loss_and_accuracy(
    hyperparams: Hyperparameters, loader: DataLoader[TensorDataset], model: BaseNet, criterion: torch.nn.Module
) -> tuple[float, float]:
    """Compute the loss and accuracy of the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for inputs, solutions in loader:
            # Compute loss
            latents = model.input_to_latent(inputs)
            if model.name == 'it_net':
                latents = model.latent_forward(
                    latents, inputs, iters=hyperparams.iters, tolerance=hyperparams.tolerance
                )  # type: ignore
            else:  # model.name == 'base_net'
                latents = model.latent_forward(latents, inputs, iters=hyperparams.iters)  # type: ignore
            outputs = model.latent_to_output(latents)
            torch.use_deterministic_algorithms(False)
            loss = criterion(outputs, solutions).mean()
            torch.use_deterministic_algorithms(True)
            total_loss += loss.item()

            # Compute corrects
            predictions = model.output_to_prediction(outputs, inputs)
            corrects = is_minimal_path(inputs, predictions, solutions)
            total_correct += int(corrects.sum().item())
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(hyperparams: Hyperparameters) -> float:
    """Train the model with the given parameters."""
    # Set seed, start time, and initialize best validation accuracy
    set_seed(hyperparams.seed)
    start_time = time.time()
    best_val_acc = 0.0

    # Create run directory
    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    run_dir = os.path.join('models', hyperparams.model_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f'Starting training run {run_name} for {hyperparams.model_name} on {DEVICE}.')

    # Initialize results, and TensorBoard writer
    results: dict[str, Any] = {
        'run': run_name,
        'device': str(DEVICE),
        'best_epoch': None,
        'best_validation_accuracy': 0.0,
        'total_train_time': None,
        'hyperparameters': asdict(hyperparams),
        'epoch_summaries': {},
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    writer = SummaryWriter(os.path.join(run_dir, 'logs'))

    # Get maze loaders
    train_loader, train_subset_loader, validation_loader = maze_loaders(hyperparams)

    # Load the model
    model: BaseNet = load_model(hyperparams.model_name, hyperparams.pretrained)

    # Create the criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = getattr(torch.optim, hyperparams.optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=hyperparams.learning_rate)

    # Create the learning rate scheduler
    scheduler_class = getattr(torch.optim.lr_scheduler, hyperparams.learning_rate_scheduler_name)
    learning_rate_scheduler = scheduler_class(
        optimizer, mode='max', factor=hyperparams.reduction_factor, patience=hyperparams.patience
    )

    for epoch in range(hyperparams.epochs):
        # Train the model for an epoch
        train_loss, train_acc, val_loss, val_acc = train_epoch(
            hyperparams,
            train_loader,
            train_subset_loader,
            validation_loader,
            model,
            criterion,
            optimizer,
            learning_rate_scheduler,
            writer,
            epoch,
        )
        logger.info(f'Trained epoch {epoch + 1}/{hyperparams.epochs}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            results['best_validation_accuracy'] = round(best_val_acc, 4)
            results['best_epoch'] = epoch + 1

            # Save best model
            best_file = os.path.join(run_dir, 'best.pth')
            torch.save(model.state_dict(), best_file)

        # Save last model
        last_file = os.path.join(run_dir, 'last.pth')
        torch.save(model.state_dict(), last_file)

        # Save epoch summary to results
        current_time = time.time() - start_time
        results['total_train_time'] = time.strftime('%Hh %Mm %Ss', time.gmtime(current_time))
        results['epoch_summaries'][f'Epoch: {epoch + 1}/{hyperparams.epochs}'] = (
            f'Time: {time.strftime("%Hh %Mm %Ss", time.gmtime(current_time))}, '
            f'Train Loss: {round(train_loss, 8):.8f}, '
            f'Train Acc: {round(train_acc, 4):.4f}, '
            f'Val Loss: {round(val_loss, 8):.8f}, '
            f'Val Acc: {round(val_acc, 4):.4f}, '
            f'LR: {round(learning_rate_scheduler.get_last_lr()[0], 8):.8f}'
        )
        with open(os.path.join(run_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # Save checkpoints
        if (epoch + 1) % hyperparams.checkpoint_freq == 0:
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_file)

    # Save total training time
    results['total_train_time'] = time.strftime('%Hh %Mm %Ss', time.gmtime(current_time))
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    writer.close()

    logger.info(f'Trained {hyperparams.model_name} for {hyperparams.epochs} epochs.')

    return best_val_acc
