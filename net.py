from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod

def plot_losses(losses, window=1):
    weights = np.ones(window) / window
    
    fig, ax = plt.subplots()
    ax.plot(
        np.convolve(losses, weights, mode='valid'),
        label="Loss"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    return fig, ax

class Model(ABC):
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        train_dataset,
        test_dataset,
        device=torch.device("cpu")
    ):
        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer

        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    @abstractmethod
    def get_loss(self, input):
        pass
    
    def train(self, epoch):
        self.model.train()
        losses = []

        with tqdm(self.train_loader, unit="batch", desc=f"Epoch #{epoch+1}") as pbar:
            for input, _ in pbar:
                input = input.to(self.device)

                loss = self.get_loss(input)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                pbar.set_postfix({
                    'Avg Loss': f'{sum(losses) / len(losses):.4f}'
                })

        return losses
    
    def test(self):
        self.model.eval()

        losses = []

        with torch.no_grad():
            with tqdm(self.test_loader, unit="batch", desc=f"Test Set") as pbar:
                for input, _ in pbar:
                    input = input.to(self.device)

                    loss = self.get_loss(input)

                    losses.append(loss.item())

                    pbar.set_postfix({
                        'Avg Loss': f'{sum(losses) / len(losses):.4f}'
                    })
        
        return losses


class AutoEncoder(Model):
    def __init__(self, model, optimizer, batch_size, train_dataset, test_dataset, device=torch.device("cpu")):
        super().__init__(model, optimizer, batch_size, train_dataset, test_dataset, device)

        self.loss_func = nn.MSELoss()

    def get_loss(self, images):
        reconstructed = self.model(images)
        return self.loss_func(reconstructed, images)

    def get_batch(self):
        with torch.no_grad():
            images, _ = next(iter(DataLoader(
                self.test_loader.dataset,
                batch_size=self.test_loader.batch_size,
                shuffle=True
            )))

            images = images.to(self.device)

            reconstructed = self.model(images)

        return images, reconstructed