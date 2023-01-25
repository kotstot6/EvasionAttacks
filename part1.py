
print()
print('CSE 598 Assignment 1: Part 1')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

# Import libraries

import argparse
import sys
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

# Parameters

parser = argparse.ArgumentParser(description='CSE 598 Assignment 1 Part 1')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Network
parser.add_argument('--pooling', type=str, default='avg', choices={'avg', 'max'}, help='type of pooling')
parser.add_argument('--activation', type=str, default='tanh', choices={'tanh', 'relu'}, help='type of activation function')
parser.add_argument('--dropout', type=float, default=0, help='dropout probability')

# Optimization
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'sgd'}, help='type of optimization')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay')

# Training
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--epoch_step', type=int, default=1, help='number of epochs between validation checkpoints')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of images')

# Save settings
parser.add_argument('--save_figure', action='store_true', help='saves a figure of train and validation loss over time')
parser.add_argument('--save_model', action='store_true', help='saves the model')
parser.set_defaults(save_figure=False, save_model=False)

args = parser.parse_args()

setting = '-'.join(sys.argv[1:]).replace('---', '--').replace('--', '-')

print('Setting:', setting)

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load the dataset and train, val, test splits

print("Loading datasets...")

transform_data = transforms.Compose([transforms.ToTensor()])

train_val_data = datasets.FashionMNIST('data', download=True, train=True, transform=transform_data)
train_data = Subset(train_val_data, range(50000))
val_data = Subset(train_val_data, range(50000,60000))
test_data = datasets.FashionMNIST('data', download=True, train=False, transform=transform_data)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print("Done!")

# Implement LeNet-5 network architecture

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        def pooling():
            return (nn.AvgPool2d(kernel_size=2, stride=2) if args.pooling == 'avg'
                    else nn.MaxPool2d(kernel_size=2, stride=2))

        def activation():
            return nn.Tanh() if args.activation == 'tanh' else nn.ReLU()

        self.main = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            activation(),
            pooling(),
            nn.Conv2d(6, 16, kernel_size=5),
            activation(),
            pooling(),
            nn.Flatten(start_dim=1),
            nn.Dropout(args.dropout),
            nn.Linear(400, 120),
            activation(),
            nn.Dropout(args.dropout),
            nn.Linear(120, 84),
            activation(),
            nn.Dropout(args.dropout),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.main(x)

# Initialize model, optimizer, & loss function

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Network().to(device)
best_model = None
best_acc, best_epoch = 0, 0
criterion = nn.CrossEntropyLoss()

optim_type = optim.Adam if args.optimizer == 'adam' else optim.SGD
optimizer = optim_type(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs*len(train_loader))

# Define evaluation function

def evaluate(model, data_loader):

    model.eval()
    correct, loss = 0, 0
    total, n_batches = len(data_loader.dataset), len(data_loader)

    with torch.no_grad():

        for images, labels in tqdm(data_loader):

            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels)
            correct += (output.argmax(dim=1) == labels).sum()

    return float(correct / total), float(loss / n_batches)

print('Start training...')

epochs, train_losses, val_losses = [], [], []

for epoch in range(1,args.n_epochs+1):

    print('----- Epoch ', epoch, '-----')

    model.train()

    for images, labels in tqdm(train_loader):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    if epoch % args.epoch_step == 0:
        train_acc, train_loss = evaluate(model, train_loader)
        print('Train Accuracy:', train_acc)
        print('Train Loss:', train_loss)

        val_acc, val_loss = evaluate(model, val_loader)
        print('Validation Accuracy:', val_acc)
        print('Validation Loss:', val_loss)

        print('Learning Rate:', scheduler.get_last_lr()[0])

        # Update best results
        if val_acc > best_acc:
            print('Saving new best model...')
            best_model = copy.deepcopy(model)
            best_acc = val_acc
            best_epoch = epoch

        # Store loss metrics
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print('--------------------')
    print()

print('Done!')

print('Evaluating...')
test_acc, _ = evaluate(best_model, test_loader)
print('********************')
print('Test Accuracy:', test_acc)
print('********************')

with open('part1_metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([setting, best_epoch, test_acc])

# If model should be saved

if args.save_model:
    torch.save(best_model.state_dict(), 'models/' + setting.replace('.', '') + '.pt')

# If figure should be saved

if args.save_figure:
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('figures/' + setting.replace('.', ''))
