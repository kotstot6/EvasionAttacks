
print()
print('CSE 598 Assignment 1: Part 2')
print('Author: Kyle Otstot')
print('-------------------------------')
print()

# Import libraries

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.autograd import Variable

# Parameters

parser = argparse.ArgumentParser(description='CSE 598 Assignment 1 Part 2')

# Reproducibility
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Adversarial attack
parser.add_argument('--attack', type=str, default='fgsm', choices={'fgsm', 'pgd'}, help='type of adversarial attack')
parser.add_argument('--epsilon', type=float, default=25/255, help='epsilon parameter for FGSM & PGD')
parser.add_argument('--alpha', type=float, default=25/255, help='alpha parameter for PGD')
parser.add_argument('--n_steps', type=int, default=2, help='number of steps for PGD')

# Load & save settings
parser.add_argument('--model_file', type=str, default='chosen_model.pt', help='model file to be loaded')
parser.add_argument('--n_saved', type=int, default=10, help='number of images to save')

args = parser.parse_args()

# Load test data

transform_data = transforms.Compose([transforms.ToTensor()])
test_data = datasets.FashionMNIST('data', download=True, train=False, transform=transform_data)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

label_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Define LeNet model

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        def pooling():
            return nn.AvgPool2d(kernel_size=2, stride=2)

        def activation():
            return nn.ReLU()

        self.main = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            activation(),
            pooling(),
            nn.Conv2d(6, 16, kernel_size=5),
            activation(),
            pooling(),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(400, 120),
            activation(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            activation(),
            nn.Dropout(0.2),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.main(x)

# Load model

model = Network()
model.load_state_dict(torch.load('models/' + args.model_file, map_location=torch.device('cpu')))
model.eval()

criterion = nn.CrossEntropyLoss()

# Define adversarial attacks

def FGSM(images, labels):

    images = Variable(images, requires_grad=True)
    output = model(images)
    loss = criterion(output, labels)
    loss.backward(retain_graph=True)

    grads = torch.sign(images.grad.data)
    adv_images = images + args.epsilon * grads

    return adv_images

def PGD(images, labels):

    X = images

    for i in range(args.n_steps):
        X = Variable(X, requires_grad=True)
        output = model(X)
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)

        grads = torch.sign(X.grad.data)
        X = clip(X + args.alpha * grads, images)

    return X

def clip(adv_images, images):
    return torch.min(torch.max(adv_images, images - args.epsilon), images + args.epsilon)

attack = FGSM if args.attack == 'fgsm' else PGD

print('-----Adversarial Attack-----')
print('Type:', args.attack)
print('Epsilon:', args.epsilon)
if args.attack == 'pgd':
    print('Alpha:', args.alpha)
    print('Number of steps:', args.n_steps)
print('----------------------------')
print()

# Compute attack metrics

for images, labels in test_loader:

    adv_images = attack(images, labels)

    output, adv_output = model(images), model(adv_images)

    def metrics(output):
        loss = float(criterion(output, labels))
        conf = F.softmax(output,dim=1).amax(dim=1).reshape(-1).detach().numpy()
        preds = output.argmax(dim=1).reshape(-1).detach().numpy()
        correct = (output.argmax(dim=1) == labels).reshape(-1).detach().numpy()
        return loss, conf, preds, correct

    loss, conf, preds, correct = metrics(output)
    adv_loss, adv_conf, adv_preds, adv_correct = metrics(adv_output)

    print('-----Metrics-----')
    print('Original loss:', loss)
    print('Adversarial loss:', adv_loss)
    print()
    print('Original accuracy:', np.mean(correct))
    print('Adversarial accuracy:', np.mean(adv_correct))
    print()

    def bayes(array1, array2):
        return np.sum(np.logical_and(array1, array2)) / np.sum(array1)

    print('% Correct --> Correct:', bayes(correct, adv_correct))
    print('% Correct --> Incorrect:', bayes(correct, np.logical_not(adv_correct)))
    print('% Incorrect --> Correct:', bayes(np.logical_not(correct), adv_correct))
    print('% Incorrect --> Incorrect:', bayes(np.logical_not(correct), np.logical_not(adv_correct)))
    print('------------------------')

    # Qualitative results

    if args.n_saved > 0:

        save_indices = np.random.choice(len(test_data), args.n_saved)

        path = 'images/' + args.attack + '-' + str(args.epsilon) + '-' + str(args.alpha) + '-' + str(args.n_steps)

        if os.path.isdir(path):
            os.system('rm -rf ' + path)

        os.mkdir(path)

        for index in save_indices:

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(images.reshape(-1,28,28)[index,:,:].detach().numpy(), cmap='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])

            orig_text = ('Original\n\n"' + label_names[preds[index]] +  '"\n'
                            + str(round(1000 * conf[index]) / 10) + '% confidence')
            ax1.set_xlabel(orig_text, fontsize=14)

            ax2.imshow(adv_images.reshape(-1,28,28)[index,:,:].detach().numpy(), cmap='gray')
            ax2.set_xticks([])
            ax2.set_yticks([])

            adv_text = ('Adversarial\n\n"' + label_names[adv_preds[index]] +  '"\n'
                            + str(round(1000 * adv_conf[index]) / 10) + '% confidence')
            ax2.set_xlabel(adv_text, fontsize=14)

            plt.savefig(path + '/' + str(index) + '.png')
