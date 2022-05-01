from datetime import datetime

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import RSDataset
from model import get_model
from test import test
import numpy as np

NUM_EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 0.0015
SPLIT_RATIO = np.array([0.7, 0.1, 0.2])


def train_(num_epochs, learning_rate, batch_size):
    pretrain_model, model = get_model()
    cudnn.benchmark = True
    device = 'cuda'

    dataset = RSDataset('UCMerced_LandUse/Images')
    split_num = (SPLIT_RATIO * len(dataset)).astype(np.int64)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, split_num)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size * 4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4)

    criticizer = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1 / 3)
    writer = SummaryWriter(f'./runs/{datetime.now().strftime("%m-%d_%H-%M")}')

    accuracy = [0, ]
    for epoch in range(num_epochs):
        model.train()
        pretrain_model.eval()
        program_bar = tqdm(total=len(train_dataloader), leave=False)
        for i, (images, labels) in enumerate(train_dataloader, start=1):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                x = pretrain_model(images)
                x = x.detach()
            y = model(x)
            loss = criticizer(y, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            program_bar.update()

            if i % 10 == 0:
                iter_times = epoch * len(train_dataloader) + i
                program_bar.set_description_str(f'Epoch:{epoch}')
                program_bar.set_postfix_str(f'Loss:{loss.item()}')
                writer.add_scalar('training loss', loss.item(), iter_times)
        scheduler.step()

        current_acc = test((pretrain_model, model), valid_dataloader)
        writer.add_scalar('Validation acc', current_acc, epoch)

        if current_acc > max(accuracy):
            if current_acc > 92:
                current_acc = test((pretrain_model, model), test_dataloader)
                model.eval()
                torch.save(model,
                           f'models/{current_acc:.6f}_{datetime.now().strftime("%m-%d_%H-%M")}.pth')
                tqdm.write("save model")
        tqdm.write(f'Epoch:{epoch}  Validation_Acc={current_acc:.4f}%')
        accuracy.append(current_acc)
    writer.close()


if __name__ == '__main__':
    train_(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)
