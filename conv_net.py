import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':

    scaler = GradScaler()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    def remove_inf_values(tensor):
        # Create a mask for non-infinite values
        mask = torch.isfinite(tensor)

        # Apply the mask to the tensor to remove infinite values
        tensor = tensor[mask]

        return tensor

    class CrossEntropyConf(nn.Module):

        def __init__(self, reduction='mean'):

            super(CrossEntropyConf, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Apply softmax to the input
            softmax_output = nn.functional.softmax(input, dim=1)

            # Apply logit function to the softmax output
            logits = torch.special.logit(softmax_output, eps=1e-7)

            # Compute cross-entropy loss
            loss = nn.functional.nll_loss(
                logits, target, reduction=self.reduction)

            return loss

    class CrossEntropyConfMask(nn.Module):
        def __init__(self, reduction='mean'):

            super(CrossEntropyConfMask, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Apply softmax to the input
            softmax_output = nn.functional.softmax(input, dim=1)

            # Apply logit function to the softmax output
            logits = torch.special.logit(softmax_output, eps=1e-7)

            # Create a mask to ignore the terms that do not correspond to the target class
            mask = torch.zeros_like(logits)
            mask.scatter_(1, target.unsqueeze(1), 1)

            loss = -torch.sum(logits * mask, dim=1)

            if self.reduction == 'mean':

                loss = loss.mean()

            elif self.reduction == 'sum':

                loss = loss.sum()

            return loss

    class CrossEntropyConfMask0(nn.Module):

        def __init__(self, reduction='mean'):

            super(CrossEntropyConfMask0, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Apply softmax to the input
            softmax_output = nn.functional.softmax(input, dim=1)

            # Apply logit function to the softmax output
            logits = torch.special.logit(softmax_output, eps=1e-7)

            # Create a mask to ignore the terms that do not correspond to the target class
            mask = torch.zeros_like(logits)
            mask.scatter_(1, target.unsqueeze(1), 1)

            loss = -torch.sum(logits * mask, dim=1)

            return loss

    class CrossEntropyLossMask(nn.Module):

        def __init__(self, reduction='mean'):

            super(CrossEntropyLossMask, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Create a mask to ignore all values except the target label
            mask = torch.zeros_like(input)
            mask.scatter_(1, target.unsqueeze(1), 1)

            log_probs = torch.log_softmax(input, dim=1)
            loss = -torch.sum(log_probs * mask, dim=1)

            if self.reduction == 'mean':

                loss = loss.mean()

            elif self.reduction == 'sum':

                loss = loss.sum()

            return loss

    class CrossEntropyLossMask0(nn.Module):

        def __init__(self, reduction='mean'):

            super(CrossEntropyLossMask0, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Create a mask to ignore all values except the target label
            mask = torch.zeros_like(input)
            mask.scatter_(1, target.unsqueeze(1), 1)

            log_probs = torch.log_softmax(input, dim=1)
            loss = -torch.sum(log_probs * mask, dim=1)

            return loss
    '''
    # Define the CNN architecture
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.bn6 = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.fc1 = nn.Linear(256 * 4 * 4, 1024)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 10)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool2(x)
            
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool3(x)
            
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    '''

    class BasicBlock(nn.Module):

        def __init__(self, in_channels, out_channels, stride, dropout_rate):

            super(BasicBlock, self).__init__()

            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.dropout = nn.Dropout(dropout_rate)
            self.shortcut = nn.Identity()

            if stride != 1 or in_channels != out_channels:

                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):

            out = self.relu1(self.bn1(x))
            out = self.conv1(out)
            out = self.relu2(self.bn2(out))
            out = self.dropout(out)
            out = self.conv2(out)
            out += self.shortcut(x)

            return out

    class SimpleCNN(nn.Module):

        def __init__(self, num_classes=10, widen_factor=10, dropout_rate=0.3):

            super(SimpleCNN, self).__init__()

            self.in_channels = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.layer1 = self._make_layer(
                16 * widen_factor, 1, dropout_rate, 4)
            self.layer2 = self._make_layer(
                32 * widen_factor, 2, dropout_rate, 4)
            self.layer3 = self._make_layer(
                64 * widen_factor, 2, dropout_rate, 4)
            self.bn = nn.BatchNorm2d(64 * widen_factor)
            self.relu = nn.ReLU(inplace=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * widen_factor, num_classes)

        def _make_layer(self, out_channels, stride, dropout_rate, num_blocks):

            layers = []

            for i in range(num_blocks):

                if i == 0:

                    layers.append(BasicBlock(self.in_channels,
                                  out_channels, stride, dropout_rate))

                else:

                    layers.append(BasicBlock(
                        out_channels, out_channels, 1, dropout_rate))

                self.in_channels = out_channels

            return nn.Sequential(*layers)

        def forward(self, x):

            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    trainloader0 = torch.utils.data.DataLoader(
        trainset, batch_size=2000, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2000, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    num_classes = 10
    num_epochs = 200
    widen_factor = 1
    weight_decay = 0.0005
    lr_step = 0.1
    dropout_rate = 0.3
    schedule = [60, 120, 180]
    lr = 0.1
    # Create an instance of the CNN model
    model = SimpleCNN(num_classes=num_classes,
                      widen_factor=widen_factor, dropout_rate=dropout_rate).cuda()

    # Define the loss function and optimizer
    criterion2 = CrossEntropyLossMask0().cuda()
    criterion1 = CrossEntropyConfMask0().cuda()
    criterion0 = CrossEntropyConfMask().cuda()
    criterion = CrossEntropyLossMask().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, schedule, gamma=lr_step)
    a = torch.tensor(0.1)

    confs = torch.zeros((num_epochs, 2))
    losses = torch.zeros((num_epochs, 2))
    test_confs = torch.zeros((num_epochs, 2))
    test_losses = torch.zeros((num_epochs, 2))
    gen_confs = torch.zeros((num_epochs, 2))
    gen_losses = torch.zeros((num_epochs, 2))
    gen_acc = torch.zeros((num_epochs, 2))
    train_accs = torch.zeros((num_epochs, 2))
    test_accs = torch.zeros((num_epochs, 2))
    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    clip_threshold = 0.5
    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in progress_bar:

        model.train()

        running_loss = 0.0
        running_conf = 0.0
        running_test_loss = 0.0
        running_test_conf = 0.0
        correct = 0.0
        correct_test = 0.0
        total = 0.0
        total_test = 0.0

        for data in trainloader:

            inputs, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)
            optimizer.zero_grad()

            with autocast():

                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()

        with torch.no_grad():

            for data in trainloader0:

                images, labels = data[0].cuda(
                    non_blocking=True), data[1].cuda(non_blocking=True)

                with autocast():

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    conf = criterion0(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_conf += conf.item()
                running_loss += loss.item()

        with torch.no_grad():

            for data in testloader:

                images, labels = data[0].cuda(
                    non_blocking=True), data[1].cuda(non_blocking=True)

                with autocast():

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    conf = criterion0(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                running_test_conf += conf.item()
                running_test_loss += loss.item()

        if epoch % 10 == 0 and epoch > 0:

            t1 = time.time()
            print(
                f'Epoch [{epoch}/{num_epochs}], Train Accuracy: {100 * correct / total:.5f}, Test Accuracy: {100 * correct_test / total_test:.5f}')
            print(
                f'Epoch [{epoch}/{num_epochs}], Train Loss: {running_loss / len(trainloader0):.5f}, Train Confidence: {-running_conf / len(trainloader0):.5f}')
            print(
                f'Epoch [{epoch}/{num_epochs}], Test Loss: {running_test_loss / len(testloader):.5f}, Test Confidence: {-running_test_conf / len(testloader):.5f}')
            print(f'Time for 10 epochs: {t1 - t0}')
            t0 = time.time()

        confs[epoch] = torch.tensor([epoch, -running_conf / len(trainloader0)])
        losses[epoch] = torch.tensor([epoch, running_loss / len(trainloader0)])
        test_confs[epoch] = torch.tensor(
            [epoch, -running_test_conf / len(testloader)])
        test_losses[epoch] = torch.tensor(
            [epoch, running_test_loss / len(testloader)])
        gen_confs[epoch] = torch.tensor([epoch, np.abs(
            running_test_conf / len(testloader) - running_conf / len(trainloader0))])
        gen_losses[epoch] = torch.tensor([epoch, np.abs(
            running_test_loss / len(testloader) - running_loss / len(trainloader0))])
        gen_acc[epoch] = torch.tensor([epoch, np.abs(
            100 * (1 - (correct_test / total_test) - (1 - (correct / total))))])
        train_accs[epoch] = torch.tensor([epoch, correct / total])
        test_accs[epoch] = torch.tensor([epoch, correct_test / total_test])

    print('Training finished')

    # Evaluate the model on the test and train sets
    correct = 0.0
    total = 0.0
    logits_train = torch.empty(0).cuda()
    logits_test = torch.empty(0).cuda()
    losses_train = torch.empty(0).cuda()
    losses_test = torch.empty(0).cuda()
    running_loss = 0.0
    running_conf = 0.0
    running_test_loss = 0.0
    running_test_conf = 0.0

    model.eval()

    with torch.no_grad():

        for data in trainloader0:

            images, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)

            with autocast():

                outputs = model(images)

                loss = criterion(outputs, labels)
                conf = criterion0(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            logits_train = torch.cat(
                (logits_train, criterion1(outputs, labels)), 0)
            losses_train = torch.cat(
                (losses_train, criterion2(outputs, labels)), 0)

            running_conf += conf.item()
            running_loss += loss.item()

    print(f'Accuracy on the train set: {100 * correct / total:.5f}%')
    print(f'Loss on the train set: {running_loss / len(trainloader0):.5f}')
    print(f'Confidence on the train set: {
          -running_conf / len(trainloader0):.5f}')

    correct_test = 0.0
    total_test = 0.0

    with torch.no_grad():

        for data in testloader:

            images, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)

            with autocast():

                outputs = model(images)

                loss = criterion(outputs, labels)
                conf = criterion0(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            logits_test = torch.cat(
                (logits_test, criterion1(outputs, labels)), 0)
            losses_test = torch.cat(
                (losses_test, criterion2(outputs, labels)), 0)

            running_test_conf += conf.item()
            running_test_loss += loss.item()

    print(f'Accuracy on the test set: {100 * correct_test / total_test:.5f}%')
    print(f'Loss on the test set: {running_test_loss / len(testloader):.5f}')
    print(f'Confidence on the test set: {
          -running_test_conf / len(testloader):.5f}')

    print(f'Generalization error of CE loss: {np.abs(
        running_test_loss / len(testloader) - running_loss / len(trainloader0))}')
    print(f'Generalization error of 0-1 loss: {
          np.abs(100 * (1 - (correct_test / total_test) - (1 - (correct / total))))}')

print(f'Widen factor: {widen_factor}')
print(f'Schedule: {schedule}')
print(f'Epochs: {num_epochs}')
print(f'Weight decay: {weight_decay}')
print(f'Dropout rate: {dropout_rate}')
print(f'Initial learning rate: {lr}')
print(f'Learning rate step: {lr_step}')


fig, axs = plt.subplots()

axs.plot(confs[:, 0], confs[:, 1], 'y-', label='Tr. Confidences')
axs.plot(losses[:, 0], losses[:, 1], 'r-', label='Tr. Losses')
axs.plot(test_confs[:, 0], test_confs[:, 1], 'g-', label='Te.Confidences')
axs.plot(test_losses[:, 0], test_losses[:, 1], 'b-', label='Te. Losses')
axs.set_xlabel('Epoch')
axs.set_ylabel('Value')
axs.set_title('Confidences and Losses')
axs.legend()

plt.grid()

fig, ax = plt.subplots()

ax.plot(gen_confs[:, 0], gen_confs[:, 1], 'b-', label='Gen. Confidences')
ax.plot(gen_losses[:, 0], gen_losses[:, 1], 'r-', label='Gen. Losses')
ax.plot(gen_acc[:, 0], gen_acc[:, 1], 'g-', label='Gen. Accuracies')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Generalization')
ax.legend()

plt.grid()

fig1, ax1 = plt.subplots()

ax1.plot(test_accs[:, 0], test_accs[:, 1], 'b-', label='Test Accuracy')
ax1.plot(train_accs[:, 0], train_accs[:, 1], 'r-', label='Train Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Value')
ax1.set_title('Accuracy')
ax1.legend()

plt.grid()

fig2, ax2 = plt.subplots()

ax2.hist(remove_inf_values(logits_test).cpu(),
         bins=50, density=True, label='Test Logits')
ax2.hist(remove_inf_values(logits_train).cpu(),
         bins=50, density=True, label='Train Logits')
ax2.set_title('Logits')
ax2.legend()

plt.grid()

fig3, ax3 = plt.subplots()

ax3.hist(remove_inf_values(losses_test).cpu(),
         bins=50, density=True, label='Test Losses')
ax3.hist(remove_inf_values(losses_train).cpu(),
         bins=50, density=True, label='Train Losses')
ax3.set_title('Losses')
ax3.legend()

plt.grid()

plt.tight_layout()
plt.show()
