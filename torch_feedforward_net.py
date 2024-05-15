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
import warnings
from opacus import PrivacyEngine
from scipy.stats import lognorm

warnings.simplefilter("ignore")

if __name__ == '__main__':

    scaler = GradScaler()

    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def remove_inf_values(tensor):
        # Create a mask for non-infinite values
        mask = torch.isfinite(tensor)

        # Apply the mask to the tensor to remove infinite values
        tensor = tensor[mask]

        return tensor

    class CrossEntropyConfMask(nn.Module):
        def __init__(self, reduction='mean'):

            super(CrossEntropyConfMask, self).__init__()

            self.reduction = reduction

        def forward(self, input, target):

            # Create a mask to select the logit corresponding to the target class
            mask = torch.zeros_like(input)
            mask.scatter_(1, target.unsqueeze(1), 1)

            # Compute the log of the sum of the exps
            log_sum_exp = torch.logsumexp(
                input * (1 - mask), dim=1)

            # Compute the conf
            loss = torch.sum(mask * (input -
                                     log_sum_exp.unsqueeze(1)), dim=1)

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

            # Create a mask to select the logit corresponding to the target class
            mask = torch.zeros_like(input)
            mask.scatter_(1, target.unsqueeze(1), 1)

            # Compute the log of the sum of the exps
            log_sum_exp = torch.logsumexp(
                input * (1 - mask), dim=1)

            # Compute the conf
            loss = torch.sum(mask * (input -
                                     log_sum_exp.unsqueeze(1)), dim=1)

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

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    batch_size = 512
    randomized = True
    num_classes = 10
    num_epochs = 1
    weight_decay = 0.001
    lr_step = 0.1
    schedule = [30, 40]
    dropout_rate = 0
    lr = 0.1
    shuffle = True
    private = False
    epsilon = 50
    delta = 1e-5
    interval = 5

    if randomized == False:

        torch.manual_seed(0)
        np.random.seed(0)
        shuffle = False

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=5, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    trainloader0 = torch.utils.data.DataLoader(
        trainset, batch_size=2000, shuffle=False, num_workers=3, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2000, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    # Create an instance of the CNN model
    model = SimpleNet().cuda()

    # Define the loss function and optimizer
    criterion2 = CrossEntropyLossMask0().cuda()
    criterion1 = CrossEntropyConfMask0().cuda()
    criterion0 = CrossEntropyConfMask().cuda()
    criterion = CrossEntropyLossMask().cuda()
    criterion0_eval = CrossEntropyConfMask(reduction='sum').cuda()
    criterion_eval = CrossEntropyLossMask(reduction='sum').cuda()
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

    if private == True:

        privacy_engine = PrivacyEngine()

        model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=num_epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=1,
        )

    cnt = 0

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % interval == 0:

            running_loss = 0.0
            running_conf = 0.0
            running_test_loss = 0.0
            running_test_conf = 0.0
            correct = 0.0
            correct_test = 0.0
            total = 0.0
            total_test = 0.0

            model.eval()

            with torch.no_grad():

                for data in trainloader0:

                    images, labels = data[0].cuda(
                        non_blocking=True), data[1].cuda(non_blocking=True)

                    with autocast():

                        outputs = model(images)

                        loss = criterion_eval(outputs, labels)
                        conf = criterion0_eval(outputs, labels)

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

                        loss = criterion_eval(outputs, labels)
                        conf = criterion0_eval(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

                    running_test_conf += conf.item()
                    running_test_loss += loss.item()

            confs[cnt] = torch.tensor([int(epoch), running_conf / total])
            losses[cnt] = torch.tensor([int(epoch), running_loss / total])
            test_confs[cnt] = torch.tensor(
                [int(epoch), running_test_conf / total_test])
            test_losses[cnt] = torch.tensor(
                [int(epoch), running_test_loss / total_test])
            gen_confs[cnt] = torch.tensor([int(epoch), np.abs(
                running_test_conf / total_test - running_conf / total)])
            gen_losses[cnt] = torch.tensor([int(epoch), np.abs(
                running_test_loss / total_test - running_loss / total)])
            gen_acc[cnt] = torch.tensor([int(epoch), np.abs(
                100 * (1 - (correct_test / total_test) - (1 - (correct / total))))])
            train_accs[cnt] = torch.tensor([int(epoch), correct / total])
            test_accs[cnt] = torch.tensor(
                [int(epoch), correct_test / total_test])

            if epoch % 10 == 0 and epoch > 0:

                t1 = time.time()
                print(
                    f'Epoch [{epoch}/{num_epochs}], Train Accuracy: {100 * correct / total:.5f}, Test Accuracy: {100 * correct_test / total_test:.5f}')
                print(
                    f'Epoch [{epoch}/{num_epochs}], Train Loss: {running_loss / total:.5f}, Train Confidence: {running_conf / total:.5f}')
                print(
                    f'Epoch [{epoch}/{num_epochs}], Test Loss: {running_test_loss / total_test:.5f}, Test Confidence: {running_test_conf / total_test:.5f}')
                print(f'Time for 10 epochs: {t1 - t0}')
                t0 = time.time()

            cnt += 1

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

                loss = criterion_eval(outputs, labels)
                conf = criterion0_eval(outputs, labels)

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
    print(f'Loss on the train set: {running_loss / total:.5f}')
    print(
        f'Confidence on the train set: {running_conf / total: .5f}')

    correct_test = 0.0
    total_test = 0.0

    with torch.no_grad():

        for data in testloader:

            images, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)

            with autocast():

                outputs = model(images)

                loss = criterion_eval(outputs, labels)
                conf = criterion0_eval(outputs, labels)

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
    print(f'Loss on the test set: {running_test_loss / total_test:.5f}')
    print(
        f'Confidence on the test set: {running_test_conf / total_test: .5f}')

    print(
        f'Generalization error of CE loss: {np.abs(running_test_loss / total_test - running_loss / total)}')
    print(
        f'Generalization error of 0-1 loss: {np.abs(100 * (1 - (correct_test / total_test) - (1 - (correct / total))))}')

    with torch.no_grad():

        for param in model.parameters():
            '''
            print(param.cpu())
            print(torch.linalg.vector_norm(param))
            print(param.size())
            '''
            param.add_((torch.randn(param.size()).cuda() * 0.01 /
                       torch.linalg.vector_norm(param).cuda()))

    with torch.no_grad():

        for data in trainloader0:

            images, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)

            with autocast():

                outputs = model(images)

                loss = criterion_eval(outputs, labels)
                conf = criterion0_eval(outputs, labels)

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
    print(f'Loss on the train set: {running_loss / total:.5f}')
    print(
        f'Confidence on the train set: {running_conf / total: .5f}')

    correct_test = 0.0
    total_test = 0.0

    with torch.no_grad():

        for data in testloader:

            images, labels = data[0].cuda(
                non_blocking=True), data[1].cuda(non_blocking=True)

            with autocast():

                outputs = model(images)

                loss = criterion_eval(outputs, labels)
                conf = criterion0_eval(outputs, labels)

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
    print(f'Loss on the test set: {running_test_loss / total_test:.5f}')
    print(
        f'Confidence on the test set: {running_test_conf / total_test: .5f}')

    print(
        f'Generalization error of CE loss: {np.abs(running_test_loss / total_test - running_loss / total)}')
    print(
        f'Generalization error of 0-1 loss: {np.abs(100 * (1 - (correct_test / total_test) - (1 - (correct / total))))}')

print(f'Schedule: {schedule}')
print(f'Epochs: {num_epochs}')
print(f'Weight decay: {weight_decay}')
print(f'Dropout rate: {dropout_rate}')
print(f'Initial learning rate: {lr}')
print(f'Learning rate step: {lr_step}')
print(f'Random: {randomized}')
print(f'Batch size: {batch_size}')
print(f'Private: {private}, epsilon: {epsilon}, delta: {delta}')

fig, ax = plt.subplots()

ax.plot(gen_confs[:, 0].int(), gen_confs[:, 1], 'b-', label='Gen. Confidences')
ax.plot(gen_losses[:, 0].int(), gen_losses[:, 1], 'r-', label='Gen. Losses')
ax.plot(gen_acc[:, 0].int(), gen_acc[:, 1], 'g-', label='Gen. Accuracies')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Generalization')
ax.legend()

plt.grid()

fig1, ax1 = plt.subplots()

ax1.plot(test_accs[:, 0].int(), test_accs[:, 1], 'b-', label='Test Accuracy')
ax1.plot(train_accs[:, 0].int(), train_accs[:, 1],
         'r-', label='Train Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Value')
ax1.set_title('Accuracy')
ax1.legend()

plt.grid()

fig2, ax2 = plt.subplots()

x = np.linspace(np.min(np.concatenate((remove_inf_values(logits_test).cpu().numpy(), remove_inf_values(logits_train).cpu().numpy()), axis=0)),
                np.max(np.concatenate((remove_inf_values(logits_test).cpu().numpy(), remove_inf_values(logits_train).cpu().numpy()), axis=0)), 1000)

shape0, loc0, scale0 = lognorm.fit(remove_inf_values(logits_test).cpu())
shape1, loc1, scale1 = lognorm.fit(remove_inf_values(logits_train).cpu())

ax2.hist(remove_inf_values(logits_test).cpu(),
         bins=50, density=True, label='Test Logits')
ax2.hist(remove_inf_values(logits_train).cpu(),
         bins=50, density=True, label='Train Logits')
ax2.plot(x, lognorm.pdf(x, shape0, loc0, scale0))
ax2.plot(x, lognorm.pdf(x, shape1, loc1, scale1))

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
