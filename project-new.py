import cv2
import numpy as np
# import matplotlib.pyplot as plt
from os import walk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse
import time
import sys
# Customized mask data set reading
class MaskDataSet(Dataset):
    def __init__(self, new_size, all_images, all_labels):
        super(MaskDataSet, self).__init__()
        self.images = all_images
        self.labels = all_labels
        
        # transform to torch tensor
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img_path: Image.open(img_path).convert("RGB")),
            transforms.Resize((int(new_size), int(new_size))),
            transforms.ToTensor()
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image_tensor = self.transform(image_path)
        return image_tensor, label


    def __len__(self):
        return len(self.images)


class Customized_LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(Customized_LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=n_classes)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        probs = F.log_softmax(out, dim=1)
        return probs

# check if model exists
checkpoint_file = "./checkpoint.pt"
file_exists = os.path.exists(checkpoint_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if file_exists:
    net = Customized_LeNet5(2).to(device)
    state = torch.load(checkpoint_file)
    net.load_state_dict(state["net"])
    net.eval()
    test_images = ["./dataset/masked_face_dataset/yangzi/0_0_0.jpg", "./test1.jpeg", "./test2.jpeg"]
    test_labels = [1, 1, 0]
    test_set = MaskDataSet(100, test_images, test_labels)
    test_results = []
    for x, y in iter(test_set):
        x = x[None, :]
        print(x.size())
        outputs = net(x)
        _, predicted = outputs.max(1)
        if predicted[0] == 0:
            test_results.append("No mask!")
        else:
            test_results.append("Wearing mask!")
    print(test_results)
    sys.exit(0)

# dataset initialization
path_with_mask = "./dataset/face_dataset"
path_non_mask = "./dataset/masked_face_dataset"
all_images = []
all_labels = []
num_data_loader_workers = 0
num_no_mask = 0
num_mask = 0
# read all images paths from system
for (dirpath, dirnames, filenames) in walk(path_with_mask):
    for filename in filenames:
        file_path = dirpath + "/" + filename
        if file_path.endswith("Store"):
            continue
        # img_read = cv2.imread(file_path)
        # img_blob = img_to_blob(img_read)
        all_images.append(file_path)
        all_labels.append(0)
        num_no_mask += 1

for (dirpath, dirnames, filenames) in walk(path_non_mask):
    for filename in filenames:
        file_path = dirpath + "/" + filename
        if file_path.endswith("Store"):
            continue
        all_images.append(file_path)
        all_labels.append(1)
        num_mask += 1
n_samples = [num_no_mask, num_mask]
normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
loss_func = nn.CrossEntropyLoss(weight=torch.tensor(normed_weights))

print("length of all_images : {0}".format(len(all_images)))
print("length of all_labels : {0}".format(len(all_labels)))
        
# data init
def data_init():
    # create dataset
    full_dataset = MaskDataSet(100, all_images, all_labels)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=num_data_loader_workers)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=num_data_loader_workers)

    classes = (0, 1)
    return (train_loader, test_loader, classes)
print('######## Data initialization #########')
train_loader, test_loader, classes = data_init()

epochs = 15
net = Customized_LeNet5(2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# feed data into model
def train(epoch):
    print('\nEpoch: %d' % epoch)
    f.write('\nEpoch: %d' % epoch+"\n")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print(len(train_loader))
    training_time = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        train_start = time.perf_counter()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        training_time += time.perf_counter() - train_start

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        # print(predicted)
        # print(targets)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        if batch_idx == len(train_loader) - 1:
            train_loss_list.append(train_loss/(batch_idx+1))
            train_acc_list.append(correct/total)
            print(epoch,'Train Accuracy : ', train_loss/(batch_idx+1), 'Train Loss : ', correct/total)
            output = str(epoch) + ' Train Accuracy : ' + str(train_loss/(batch_idx+1)) + ' Train Loss : ' + str(correct/total) + "\n"
            f.write(output)
        print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("\n Epoch " + str(epoch) + " Training Time : " + str(training_time))
    write = "Epoch " + str(epoch) + " Training Time : " + str(training_time) + "\n"
    f.write(write)
    return training_time
# validate trained model
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
            if batch_idx == len(test_loader) - 1:
                test_loss_list.append(test_loss/(batch_idx+1))
                test_acc_list.append(correct/total)
                print(epoch,'Test Accuracy : ', test_loss/(batch_idx+1), 'Test Loss : ', correct/total)
                output = str(epoch) + 'Test Accuracy : ' + str(test_loss/(batch_idx+1)) + 'Test Loss : ' + str(correct/total)  + "\n"
                f.write(output)
            print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))             
f = open("output.out", "w")
total_train = 0
train_loss_list = []
train_acc_list = []
train_time_list = []
test_acc_list = []
test_loss_list = []
for epoch in range(epochs):
    f.write("*************Training started*************")
    print("*************Training started*************")
    oneEpochTime = train(epoch)
    train_time_list.append(oneEpochTime)
    total_train += oneEpochTime
    f.write("*************Testing started*************")
    print("*************Testing started*************")
    test(epoch)
    scheduler.step()
print('Saving state information..')
state = {
    'net': net.state_dict(),
}
torch.save(state, f'./checkpoint.pt')
print(train_loss_list)
print(train_acc_list)
print(test_loss_list)
print(test_acc_list)
print(train_time_list)
print(sum(train_time_list))

train_loss = ""
for l in train_loss_list:
    train_loss += str(l) + " "
f.write("Train Loss :" + train_loss + "\n")

train_acc = ""
for l in train_acc_list:
    train_acc += str(l) + " "
f.write("Train Accuracy :" + train_acc+ "\n")

test_loss = ""
for l in test_loss_list:
    test_loss += str(l) + " "
f.write("Test Loss :" + test_loss+ "\n")

test_acc = ""
for l in test_acc_list:
    test_acc += str(l) + " "
f.write("Test Accuracy :" + test_acc+ "\n")

train_time = ""
for l in train_time_list:
    train_time += str(l) + " "
f.write("Train Time :" + train_time+ "\n")
f.write("Total Train Time is " + str(sum(train_time_list)))

f.close()