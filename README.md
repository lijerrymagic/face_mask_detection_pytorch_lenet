# Wearing of Face Masks Detection Using CNN (Customized LeNet-5)

## 1. How to execute the code
Before run, make sure having the following dependencies:
- Pytorch
- Pillow

If run on CPU, use:
`python3 project.py`

If run on GPU, use:
`python3 --use_cuda 1 project.py`

## 2. Code and model structure
&nbsp;
![CNN Model](/perf_images/cnn.png)
##### Model
&nbsp;

    class MaskDataSet(Dataset):
        def __init__(self, new_size, all_images, all_labels):
            super(MaskDataSet, self).__init__()
            self.images = all_images
            self.labels = all_labels

            # transform to torch tensor
            self.transform = transforms.Compose([
                lambda img_path: Image.open(img_path).convert("RGB"),
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
##### Custom dataset class extended Pytorch Dataset, use to store and process pictures and apply transforms from inputs.
&nbsp;

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
##### Customized LeNet-5 CNN network with inout size of 100 X 100 X 3
&nbsp;


    all_images = []
    all_labels = []
    for (dirpath, dirnames, filenames) in walk(path_with_mask):
        for filename in filenames:
            file_path = dirpath + "/" + filename
            if file_path.endswith("Store"):
                continue
            all_images.append(file_path)
            all_labels.append(0)
            num_no_mask += 1
    
##### Read and store file paths in lists from dataset containing pictures of people wearing masks.
&nbsp;

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
    train_loader, test_loader, classes = data_init()
    


##### Data initializtion function produce train loader and test loader from processed dataset.
&nbsp;

    epochs = 15
    net = Customized_LeNet5(2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
##### Setting network parameters and picking optimizers.
&nbsp;

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            print(predicted)
            print(targets)
            total += targets.size(0)
    
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
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
                print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))                 
    
    total_train = 0
    for epoch in range(epochs):
        train_start = time.perf_counter()
        train(epoch)
        train_end = time.perf_counter()
        total_train += train_end - train_start
        test(epoch)
        scheduler.step()
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, f'./checkpoint.pt')

##### Train and test in each epoch and save the model when finish training.
&nbsp;

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
##### When model training completed, we can load the model and perform some quick handy tests.
&nbsp;
## 3. Performance results
![Train Acc](/perf_images/train_accuracy.jpeg)
##### Tranining accuracy VS epochs.
&nbsp;
![Train Loss](/perf_images/train_loss.jpeg)
##### Tranining loss VS epochs.
&nbsp;

