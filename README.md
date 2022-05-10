# HMPL Project - Wearing of Face Masks Detection Using CNN

## 1. How to execute the code
Before run, make sure having the following dependencies:
- Pytorch
- Pillow

If run on CPU, use:
`python3 project.py`

If run on GPU, use:
`python3 --use_cuda 1 project.py`

## 2. Code structure
`class MaskDataSet(Dataset):
...`
##### Custom dataset class extended Pytorch Dataset, use to store and process pictures and apply transforms from inputs.
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
##### When model training completed, we can load the model and perform some quick handy tests..
&nbsp;
## 3. Performance results
![Train Loss](/perf_images/train_loss.png)
