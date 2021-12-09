import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import torchvision



# define hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 50
num_classes = 2
num_epochs = 10
csv_file = 'C:/Users/Jenny/Desktop/dogs_vs_cats/train_dataset.csv'
root_dir = 'C:/Users/Jenny/Desktop/dogs_vs_cats/train'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224,224)),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# define custom dataset
class DogCatDataset(Dataset):
    '''
    load dataset from path
    '''
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0] + '.jpg')
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# save checkpoint function
def save_checkpoint(state, filename = f'dog_vs_cats_vgg16_{epoch}.pth.tar'):
    print("saving checkpoint")
    torch.save(state, filename)

# load dataset
dataset = DogCatDataset(csv_file,root_dir,transform)

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])

train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle=True)

test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle=True)

# Model
model = torchvision.models.vgg16(pretrained=False)

# for param in model.parameters():
#     param.requires_grad = False

model.to(device)

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# define optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the network
print("start training...")
for epoch in range(num_epochs):
    losses = []
    if epoch%1 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f'Loss at epoch {epoch} is {sum(losses)/len(losses)}')

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100} %')

    model.train()

print('checking accuracy on trianing set')
check_accuracy(train_loader, model)

print('checking accuracy on test set')
check_accuracy(test_loader, model)

# now it is printing loss at each epoch but did not print accuracy at each epoch. I would like to see accuracy at each epoch
# save the best model rather than the lastest model
# the saved model file type? I would like the model name have epoch info in it. like epoch 1 model, epoch 2 model etc.
