norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                    transforms.RandomRotation(10),
                                    transforms.RandomVerticalFlip(p=0.2),
                                     transforms.ToTensor(),
                                    norm])
valid_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     norm])
test_transform = transforms.Compose([transforms.Resize(224), 
                                      transforms.ToTensor(),
                                    norm])
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
valid_data = datasets.ImageFolder(valid_dir, transform = data_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
batch_size = 20
num_workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
loaders_scratch = {'train' : train_loader, 'valid' : valid_loader, 'test': test_loader}


import torch.nn as nn
import torch.nn.functional as F
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16,32,3, padding = 1)
        self.conv3 = nn.Conv2d(32,64,3, padding = 1)
        self.conv4 = nn.Conv2d(64,80,3, padding = 1)
        self.lin1 = nn.Linear(80*14*14,500)
        self.lin2 = nn.Linear(500,300)
        self.lin3 = nn.Linear(300,133)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)
 
    
    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
 
        x = x.view(-1, 80 * 14 * 14)
        x = (F.relu(self.lin1(x)))
        x = self.dropout(x)
        x = (F.relu(self.lin2(x)))
        x = self.dropout(x)
        x = self.lin3(x)
        
        return x
        

import torch.optim as optim

criterion_scratch = nn.CrossEntropyLoss()

optimizer_scratch = optim.SGD(model_scratch.parameters(), lr = 0.01)



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
            
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
    
    #average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format( epoch, train_loss, valid_loss ))

    if valid_loss <= valid_loss_min:
        torch.save(model.state_dict(), 'model_scratch.pt')
        valid_loss_min = valid_loss
        
    # return trained model
    return model
# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')
# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
