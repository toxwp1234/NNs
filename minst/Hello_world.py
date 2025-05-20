import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt



"""

Podzielenie danych

Przygotowanie danych

DataLoader


Przygotować sieć

Zrobic funkcje uczenia i weryfikacji





"""
train_set = torchvision.datasets.MNIST("./data/", train=True, download=False)
check_set = torchvision.datasets.MNIST("./data/", train=False, download=False)


konwert = transforms.Compose([
    transforms.ToTensor()
])


###trwansformacja

train_set.transform = konwert
check_set.transform = konwert


##data loader

batch_size = 32

load_trained = DataLoader(train_set,batch_size,True)
load_test = DataLoader(train_set,batch_size,False)


input_size = 1 * 28 * 28
N_classes = 10


layers = []


layers = [

    nn.Flatten(),
    nn.Linear(input_size,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,N_classes)
        ]


model = nn.Sequential(*layers)
model.cuda()

## Funkcja straty

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

trainded_n = len(load_trained.dataset)
test_n = len(load_test.dataset)


def celnosc_batcha(output,true_y,size_of_batch):

    y_strzal = output.argmax(dim=1,keepdim=True)
    correct = y_strzal.eq(true_y.view_as(y_strzal)).sum().item()
    #print(true_y.view_as(y_strzal))
    return correct / size_of_batch


def train():

    loss = 0
    celnosc = 0

    model.train() #aktywacja trybu trenowania
    for x ,y in load_trained: 
        x,y=x.cuda() , y.cuda()
        output = model(x)
        optimizer.zero_grad() #czysci pamiec z poprzedniego gradientu
        batch_loss = loss_function(output,y)
        #backpropagation
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        celnosc += celnosc_batcha(output, y, trainded_n)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, celnosc))



def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in load_test:
            x,y=x.cuda() , y.cuda()
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += celnosc_batcha(output, y, test_n)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


epochs = 3

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

torch.save(model, 'model.pth')


