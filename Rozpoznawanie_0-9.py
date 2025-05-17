import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image 




model = torch.load('model.pth',weights_only=False).cuda()
print(model)

check_set = torchvision.datasets.MNIST("./data/", train=False, download=False)




konwerter = transforms.Compose([
    transforms.ToTensor()
])



check_set_did = check_set

check_set_did.transform = konwerter


vx,vy = check_set_did[2]



t=0
_input = t
while t <100 :
    
    vx,vy = check_set[t]
    

    strzal = model(vx.cuda()).argmax()
    print("\n\n---")
    print(f"Strzalem sieci jest {strzal}\n")
    print(f"Poprawna odpowiedz to --> {vy}\n\n")
    print("------")
   
    t+=1


