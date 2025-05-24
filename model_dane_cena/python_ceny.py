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




"""

Ten plik poświęcam uczeniem sieci mówieniem mi czy trend pomiędzy cenami oddalonymi o 100 dni jest trend rosnący czy malejący.


podział danych.

"6364" --> tyle mam danych

podzele je 60/40.




"""


def cut_to_N(N,lista):
#funckja dzieli podaną liste na liste składającą sie z list mniejszych o długości N
   
    lenght = len(lista)
    if(N>lenght):print("błedne dane N>len(list)")

    cut_lista : list =[]
    flaga = True
    t=0
    
    while (t+1)*N<lenght:
        cut_lista.append(lista[N*(t):N*(t+1)])
        t+=1
    return cut_lista







dane_file = open(r"X:\programowanie\c++\NNs\model_dane_cena\data.csv", "+r")

#przewijam przez każda linikę i tworze sobie liste cen

full_set_prices : list = []

for line in dane_file:
    full_set_prices.append(float(line.split(",")[1]))


lenght_of_Full_set = len(full_set_prices)


#rozdzielenie danych | na testowy i sprawdzający
test_set = full_set_prices[:int(lenght_of_Full_set*0.6)]
valid_set = full_set_prices[int(lenght_of_Full_set*0.6)+1:]

N = 100 #dziele listy na podlisty o długości N

test_set = cut_to_N(N,test_set)
valid_set = cut_to_N(N,valid_set)


#transformacja danych w tensory do feedowania

test_set = torch.FloatTensor(test_set)
valid_set = torch.FloatTensor(valid_set)



set_number : int = len(test_set) # licze ile mam danych do nauki







print(test_set[1])

