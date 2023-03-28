from Training.train import *
from parameters import *
from configuration import *


print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
model = TrainNetwork(device, model, optimizer, loss_function, train_loader, valid_loader, meta, ds).train()
# model = TrainNetwork(device, model, optimizer, loss_function, valid_loader, train_loader, meta, ds).train()
