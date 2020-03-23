import sys
sys.path.append('../')

from dataset import PortraitsDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from nn import Network
from torch.nn import MSELoss 

device = torch.device("cpu")

dataset = PortraitsDataset('../training_data')
dataloader = DataLoader(dataset, batch_size=5,
                        shuffle=True)

network = Network().to(device)
network.float()
optimizer = optim.Adam(network.parameters(), lr = 0.01)
total_loss =0
total_correct = 0
for epoch in range(100):    
    print("this is epoch", epoch)
    for batch in dataloader:
        print("a batch")
        gray_image, colors = batch["gray_image"], batch["colored_image"].to(device)
        preds = network(gray_image.float()).to(device)
        loss = MSELoss()
        loss = loss(torch.flatten(preds.squeeze(dim=0), start_dim=1), torch.flatten(colors.squeeze(dim=0),start_dim=1))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("epoch", epoch, "total_correct", total_correct, "loss", total_loss)

torch.save(network, ".model/model.pt")