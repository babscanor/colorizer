import sys
sys.path.append('../')
sys.path.append('./')

from data_embeddings import PortraitsDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from encoder import Network
from torch.nn import MSELoss 


if torch.cuda.is_available():
    device = torch.cuda(0)
else:
    device = torch.device("cpu")

batch_size = 30
dataset = PortraitsDataset('../training_data')
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True)

network = Network(batch_size).to(device)
network.double()
optimizer = optim.Adam(network.parameters(), lr = 0.01)
total_loss =0
total_correct = 0

for epoch in range(100):    
    print("this is epoch", epoch)
    for batch in dataloader:
        print("a batch")
        gray_image, colors, irv_2 = batch["gray_image"].to(device), batch["colored_image"].to(device), batch["IRV2"].to(device)
        preds = network(gray_image, irv_2)
        loss = MSELoss()
        loss = loss(torch.flatten(preds.squeeze(dim=0), start_dim=1), torch.flatten(colors.squeeze(dim=0),start_dim=1))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("epoch", epoch, "total_correct", total_correct, "loss", total_loss)

torch.save(network, ".model/model.pt")