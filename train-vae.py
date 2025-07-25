import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from constants import *
import pandas as pd
import numpy as np





class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + num_classes, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = torch.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim + num_classes, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, z, y):
        z = torch.cat([z, y], dim=-1)
        z = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))



def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, input_dim, num_classes)
        self.input_dim = input_dim

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL





def generate_fake_data(model, num_classes, batch_size=64, latent_dim=20, device='cuda'):
    model.eval()
    all_fake_data = []
    all_labels = []
    for class_idx in range(num_classes):
        label = torch.zeros(batch_size, num_classes).to(device)
        label[:, class_idx] = 1

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = model.decoder(z, label)

        all_fake_data.append(fake_data)
        all_labels.append(label)

    fake_data = torch.cat(all_fake_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return fake_data, labels




import time
import torch.optim as optim

def train_vae(model, train_loader, epochs=10, lr=1e-3,save_dir='generated_data', device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()


            recon_batch, mu, logvar = model(data, labels)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}, Time: {epoch_duration:.4f} seconds")
    torch.save(model.state_dict(), os.path.join(save_dir, 'vaemodel_'+DEFAULT_SET+'.pth'))
    print(f"Model saved to {save_dir}")
def save_generated_data(fake_data, fake_labels, save_dir='generated_data'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(fake_data, os.path.join(save_dir, 'fake_data_'+DEFAULT_SET+'.pth'))
    integer_labels = torch.argmax(fake_labels, dim=1)
    torch.save(integer_labels, os.path.join(save_dir, 'fake_labels_'+DEFAULT_SET+'.pth'))
    print(len(fake_data))
    fake_dataset = TensorDataset(fake_data, integer_labels)
    fake_data_loader = DataLoader(fake_dataset, batch_size=64, shuffle=True)
    torch.save(fake_data_loader, os.path.join(save_dir, 'fake_data_loader_'+DEFAULT_SET+'.pth'))
def main():
    num_batches=1
    if DEFAULT_SET == "Purchase100":
        input_dim = 600
        latent_dim = 20
        num_classes = 100
        path = PURCHASE100_PATH
        data_frame = pd.read_csv(path, header=None)

        # extract the label
        labels = torch.tensor(data_frame[LABEL_COL].to_numpy(), dtype=torch.int64).to(DEVICE)
        labels -= 1
        data_frame.drop(LABEL_COL, inplace=True, axis=1)
        # extract the data
        data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)
    elif DEFAULT_SET == "Location30":
        input_dim = 446
        latent_dim = 20
        num_classes = 30
        path = LOCATION30_PATH
        label_column = LABEL_COL
        data_frame = pd.read_csv(path, header=None)
        # extract the label
        labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
        labels -= 1
        data_frame.drop(label_column, inplace=True, axis=1)
        # extract the data
        data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)
    labels_onehot = torch.zeros((labels.size(0), num_classes), dtype=torch.float).to(DEVICE)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)
    print(labels_onehot.shape)
    print(labels_onehot[0])
    dataset = TensorDataset(data, labels_onehot)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(train_loader)
    print(len(train_loader))
    #train_loader = DataLoader(TensorDataset(dataset), batch_size=64, shuffle=True)
    vae_model = VAE(input_dim, latent_dim, num_classes)
    #train_vae(vae_model, train_loader, epochs=100, lr=1e-4, device=DEVICE)
    save_dir = 'generated_data'
    vae_model.load_state_dict(torch.load(os.path.join(save_dir, 'vaemodel_'+DEFAULT_SET+'.pth')))
    vae_model=vae_model.to(DEVICE)
    #fake_data, fake_labels = generate_fake_data(vae_model, num_classes=num_classes, batch_size=BATCH_SIZE,latent_dim=latent_dim, device=DEVICE)
    all_fake_data = []
    all_fake_labels = []
    for i in range(num_batches):
        starttime=time.time()
        print(i)
        fake_data, fake_labels = generate_fake_data(vae_model, num_classes=num_classes, batch_size=10,
                                                    latent_dim=latent_dim, device=DEVICE)
        endtime=time.time()
        all_fake_data.append(fake_data)
        all_fake_labels.append(fake_labels)
    print(endtime-starttime)
    all_fake_data = torch.cat(all_fake_data, dim=0)
    all_fake_labels = torch.cat(all_fake_labels, dim=0)
    save_generated_data(all_fake_data, all_fake_labels, save_dir='generated_data')
if __name__ == "__main__":

    main()
