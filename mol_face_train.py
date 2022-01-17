from argparse import ArgumentParser
from pathlib import Path
import pathlib
import math

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

from tqdm import tqdm, trange

from synt_dataset import SyntheticMoldrawingDataset, SyntheticMoldrawingDatasetConfig, get_background_imgs, MolImageCollator

class Projection(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

def cut_head(model):
    return torch.nn.Sequential(*(list(model.children())[:-1]))

class MolSimCLR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = cut_head(resnet50(pretrained=False))
        self.projection = Projection(input_dim=2048)  # TODO: this should not be hardcoded

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x).squeeze()
        projection = self.projection(embedding)
        return projection


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """

    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss



def main():
    parser = ArgumentParser(description='Script for training molecular face recognition')
    parser.add_argument('training_smiles', help='Path to a file listing SMILES to use for training', type=Path)
    parser.add_argument('dev_smiles', help='Path to file listing SMILES to use for early stopping', type=Path)
    parser.add_argument('test_smiles', help='Path to file listing  SMILES to use for testing', type=Path)
    parser.add_argument('--backgrounds', help='Path to directory containing images to use for the background', type=Path)
    parser.add_argument('--random-seed', help='Constant to seed the random number generator with', default=1729)

    args = parser.parse_args()

    with open(args.training_smiles) as fp:
        training_smiles = [line.strip() for line in fp]
    with open(args.dev_smiles) as fp:
        dev_smiles = [line.strip() for line in fp]
    with open(args.test_smiles) as fp:
        test_smiles = [line.strip() for line in fp]

    background_images = get_background_imgs(args.backgrounds)

    dataset_rng = np.random.default_rng(args.random_seed)

    dataset_config = SyntheticMoldrawingDatasetConfig(width=224, height=224)
    training_dataset = SyntheticMoldrawingDataset(config=dataset_config, smiles_list=training_smiles, rng=dataset_rng)
    dev_dataset = SyntheticMoldrawingDataset(config=dataset_config, smiles_list=dev_smiles, rng=dataset_rng)
    test_dataset = SyntheticMoldrawingDataset(config=dataset_config, smiles_list=test_smiles, rng=dataset_rng)
    collator = MolImageCollator(backgrounds=background_images, working_size=dataset_config.working_size)
    
    batch_size=64
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=collator, num_workers=12)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collator, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, num_workers=12)
    
    device = torch.device('cuda:0')
    model = MolSimCLR()
    model = model.to(device=device)
    temperature = 1

    optimizer = AdamW(model.parameters(), lr=1e-3)
    max_epochs = 1

    for epoch in range(max_epochs):
        model.train()
        for batch in tqdm(training_dataloader, desc='Training batch'):

            optimizer.zero_grad()
            batch = batch.to(device)
            n_pairs, _, *trailing_dims = batch.shape
            merged = batch.view(n_pairs*2, *trailing_dims)
            out = model(merged)
            n_images, *out_trailing_dims = out.shape
            paired = out.view(n_pairs, 2, *out_trailing_dims)
            out_1 = paired[:,0]  # The second axis is the pair-axis
            out_2 = paired[:,1]  # The second axis is the pair-axis

            loss = nt_xent_loss(out_1, out_2, temperature)

            loss.backward()
            print(f'Training loss: {loss.item()}')
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(dev_dataloader, desc='Dev batch'):
                in_1, in_2 = batch
                out_1 = model(in_1.to(device))
                out_2 = model(in_2.to(device))
                loss = nt_xent_loss(out_1, out_2, temperature)
                print(f'Dev loss: {loss.item()}')



    


if __name__ == '__main__':
    main()