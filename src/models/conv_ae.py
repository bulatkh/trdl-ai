import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision import transforms


class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.tensor(np.random.normal(loc=0, scale=self.sigma, size=x.shape)).to(x)
        x = x + noise
        return x


def init_transforms(mean, std):
    train = transforms.Compose(
        [   
            transforms.Lambda(lambda x: (x - mean) / std)
        ]
    )
    test = transforms.Compose(
        [
            transforms.Lambda(lambda x: (x - mean) / std)
        ]
    )
    return train, test


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pooling_kernel, pooling_padding):
        super(Encoder, self).__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(pooling_kernel)
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(pooling_kernel)
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[2]),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(pooling_kernel)
        )

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsampling, stride=1, padding=1):
        super(Decoder, self).__init__()
        padding = int(kernel_size / 2)
        self.out_channels = out_channels
        self.decnn_block1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            # nn.Upsample(upsampling[0])
        )
        self.decnn_block2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            # nn.Upsample(upsampling[1])
        )
        self.decnn_block3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.Upsample(upsampling[2])
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], self.out_channels[-1], -1)
        x = self.decnn_block1(x)
        x = self.decnn_block2(x)
        x = self.decnn_block3(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, conv_out_size, latent_size):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(conv_out_size, latent_size)
        self.linear2 = nn.Linear(latent_size, conv_out_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Autoencoder(LightningModule):
    def __init__(self, in_channels, out_channels, latent_size, kernel_size=3, stride=1, padding=1, pooling_kernel=2, pooling_padding=0, max_len=166, metric_scheduler='loss', lr=0.001, optimizer_name='adam', noise_val=0, **kwargs):
        super(Autoencoder, self).__init__()
        # architecture
        self.name = 'cae'
        self.num_layers = len(out_channels)
        self.encoder = Encoder(in_channels, out_channels, kernel_size, stride, padding, pooling_kernel, pooling_padding)
        conv_out_size = max_len
        self.upsampling = [max_len]
        for _ in range(self.num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            # conv_out_size = int((conv_out_size + 2 * pooling_padding - (pooling_kernel - 1) - 1) / pooling_kernel + 1)
            self.upsampling.append(conv_out_size)
        conv_out_size = int(out_channels[-1] * conv_out_size)
        self.upsampling = self.upsampling[::-1][1:]

        self.bottleneck = Bottleneck(conv_out_size, latent_size)
        self.flatten = nn.Flatten()

        self.decoder = Decoder(in_channels, out_channels, kernel_size, self.upsampling, stride, padding)

        self.noise_val = noise_val
        # hyperparameters
        self.loss = nn.MSELoss()
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

    def _prepare_batch(self, batch):
        x = batch[0].permute(0, 2, 1)
        x = x.float()
        y = batch[1]
        return x, y
    
    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        x_noise = Jittering(self.noise_val)(x) if self.noise_val != 0 else x
        out = self(x_noise)
        loss = self.loss(out, x)

        self.log("train_loss", loss)
        return {
            'reconstructed': out,
            'x': x,
            'noised': x_noise,
            'loss': loss,
            'y': y
        }

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        return x, self(x), y

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = self._prepare_batch(batch)

        out = self(x)
        loss = self.loss(out, x)
        self.log(f"{prefix}_loss", loss)
        
        x_noise = x

        pairwise_loss = nn.MSELoss(reduce=False)(out, x).view(x.shape[0], -1).mean(dim=1)
        
        return {
            'reconstructed': out,
            'x': x,
            'noised': x_noise,
            'loss': loss,
            'pairwise_loss': pairwise_loss,
            'y': y
        }

    def configure_optimizers(self):
      return self._initialize_optimizer()

    def _initialize_optimizer(self):
        ### Add LR Schedulers
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['train', self.metric_scheduler])
            }
        }