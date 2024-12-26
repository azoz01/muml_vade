import pytorch_lightning as pl
from torch.utils.data import DataLoader

from vade.data import DATASETS
from vade.device import DEVICE
from vade.model import AE, VADE, VAE

model = VADE([784, 256, 128], 10).to(DEVICE)
model = model.double()
dataset = DATASETS["MNIST"]
train_loader = DataLoader(dataset["train"], batch_size=32)
test_loader = DataLoader(dataset["test"], batch_size=32)
trainer = pl.Trainer(max_epochs=3, check_val_every_n_epoch=1)
trainer.fit(model, train_loader, test_loader)
