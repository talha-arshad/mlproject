from lightning.pytorch import LightningDataModule
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST as TorchvisionMNIST
from torchvision import transforms

INPUT_DIMS = (1, 28, 28)
OUTPUT_DIMS = (1, )
MAPPING = list(range(10))
VAL_SIZE = 5000
BATCH_SIZE = 32

class MNIST(LightningDataModule):
    def __init__(self, data_dir: str = "./data/datafiles", val_size=VAL_SIZE, batch_size=BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.val_size = val_size
        self.batch_size = batch_size
        self.input_dims = INPUT_DIMS
        self.output_dims = OUTPUT_DIMS
        self.mapping = MAPPING
        self.save_hyperparameters()

    def prepare_data(self):
        # download
        TorchvisionMNIST(self.data_dir, train=True, download=True)
        TorchvisionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full_dataset = TorchvisionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [len(full_dataset)-self.val_size, self.val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = TorchvisionMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict_dataset = TorchvisionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)

def main():
    dm = MNIST()
    dm.prepare_data()
    dm.setup(stage="fit")
    print('train_dataset: \n', dm.train_dataset)
    
if __name__ == '__main__':
    main()