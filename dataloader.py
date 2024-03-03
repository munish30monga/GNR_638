import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from prettytable import PrettyTable

class CUB_Dataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None, split_ratio=0.2):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.target2class_dict = {}
        self._load_metadata()
    
    def _load_metadata(self):
        images = pd.read_csv(self.dataset_dir / 'CUB_200_2011' / 'images.txt', sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(self.dataset_dir / 'CUB_200_2011' / 'image_class_labels.txt', sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(self.dataset_dir / 'CUB_200_2011' / 'train_test_split.txt', sep=' ', names=['img_id', 'is_training_img'])
        classes = pd.read_csv(self.dataset_dir / 'CUB_200_2011' / 'classes.txt', sep=' ', names=['class_id', 'class_name'], index_col=False)
        self.target2class_dict = pd.Series(classes.class_name.values, index=classes.class_id).to_dict()

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        if self.split == 'train':
            self.data = data[data.is_training_img == 1]
        else:  # 'test'
            self.data = data[data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = self.dataset_dir / 'CUB_200_2011' / 'images' / sample.filepath
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class CUB_DataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=32, num_workers=8):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = CUB_Dataset(self.dataset_dir, split='train', transform=self.transform)
        if stage in ('test', None):
            self.test_dataset = CUB_Dataset(self.dataset_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
def dataset_summary(dataset_dir):
    print('=> Dataset Summary:')
    # Initialize datasets to load their metadata  
    train_dataset = CUB_Dataset(dataset_dir, split='train')
    test_dataset = CUB_Dataset(dataset_dir, split='test')

    # Calculate number of samples for each split
    num_samples_train = len(train_dataset)
    num_samples_test = len(test_dataset)
    total_samples = num_samples_train + num_samples_test
    
    # Create and fill the table
    table = PrettyTable()
    table.field_names = ["Split", "Number of Samples", "Percentage"]
    
    # Calculate and add the percentage for each split
    percentage_train = (num_samples_train / total_samples) * 100
    percentage_test = (num_samples_test / total_samples) * 100
    
    table.add_row(["Train", num_samples_train, f"{percentage_train:.2f}%"])
    table.add_row(["Test", num_samples_test, f"{percentage_test:.2f}%"])
    
    print(table)
    
    num_classes = len(set(train_dataset.data['target']))
    print(f"Number of classes: {num_classes}")
    
    dataset_summary_dict = {
        'train_dataset': train_dataset,
        'test_dataset':test_dataset,
        'num_classes':num_classes
    }
    return dataset_summary_dict