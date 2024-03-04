import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, target

class CUB_DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_dir = Path(cfg.dataset_dir)
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.transforms = get_transforms()
        self.cfg = cfg
        
    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = CUB_Dataset(self.dataset_dir, split='train', transform=self.transforms['train'] if self.cfg.use_augm else self.transforms['val'])
        if stage in ('validate', None):
            self.val_dataset = CUB_Dataset(self.dataset_dir, split='test', transform=self.transforms['val'])
        if stage in ('test', None):
            self.test_dataset = CUB_Dataset(self.dataset_dir, split='test', transform=self.transforms['test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
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

def get_transforms():
    transforms = {
        'train': A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),                                        
            A.CoarseDropout(max_holes=4, max_height=15, max_width=15, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'test': A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    }
    return transforms
