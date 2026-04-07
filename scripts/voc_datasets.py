import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.datasets import VOCSegmentation


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

NUM_CLASSES = 21

IGNORE_INDEX = 255

class PascalVOCSegDataset(Dataset):
    def __init__(self, root:str, img_set:str, img_size:int):
        self.root = root
        self.img_set = img_set
        self.img_size = img_size

        self.image_transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToTensor()
        ])

        self.mask_transform = v2.Compose([
            v2.Resize((img_size,img_size),
            interpolation = v2.InterpolationMode.NEAREST),
            v2.PILToTensor()
            ])

        self.dataset = VOCSegmentation(
            root=root,
            year="2007",
            image_set=img_set,
            download=True
        )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, mask = self.dataset[index]

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.as_tensor(mask, dtype=torch.long)

        if mask.ndim == 3:
            mask = mask.squeeze(0)
        
        return image, mask


def build_voc_datasets(root:str, img_size:int):
    train_set = PascalVOCSegDataset(
        root = root,
        img_set = 'train',
        img_size=img_size

    )

    val_set = PascalVOCSegDataset(
        root = root,
        img_set = 'val',
        img_size = img_size
    )

    return train_set, val_set
