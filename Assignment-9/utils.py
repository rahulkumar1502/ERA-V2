from torchvision import datasets
import albumentations
from albumentations import HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch import ToTensorV2

class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

class LoadDataset():

  def __init__(self):
    pass

  def getTransforms(self):
    train_transforms = albumentations.Compose(
        [
            HorizontalFlip(),
            ShiftScaleRotate(),
            CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.49139968,0.48215841,0.44653091), mask_fill_value = None),
            albumentations.Normalize((0.49139968,0.48215841,0.44653091), (0.24703223,0.24348513,0.26158784)),
            ToTensorV2(),
        ]
    )

    test_transforms = albumentations.Compose(
        [
            albumentations.Normalize((0.49139968,0.48215841,0.44653091), (0.24703223,0.24348513,0.26158784)),
            ToTensorV2(),
        ]
    )
    return train_transforms,test_transforms
  
  def getData(self):

    train_transforms,test_transforms = self.getTransforms()
    train = Cifar10SearchDataset('./data', train=True, download=True, transform=train_transforms)
    test = Cifar10SearchDataset('./data', train=False, download=True, transform=test_transforms)
    print('Train and Test data loaded')
    return train,test


    