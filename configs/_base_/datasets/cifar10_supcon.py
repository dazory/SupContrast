from torchvision import transforms, datasets

root_dir = '/ws/data'

size = 32
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
data = dict(
    n_cls=10,
    batch_size=256,
    size=size,
    num_workers=16,
    dataset='cifar10',
    data_folder=f"{root_dir}/cifar",
    train=dict(transform=train_transform),
    val=dict(transform=val_transform),
)