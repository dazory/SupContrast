from torchvision import transforms, datasets

root_dir = '/ws/data'

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
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
    size=32,
    num_workers=16,
    dataset='cifar10',
    data_folder=f"{root_dir}/cifar",
    train=dict(transform=train_transform),
    val=dict(transform=val_transform),
)