from torchvision import transforms, datasets

root_dir = '/ws/data'

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

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
    n_cls=100,
    batch_size=256,
    size=32,
    num_workers=16,
    dataset='cifar100',
    data_folder=f"{root_dir}/cifar100",
    train=dict(transform=train_transform),
    val=dict(transform=val_transform),
)