import torch
import torchvision

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
target_classes = (3,5)

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class ClassSelector(torch.utils.data.sampler.Sampler):
    """Select target classes from the dataset"""
    def __init__(self, target_classes, data_source):
        self.mask = torch.tensor([1 if data_source[i][1] in target_classes else 0 for i in range(len(data_source))])
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

class ToBinary(object):
    """Convert Tensor  values binary values"""
    def __call__(self, sample):
        if sample.max() <=1:
            return sample.round()
        else:
            return (sample/sample.max()).round()

train_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               ToBinary(),
                             ]))

test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               ToBinary(),
                             ])),

train_loader = torch.utils.data.DataLoader(train_dataset,
  batch_size=batch_size_train, sampler = ClassSelector(target_classes, train_dataset))

test_loader = torch.utils.data.DataLoader(test_dataset,
  batch_size=batch_size_test, sampler = ClassSelector(target_classes, test_dataset))


if __name__ == "__main__":
    example_data = next(iter(train_loader))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[0][i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_data[1][i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

