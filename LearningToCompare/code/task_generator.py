# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def omniglot_character_folders():
    data_folder = '../datas/omniglot_resized/'

    #在os.path.join()方法中，如果某个路径的最后没有斜杠/，则会自动添加反斜杠
    character_folders = [os.path.join(data_folder, family+'/', character+'/') \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num, test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        
        #从文件夹中随机选取指定数目的文件夹
        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        #建立文件夹（最深直到character文件夹）的字典，用于在各个姿态的文字与1~n的序号之间建立映射
        labels = dict(zip(class_folders, labels))
        samples = dict()

        #print(class_folders)
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            #如果某个文字的文件夹中存在不是文件夹的元素（不是character文件夹的文件）
            #需要跳过之
            if not os.path.isdir(c):
                continue;
            #temp：character文件夹内的图片文件的绝对路径
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            #利用random.sample将这些图片打乱，集成成为列表加入到sample的字典中
            #以图片的character文件夹的路径为键
            samples[c] = random.sample(temp, len(temp))

            #将sample[c]的元素，即20张图片，分成训练集和测试集，添加到两个列表中
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]
        
        #根据各个图片的绝对路径（来自train_root），先得到其的character文件夹名
        #再根据之前建立的character文件夹名与序号的字典，得到某张图片的标签值
        #这个标签值对应着character文件夹
        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        #print('train_roots num: ', len(self.train_roots), 'class folder num: ', len(class_folders), 'test_roots num: ', len(self.test_roots))
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        #先将图片地绝对路径拆开，得到其除去最后一部分的路径再拼接
        #这个结果就是图片的character文件夹的路径
        result = '/'.join(sample.split('/')[:-1]) + '/'
        return result


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        #print('Index:', idx, ' Type:', type(idx))
        #print('Length:', len(self.image_roots))
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        #print(image)
        #image = np.array(image, dtype=np.float32)
        
        #在transform之前，有黑色点为0，白色点为1,
        #tramform之后，黑点为-10.9430，这是0-0.92206/0.08426的结果
        #transform之后，白点为0.9250，这是1-0.92206/0.08426的结果
        #由于toTensor，因此每一个向量都是28x28的
        if self.transform is not None:
            print(image)
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    
    #点的转换政策：1.旋转 2.变成向量 3.平均化
    dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    #print(type(task))
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

