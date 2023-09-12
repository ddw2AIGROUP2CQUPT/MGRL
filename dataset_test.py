import numpy as np
import torch
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import pandas as pd
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_img_4(p_img,n_img,s_img):
    p_splitimg = []
    n_splitimg = []
    s_splitimg = []
    bool_mat = []
    size_img = p_img.size
    weight = int(size_img[0] // 2)
    height = int(size_img[1] // 2)
    for j in range(2):
        for k in range(2):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            p_region = p_img.crop(box)
            n_region = n_img.crop(box)            
            s_region = s_img.crop(box)
            s_region_jud = np.array(s_img.crop(box))
            if s_region_jud.sum()==0:
                bool_mat.append(0)
            else:
                bool_mat.append(1)
            p_splitimg.append(p_region)
            s_splitimg.append(s_region)
            n_splitimg.append(n_region)
    return p_splitimg,n_splitimg,s_splitimg,bool_mat

def split_img_4_test(p_img,s_img):
    p_splitimg = []
    s_splitimg = []
    bool_mat = []
    size_img = p_img.size
    weight = int(size_img[0] // 2)
    height = int(size_img[1] // 2)
    for j in range(2):
        for k in range(2):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            p_region = p_img.crop(box)
            s_region = s_img.crop(box)
            s_region_jud = np.array(s_img.crop(box))
            if s_region_jud.sum()==0:
                bool_mat.append(0)
            else:
                bool_mat.append(1)
            p_splitimg.append(p_region)
            s_splitimg.append(s_region)
    return p_splitimg,s_splitimg,bool_mat
def split_img_9(p_img,n_img,s_img):
    p_splitimg = []
    n_splitimg = []
    bool_mat = []
    s_splitimg = []
    size_img = p_img.size
    weight = int(size_img[0] // 3)
    height = int(size_img[1] // 3)
    for j in range(3):
        for k in range(3):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            p_region = p_img.crop(box)
            n_region = n_img.crop(box)            
            s_region = s_img.crop(box)
            s_region_jud = np.array(s_img.crop(box))
            if s_region_jud.sum()==0:
                bool_mat.append(0)
            else:
                bool_mat.append(1)
            p_splitimg.append(p_region)
            s_splitimg.append(s_region)
            n_splitimg.append(n_region)
    return p_splitimg,n_splitimg,s_splitimg,bool_mat

def split_img_9_test(p_img,s_img):
    p_splitimg = []
    bool_mat = []
    s_splitimg = []
    size_img = p_img.size
    weight = int(size_img[0] // 3)
    height = int(size_img[1] // 3)
    for j in range(3):
        for k in range(3):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            p_region = p_img.crop(box)
            s_region = s_img.crop(box)
            s_region_jud = np.array(s_region)
            if s_region_jud.sum()==0:
                bool_mat.append(0)
            else:
                bool_mat.append(1)
            p_splitimg.append(p_region)
            s_splitimg.append(s_region)
    return p_splitimg,s_splitimg,bool_mat

def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(320), transforms.CenterCrop(299)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

def get_part4_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(180), transforms.CenterCrop(170)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(170)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

def get_part9_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(180), transforms.CenterCrop(170)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(170)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

class MGRL_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode

        if hp.dataset_name == "Face-1000":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', '1000')
        elif hp.dataset_name == "Face-450":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', '450')

        self.train_photo_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'train','photo', '*')))
        self.train_sketch_paths = sorted(glob(os.path.join(self.root_dir, 'sketch', 'train', '*')))
        self.test_photo_paths = sorted(glob(os.path.join(self.root_dir, 'comp', 'test', 'photo', '*')))
        self.test_sketch_paths = sorted(glob(os.path.join(self.root_dir, 'sketch', 'test', '*')))
        
        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')
        self.train_transform_split4 = get_part4_transform('Train')
        self.test_transform_split4 = get_part4_transform('Test')
        self.train_transform_split9 = get_part9_transform('Train')
        self.test_transform_split9 = get_part9_transform('Test')

    def __getitem__(self, item):
        sample = {}
        if self.mode == 'Train':
            sketch_path = self.train_sketch_paths[item]
            positive_name = 'image' + sketch_path.split('/')[-1].split('_')[0][6:]
            positive_path = os.path.join(self.root_dir, 'comp', 'train', 'photo', positive_name + '.jpg')
            negative_path = self.train_photo_paths[randint(0, len(self.train_photo_paths) - 1)]
            negative_name = negative_path.split('/')[-1].split('.')[0]

            sketch_img = np.array(Image.open(sketch_path).convert('RGB'))
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            positive_img = Image.open(positive_path).resize((sketch_img.size[0],sketch_img.size[1])).convert('RGB')
            negative_img = Image.open(negative_path).resize((sketch_img.size[0],sketch_img.size[1])).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            positive_split4,negative_split4,sketch_split4,bool_mat_4=split_img_4(positive_img,negative_img,sketch_img)
            positive_split9,negative_split9,sketch_split9,bool_mat_9=split_img_9(positive_img,negative_img,sketch_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sketch_part4 =  [self.train_transform_split4(sketch) for sketch in sketch_split4]
            positive_part4 =  [self.train_transform_split4(positive) for positive in positive_split4]
            negative_part4 =  [self.train_transform_split4(negative) for negative in negative_split4]

            sketch_part9 =    [self.train_transform_split9(sketch) for sketch in sketch_split9]
            positive_part9 =  [self.train_transform_split9(positive) for positive in positive_split9]
            negative_part9 =  [self.train_transform_split9(negative) for negative in negative_split9]


            sample = {'sketch_img': sketch_img, 'sketch_part4': sketch_part4,'sketch_part9': sketch_part9,'sketch_path': sketch_path,
                    'positive_img': positive_img,'positive_part4':positive_part4, 'positive_part9':positive_part9,'positive_path': positive_path, 
                    'negative_img': negative_img,'negative_part4':negative_part4,'negative_part9':negative_part9, 'negative_path': negative_path,
                    'bool_mat_4':bool_mat_4,'bool_mat_9':bool_mat_9}

        elif self.mode == 'Test':
            
            sketch_path = self.test_sketch_paths[item]
            positive_name = 'image' + sketch_path.split('/')[-1].split('_')[0][6:]
            positive_path = os.path.join(self.root_dir, 'comp', 'test', 'photo', positive_name + '.jpg')
            sketch_img = np.array(Image.open(sketch_path).convert('RGB'))
            sketch_img = Image.fromarray(sketch_img).convert('RGB')
            positive_img = Image.open(positive_path).resize((sketch_img.size[0],sketch_img.size[1])).convert('RGB')


            positive_split4,sketch_split4,bool_mat_4=split_img_4_test(positive_img,sketch_img)
            positive_split9,sketch_split9,bool_mat_9=split_img_9_test(positive_img,sketch_img)
            bool_mat=[]
            bool_mat.extend(bool_mat_4)
            bool_mat.extend(bool_mat_9)
            # np_sketch=np.array(sketch_img)
            sketch_img = self.test_transform(sketch_img)
            positive_img = self.test_transform(positive_img)
            sketch_part = []
            sketch_part4 =  [self.test_transform_split4(sketch) for sketch in sketch_split4]
            positive_part4 =  [self.test_transform_split4(positive) for positive in positive_split4]
            sketch_part9 =    [self.test_transform_split9(sketch) for sketch in sketch_split9]
            positive_part9 =  [self.test_transform_split9(positive) for positive in positive_split9]

            sketch_part.extend(sketch_part4)
            sketch_part.extend(sketch_part9)           

            sample = {'sketch_img': sketch_img, 'sketch_part':sketch_part, 'sketch_path': sketch_path,
            'positive_img': positive_img,'positive_part4':positive_part4, 'positive_part9':positive_part9,'positive_path': positive_path,'bool_mat':bool_mat}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_sketch_paths)
        elif self.mode == 'Test':
            return len(self.test_sketch_paths)


def get_dataloader(hp):

    dataset_Test = MGRL_Dataset(hp, mode='Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=70, shuffle=False, num_workers=0)
    return dataloader_Test
