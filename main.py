import torch
import pandas as pd
import warnings
from PIL import Image
import numpy as np
import json
import os
import cv2
import albumentations as A
import IPython.display
from matplotlib import pyplot as plt
import PIL
import seaborn as sns
import scipy
import sklearn
from sklearn import tree

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm_notebook, tqdm
import random

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm_notebook, tqdm
import random

warnings.filterwarnings('ignore')

class IdaoTrainDataset(Dataset):
    def __init__(self, root, index_path, label_converter_path, transforms=None, RGB=False):
        super().__init__()
        self._root = root
        self._index = pd.read_csv(index_path)
        #print("index = ", self._index)
        #print("len = ", len(self._index))
        self._index["gen_label"] = self._index.apply(lambda t : str(t.nuc_type) + str(t.energy), axis=1)
        #print("index[gen_label] = ", self._index["gen_label"][0])
        with open(label_converter_path, "r") as f:
            self._label_converter = json.load(f)

        self._transforms = transforms
        self._RGB = RGB
        self.label_counter = Counter(self._index.gen_label.values.tolist())

    def __getitem__(self, idx):
        img_name, nuc_type, energy, gen_label = self._index.iloc[idx]
        path = os.path.join(self._root, str(nuc_type), str(img_name))

        if self._RGB:
            img = Image.open(os.path.join(self._root, nuc_type, img_name)).convert("RGB")
        else:
            img = Image.open(os.path.join(self._root, nuc_type, img_name))

        img = np.array(img)
        if self._transforms:
            img = torch.FloatTensor(self._transforms(image=img)["image"])
        gen_label = self._label_converter[gen_label]
        nuc_type = self._label_converter[nuc_type]
        energy = self._label_converter[str(energy)]
        return {
            "image" : img.permute(2, 0, 1) if self._RGB else img,
            "nuc_type" : nuc_type,
            "energy" : energy,
            "gen_label" : gen_label,
            "path" : path
        }

    def sample_classes_and_plot(self):
        images = dict()

        for gen_label in ["ER{}".format(i) for i in [1, 3, 6, 10, 20, 30]] + ["NR{}".format(i) for i in [1, 3, 6, 10, 20, 30]]:
            img_name, nuc_type, energy, _ = self._index[self._index.gen_label == gen_label].sample(n=1).iloc[0]
            #print(self._index[self._index.gen_label == gen_label].sample(n=1).iloc[0])
            images[gen_label] = os.path.join(self._root, nuc_type, img_name)

        fig = plt.figure(figsize=(15, 15))
        #fig = plt.figure()
        #plt.subplots
        columns = 6
        rows = 2
        ax = []

        for i, gen_label in enumerate(["ER{}".format(i) for i in [1, 3, 6, 10, 20, 30]] + ["NR{}".format(i) for i in [1, 3, 6, 10, 20, 30]]):
            img_name = images[gen_label]
            image = Image.open(img_name)
            image = np.array(image)
            if self._transforms:
                image = self._transforms(image=image)["image"]
            ax.append(fig.add_subplot(rows, columns, i+1))
            t = EstimateImage(image)
            ax[-1].set_title(gen_label + " " + str(t[:len(t) - 1]))
            nimage = t[-1]
            #plt.plot(0, 0)
            plt.imshow(nimage)
            #print(gen_label, EstimateImage(image))
        plt.show()
        return

    def make_plot_for_all(self):
        images = dict()

        for gen_label in ["ER{}".format(i) for i in [3, 10, 30]] + ["NR{}".format(i) for i in [1, 6, 20]]:
            images[gen_label] = []
            for t in range(15):
                img_name, nuc_type, energy, _ = self._index[self._index.gen_label == gen_label].sample(n=1).iloc[0]
                images[gen_label].append(os.path.join(self._root, nuc_type, img_name))

        sz = 15
        fig = plt.figure(figsize=(sz, sz))
        #fig = plt.figure()
        #plt.subplots
        columns = 6
        rows = 2
        ax = []

        #figtogether = plt.figure(figsize=(50, 50))

        colors = ["c", "g", "b", "y", "m", "r"]
        #colors = ["b", "b", "b", "r", "r", "r"]
        for i, gen_label in enumerate(["ER{}".format(i) for i in [3, 10, 30]] + ["NR{}".format(i) for i in [1, 6, 20]]):
            points = []
            avg = [0, 0]
            for j in images[gen_label]:
                img_name = j
                image = Image.open(img_name)
                image = np.array(image)
                if self._transforms:
                    image = self._transforms(image=image)["image"]
                t = EstimateImage(image)
                points.append([t[0], t[1]])
                for k in [0, 1]:
                    avg[k] += t[k]

            for k in [0, 1]:
                avg[k] //= len(points)
            if len(ax) == 0:
                ax.append(fig.add_subplot(rows, columns, i+1))
            #ax[-1].set_title(gen_label + " " + str(avg))
            ax[-1].set_title("ER - blue, NR - red")
            ax[-1].set(xlabel = "count of yellow pixels (how much it is visible)", ylabel = "variance (circle is closer to 0 than no circle)")

            #nimage = t[-1]
            plt.xlim(0, 1000)
            plt.ylim(0, 500)
            #plt.plot(0, 0)
            for point in points:
                plt.plot(point[0], point[1], colors[i] + "o")

            #plt.imshow(nimage)
            #print(gen_label, EstimateImage(image))
        plt.show()
        return

    def make_plot_for_each(self):
        images = dict()

        for gen_label in ["ER{}".format(i) for i in [1, 3, 6, 10, 20, 30]] + ["NR{}".format(i) for i in [1, 3, 6, 10, 20, 30]]:
            img_name, nuc_type, energy, _ = self._index[self._index.gen_label == gen_label].sample(n = 1).iloc[0]
            images[gen_label] = (os.path.join(self._root, nuc_type, img_name))

        sz = 15
        fig = plt.figure(figsize=(sz, sz))
        #fig = plt.figure()
        #plt.subplots
        columns = 6
        rows = 2
        ax = []

        for i, gen_label in enumerate(["ER{}".format(i) for i in [1, 3, 6, 10, 20, 30]] + ["NR{}".format(i) for i in [1, 3, 6, 10, 20, 30]]):
            img_name = images[gen_label]
            image = Image.open(img_name)
            image = np.array(image)
            if self._transforms:
                image = self._transforms(image=image)["image"]
            ax.append(fig.add_subplot(rows, columns, i+1))
            t = EstimateImage(image)
            ax[-1].set_title(gen_label + " " + str(t[:len(t) - 1]))
            nimage = t[-1]
            #plt.plot(0, 0)
            plt.imshow(nimage)
            #print(gen_label, EstimateImage(image))
        plt.show()
        #
        return

    def __len__(self):
        return len(self._index)


class IdaoTestDataset(Dataset):
    def __init__(self, root, transforms=None, RGB=False):
        super().__init__()
        self._root = root
        self._img_names = os.listdir(root)
        self._transforms = transforms
        self._RGB = RGB


    def __getitem__(self, idx):
        img_name = self._img_names[idx]
        if self._RGB:
            img = Image.open(os.path.join(self._root, img_name)).convert("RGB")
        else:
            img = Image.open(os.path.join(self._root, img_name))
        tensor = np.array(img)
        if self._transforms:
            img = torch.FloatTensor(self._transforms(image=img)["image"])
        return {
                "image" : img.permute(2, 0, 1) if self._RGB else img,
                "path" : os.path.join(self._root, img_name)
        }

    def __len__(self):
        return len(self._img_names)

    def sample_and_plot(self):
        image_name = random.sample(self._img_names, k=1)[0]
        image = Image.open(os.path.join(self._root, image_name))
        image = np.array(image)
        if self._transforms:
            image = self._transforms(image=image)["image"]
        plt.imshow(image)
        plt.title(image_name)
        return

def show_img(img):
    plt.imshow(img)
    plt.title('my picture')
    plt.show()
    return

def distanceeuclid(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5**0.5

def EstimateImage(img):
    '''for i in img:
        print(i)
    print("end")'''
    #print(img)
    threshold = 105
    nimage = [k >= threshold for k in img]
    cnt_yellow = 0
    points = []
    koef = 0.5

    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] < threshold:
                continue
            cntneighbors = 0
            isneighbor = False
            radius = 5
            cells = 0
            for k in range(-radius, radius + 1):
                for l in range(-radius, radius + 1):
                    if k**2 + l**2 > radius**2:
                        continue

                    ni = i + k
                    nj = j + l
                    if ni < 0 or ni >= len(img):
                        continue
                    if nj < 0 or nj >= len(img[i]):
                        continue

                    cells += 1

                    if img[ni][nj] >= threshold:
                        cntneighbors += 1
                        isneighbor = True

            if cells != 0 and cntneighbors / cells >= koef:
                cnt_yellow += 1
                points.append([i, j])
            else:
                nimage[i][j] = 0

    diametr = 0
    variancefromcenter = 0
    variancemaxdistance = 0
    #alpha = 2
    if len(points) > 0:
        center = [sum(i[0] for i in points) / len(points), sum(i[1] for i in points) / len(points)]
        expectdistancefromcenter = sum(distanceeuclid(i, center) for i in points) / len(points)
        expectsqrdistancefromcenter = sum(distanceeuclid(i, center)**2 for i in points) / len(points)
        variancefromcenter = expectsqrdistancefromcenter - expectdistancefromcenter**2

        expectmaxdistance = 0
        expectsqrmaxdistance = 0
        for i in points:
            maxdistance = 0
            for j in points:
                dist = distanceeuclid(i, j)
                diametr = max(diametr, dist)
                maxdistance = max(maxdistance, dist)
            expectmaxdistance += maxdistance
            expectsqrmaxdistance += maxdistance**2

        expectmaxdistance /= len(points)
        expectsqrmaxdistance /= len(points)
        variancemaxdistance = expectsqrmaxdistance - expectmaxdistance**2

    return cnt_yellow, int(max(variancefromcenter, variancemaxdistance)), nimage


    return cnt_yellow, int(diametr), int(variancefromcenter), nimage


if __name__ == "__main__":
    print('Hello')
    sz = 224
    #exit(0)
    transforms = A.Compose([
    A.CenterCrop(width=sz, height=sz),
    A.Solarize(threshold=128, p=1.0)
    #A.ToFloat(10)
    ])
train_dataset = IdaoTrainDataset("idao_dataset/train",
                                 "utils_data/index.csv",
                                 "utils_data/label_converter.json",
                                 transforms=transforms, RGB=False)
test_dataset = IdaoTestDataset("idao_dataset/public_test", transforms=transforms)
#print(train_dataset.label_counter)
#train_dataset.sample_classes_and_plot()
#train_dataset.make_plot_for_each()
train_dataset.make_plot_for_all()
#print(len(AllTrain))
