import os
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class DomainNetReal(DatasetBase):
    """DomainNet-Painting.

    This dataset is used for testing only.
    """

    dataset_dir = "domainnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "real")

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = ImageNet.read_classnames(text_file)
        label2classnames = json.load(
            open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/label2class.json'))
        # self.classnames = list(label2classnames.values())
        self._lab2cname = label2classnames
        self._classnames = list(label2classnames.values())

        data = self.read_data()

        super().__init__(train_x=data, test=data)
    
    def read_data(self):
        image_dir = '/home/manogna/TTA/PromptAlign/data/domainnet'
        img_files = open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/real_list.txt', 'r').readlines()
        items = []
        for line in img_files:
            path, label = line.split(' ')
            label = label.strip()
            impath = os.path.join(image_dir, path)
            item = Datum(impath=impath, label=int(label), classname=self._lab2cname[label])
            items.append(item)
        return items


@DATASET_REGISTRY.register()
class DomainNetClipart(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "domainnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "clipart")

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = ImageNet.read_classnames(text_file)
        label2classnames = json.load(
            open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/label2class.json'))
        # self.classnames = list(label2classnames.values())
        self._lab2cname = label2classnames
        self._classnames = list(label2classnames.values())

        data = self.read_data()

        super().__init__(train_x=data, test=data)

    # def read_data(self):
    #     image_dir = self.image_dir
    #     folders = listdir_nohidden(image_dir, sort=True)
    #     folders = [f for f in folders if f not in TO_BE_IGNORED]
    #     items = []

    #     for label, folder in enumerate(folders):
    #         imnames = listdir_nohidden(os.path.join(image_dir, folder))
    #         classname = folder #classnames[folder]
    #         for imname in imnames:
    #             impath = os.path.join(image_dir, folder, imname)
    #             item = Datum(impath=impath, label=label, classname=classname)
    #             items.append(item)

    #     return items
    
    def read_data(self):
        image_dir = '/home/manogna/TTA/PromptAlign/data/domainnet'
        img_files = open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/clipart_list.txt', 'r').readlines()
        items = []
        for line in img_files:
            path, label = line.split(' ')
            label = label.strip()
            impath = os.path.join(image_dir, path)
            item = Datum(impath=impath, label=int(label), classname=self._lab2cname[label])
            items.append(item)
        return items


@DATASET_REGISTRY.register()
class DomainNetPainting(DatasetBase):
    """DomainNet-Painting.

    This dataset is used for testing only.
    """

    dataset_dir = "domainnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "painting")

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = ImageNet.read_classnames(text_file)
        label2classnames = json.load(
            open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/label2class.json'))
        # self.classnames = list(label2classnames.values())
        self._lab2cname = label2classnames
        self._classnames = list(label2classnames.values())

        data = self.read_data()

        super().__init__(train_x=data, test=data)
    
    def read_data(self):
        image_dir = '/home/manogna/TTA/PromptAlign/data/domainnet'
        img_files = open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/painting_list.txt', 'r').readlines()
        items = []
        for line in img_files:
            path, label = line.split(' ')
            label = label.strip()
            impath = os.path.join(image_dir, path)
            item = Datum(impath=impath, label=int(label), classname=self._lab2cname[label])
            items.append(item)
        return items


@DATASET_REGISTRY.register()
class DomainNetSketch(DatasetBase):
    """DomainNet-Painting.

    This dataset is used for testing only.
    """

    dataset_dir = "domainnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "sketch")

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames = ImageNet.read_classnames(text_file)
        label2classnames = json.load(
            open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/label2class.json'))
        # self.classnames = list(label2classnames.values())
        self._lab2cname = label2classnames
        self._classnames = list(label2classnames.values())

        data = self.read_data()

        super().__init__(train_x=data, test=data)
    
    def read_data(self):
        image_dir = '/home/manogna/TTA/PromptAlign/data/domainnet'
        img_files = open('/home/manogna/TTA/PromptAlign/data/domainnet/domainnet126_lists/sketch_list.txt', 'r').readlines()
        items = []
        for line in img_files:
            path, label = line.split(' ')
            label = label.strip()
            impath = os.path.join(image_dir, path)
            item = Datum(impath=impath, label=int(label), classname=self._lab2cname[label])
            items.append(item)
        return items

