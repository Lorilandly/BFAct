import torch
import numpy as np

from utils.data_loader import load_id_data, load_ood_data
from utils.model_loader import load_model

class Evaluation:
    def __init__(self, args) -> None:
        self.id_data = load_id_data(args.id, args.batch_size)
        self.ood_data = {}
        for dataset in args.ood:
            self.ood_data[dataset] = load_ood_data(dataset, args.batch_size)
        self.model = load_model(args.model, args.id,
                len(self.id_data.dataset.classes))

    def run(self):
        print("Processing in-distribution images")
        N = len(self.id_data.dataset)
        data_iter = iter(self.ood_data['dtd'])
        img, lab = data_iter.next()
        print(img, lab)