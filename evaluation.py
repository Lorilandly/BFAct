import torch
import numpy as np

from utils.data_loader import load_id_data, load_ood_data

class Evaluation:
    def __init__(self, args) -> None:
        self.id_data = load_id_data(args.id, args.batch_size)
        self.ood_data = {}
        for dataset in args.ood:
            self.ood_data[dataset] = load_ood_data(dataset, args.batch_size)

    def run(self):
        print(self.id_data)
        print(self.ood_data)
