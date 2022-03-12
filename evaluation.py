import torch
import numpy as np
import time

from utils.data_loader import load_id_data, load_ood_data
from utils.model_loader import load_model

class Evaluation:
    def __init__(self, args) -> None:
        self.kwargs = {'n': args.butterworth, 'threshold': args.threshold}
        self.id = args.id
        self.base_dir = args.base_dir
        self.id_data = load_id_data(args.id, args.batch_size)
        self.ood_data = {}
        for dataset in args.ood:
            self.ood_data[dataset] = load_ood_data(dataset, args.batch_size)
        self.model = load_model(args.model, args.id, args.filter,
                len(self.id_data.dataset.classes))

    def run(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print("Processing in-distribution images")
        img_total = len(self.id_data.dataset)
        img_count = 0
        timer = time.perf_counter()
        with open(self.base_dir / f'{self.id}-score.txt', 'w') as f_score, open(self.base_dir / f'{self.id}-label.txt', 'w') as f_label:
            for images, labels in self.id_data:
                img_count += images.shape[0]
                inputs = images.float()
                with torch.no_grad():
                    logits = self.model.forward(inputs, **self.kwargs)
                    outputs = torch.nn.functional.softmax(logits, dim=1)
                    outputs = outputs.detach().cpu().numpy()
                    preds = np.argmax(outputs, axis=1)
                    confs = np.max(outputs, axis=1)
                    for k in range(preds.shape[0]):
                        f_label.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

                print(f"{img_count:4}/{img_total:4} images processed, {time.perf_counter()-timer:.1f} seconds used.")
                timer = time.perf_counter()