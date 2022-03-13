import torch
import numpy as np
import time

from utils.data_loader import load_id_data, load_ood_data
from utils.model_loader import load_model
from utils.score import get_score

class Evaluation:
    def __init__(self, args) -> None:
        # arguments
        self.kwargs = {'n': args.butterworth, 'threshold': args.threshold}
        self.base_dir = args.base_dir

        # datasets
        self.id_name = args.id
        self.id_data = load_id_data(args.id, args.batch_size)
        self.ood_datas = {}
        for dataset in args.ood:
            self.ood_datas[dataset] = load_ood_data(dataset, args.batch_size)

        # models
        self.model = load_model(args.model, args.id, args.filter, len(self.id_data.dataset.classes))
        self.score = get_score(args.method)

    def run(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

        print("Processing in-distribution images")
        img_total = len(self.id_data.dataset)
        img_count = 0
        timer = time.perf_counter()
        print(f"Dataset: {self.id_name}")
        with open(self.base_dir / f'{self.id_name}-scores.txt', 'w') as f_score, open(self.base_dir / f'{self.id_name}-labels.txt', 'w') as f_label:
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

                scores = self.score(logits)
                for score in scores:
                    f_score.write("{}\n".format(score))

                print(f"{img_count:4}/{img_total:4} images processed, {time.perf_counter()-timer:.1f} seconds used.")
                timer = time.perf_counter()
        
        print("Processing out-of-distribution images")
        for ood_name, ood_data in self.ood_datas.items():
            img_total = len(ood_data.dataset)
            img_count = 0
            timer = time.perf_counter()
            print(f"Dataset: {ood_name}")
            with open(self.base_dir / f'{ood_name}-scores.txt', 'w') as f_score:
                for images, labels in ood_data:
                    img_count += images.shape[0]
                    inputs = images.float()
                    with torch.no_grad():
                        logits = self.model.forward(inputs, **self.kwargs)

                    scores = self.score(logits)
                    for score in scores:
                        f_score.write("{}\n".format(score))

                    print(f"{img_count:4}/{img_total:4} images processed, {time.perf_counter()-timer:.1f} seconds used.")
                    timer = time.perf_counter()