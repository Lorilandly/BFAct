import argparse
from pathlib import Path

from evaluation import Evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD detection model')
    
    # model related
    parser.add_argument("--id", default="imagenet", type=str, help="['CIFAR100', 'imagenet']")
    parser.add_argument("--ood", default=['inat', 'sun50', 'places50', 'dtd'], type=list, help="['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument("--model", default="resnet50", type=str, help="['resnet50', 'resnet18', 'mobilenet']")
    parser.add_argument("--method", default="energy", type=str, help="odin mahalanobis CE_with_Logst")
    
    # filter related
    parser.add_argument('--filter', default='bt', type=str, help="['react', 'bt']")
    parser.add_argument('--threshold', default=1.0, type=float, help='sparsity level')
    parser.add_argument('--butterworth', default=2.0, type=float, help='butterworth n parameter')
    
    # runtime related
    parser.add_argument('-b', '--batch-size', default=25, type=int, help='mini-batch size')
    parser.add_argument('--base-dir', default='output/ood_scores', type=Path, help='result directory')
    parser.add_argument('--record_layer', default=None, type=Path, help='directory to save layer info')
    
    args = parser.parse_args()

    cls = Evaluation(args)
    cls.run()
    # compute_ood()
    # compute_in()
    print("Success!")