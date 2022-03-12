import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_msp_score(logits):
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores

def get_energy_score(logits):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores

def get_odin_score(inputs, forward_func, **kwargs):
    temper = 1000.0
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    outputs = forward_func(inputs, **kwargs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    with torch.no_grad():
        outputs = forward_func(tempInputs, **kwargs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_score(method, **kwargs):
    if method == "msp":
        scores = get_msp_score
    elif method == "odin":
        scores = get_odin_score
    elif method == "energy":
        scores = get_energy_score
    else:
        raise Exception(f"Unknown score method type: {method}")
    return scores