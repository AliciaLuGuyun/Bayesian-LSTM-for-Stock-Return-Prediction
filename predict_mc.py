# predict_mc.py

import torch
import numpy as np

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def predict_with_uncertainty(model, inputs, device, mc_samples=100):
    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(mc_samples):
            preds = model(inputs.to(device)).squeeze(-1) 
            predictions.append(preds.cpu().numpy())

    predictions = np.stack(predictions)
    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)

    return mean_preds, std_preds

