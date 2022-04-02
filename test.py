import time
import copy
import torch
from tqdm import tqdm
import csv

csv_out = [('name', 'predicted')]


def test_model(model, dataloader, device):
    model.eval()

    # Iterate over data.
    with torch.no_grad(), tqdm(enumerate(dataloader), unit="batch", total=len(dataloader)) as tepoch:
        for i, (inputs, labels) in tepoch:
            inputs = inputs.float()
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            csv_out.append((labels[0], preds[0].item()))

    print(csv_out)

    with open('./output.csv', 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        for item in csv_out:
            writer.writerow(item)
