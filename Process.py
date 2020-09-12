import Utils

import torch
import os
import json
import numpy as np

import argparse




ap = argparse.ArgumentParser(description='Predict.py')

ap.add_argument('--input', default='C:/Users/olyas/Desktop/Man_Woman', nargs='?', action="store", type = str)
ap.add_argument('--checkpoint', default='C:/Users/olyas/Desktop/Best_mobnet111.pth', nargs='?', action="store", type = str)


pa = ap.parse_args()
path_image = pa.input
path_model = pa.checkpoint


def main():
    
    model = Utils.CNNClassifier()
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()

    transform = Utils.transforms_()
    my_dataset = Utils.CustomDataSet(path_image,transform)
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=0)
    device = 'cpu'



    with torch.no_grad():
        y_pred = []
        for b in loader:
            b = b.to(device)
            y_pred.extend(torch.argmax(model(b),dim=1).cpu().numpy())
            
    y_ = ['female' if x==0 else 'male' for x in y_pred]
    gloss = dict(zip(os.listdir(path_image), list(y_)))


    npencoder = Utils.NpEncoder

    filepath = 'Process_images'
    json_txt = json.dumps(gloss, cls = npencoder)
    with open(filepath, 'w') as file:
        file.write(json_txt)

    print(gloss)    
    print('Done!')    
    
    
if __name__== "__main__":
    main()    







