import pandas as pd
import torch
from signjoey.dataset import convert_data_to_dict
#phases = {"train": 7100, "val" : 520, "test": 650}
phases = {"test": 650}

#x = torch.load("/Users/eluzzon/efrat_private/fpt4slt/data/als/sub_train.pt", map_location='cpu')


for phase, n_features in phases.items():
    save_features = []
    filename = f"/Users/eluzzon/efrat_private/fpt4slt/data/als/pre_{phase}.pt"
    loaded_data = torch.load(filename, map_location='cpu')
    loaded_object = convert_data_to_dict(loaded_data)
    for featrue in loaded_object:
        if len(save_features) == 5:
            break
        if featrue["sign"].shape[0] <= 10:
            save_features.append(featrue)
    torch.save(save_features, f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub3_{phase}.pt", _use_new_zipfile_serialization=False)