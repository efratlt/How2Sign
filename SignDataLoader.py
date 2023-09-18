import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from functools import partial

class SignDataObject:
    def __init__(self, gls, seq, sgn, signer, txt):
        self.gls = gls
        self.sequence =seq
        self.sgn = sgn
        self.signer = signer
        self.txt = txt

class DynamicSignDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_files_paths = dir_path
        self.files_names = sorted(os.listdir(self.dir_files_paths))

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        # Load data from the file at the specified index
        file_name = self.files_names[idx]
        data = torch.load(f"{self.dir_files_paths}/{file_name}")
        # Return data as a dictionary
        return SignDataObject(gls=data['gls'], seq=data['sequence'], sgn=data["sgn"], signer=data['signer'], txt=data['txt'])
        """
        return {
            'gls': data['gls'],
            'sequence': data['sequence'],
            'sgn': data["sgn"],
            'signer': data['signer'],
            'txt': data['txt']
        }
        """


def wrapper_collate_with_dynamic_padding(txt_padding, batch):
    return custom_collate_fn(batch, txt_padding)


def custom_collate_fn(batch_data, padding_index):
    n_batch = len(batch_data)
    # Pad the 'sgn' field in each batch
    sgn_batch = [example.sgn for example in batch_data]
    sgn_length = torch.tensor([sgn.shape[0] for sgn in sgn_batch])
    sgn_padded = pad_sequence(sgn_batch, batch_first=True, padding_value=0)  # Assuming 0 is the padding value
    sgn_batch_tuple = (sgn_padded, sgn_length)

    # Pad the 'txt' field in each batch
    txt_batch = [torch.tensor(example.txt) for example in batch_data]
    txt_length = torch.tensor([txt.shape[0] for txt in txt_batch])
    txt_padded = pad_sequence(txt_batch, batch_first=True, padding_value=padding_index)  # Assuming 0 is the padding value
    txt_batch_tuple = (txt_padded, txt_length)

    # Create batched dictionaries
    gls = tuple([torch.empty(n_batch, 0), torch.zeros(n_batch)])
    sequence = [example.sequence for example in batch_data]
    signer = [example.signer for example in batch_data]
    return SignDataObject(gls=gls, seq=sequence, sgn=sgn_batch_tuple, signer=signer, txt=txt_batch_tuple)

    """
    batched_data = {
        'gls': tuple([torch.empty(n_batch, 0), torch.zeros(n_batch)]) ,
        'sequence': [example['sequence'] for example in batch_data],
        'sgn': sgn_batch_tuple,
        'signer': [example['signer'] for example in batch_data],
        'txt': txt_batch_tuple
    }
    """
    #return batched_data


def get_txt_from_batch(batch_data):
    return {"txt": [example.txt for example in batch_data],
            "sequence": [example.sequence for example in batch_data]
            }

dir_path = "/Users/eluzzon/efrat_private/How2Sign/data/train_pt_files"
# Create an instance of the CustomDataset
dataset = DynamicSignDataset(dir_path)

custom_collate_fn_with_dynamic_padding = partial(wrapper_collate_with_dynamic_padding, 1)

dataset_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn_with_dynamic_padding)

# Access an example from the dataset
#example = dataset[0]
#print(example)
"""
for batch in dataset_loader:
    x = 1
    print(batch)
"""