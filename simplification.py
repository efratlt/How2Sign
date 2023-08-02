import pandas as pd
import os
import torch
from signjoey.dataset import convert_data_to_dict
from tqdm import tqdm

#x = torch.load("/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_test_comb_simplify.pt", map_location='cpu')

#for f in x:
 #   assert f["text"] != ""

path = "/Users/eluzzon/efrat_private/how2sign_sentences"
sim_path = "/Users/eluzzon/efrat_private/how2sign_sentences/simplification_sentences"
phases = ["train", "val", "test"]

#x = pd.read_csv(path + f"/cvpr23.fairseq.i3d.train.how2sign.tsv", sep='\t')
#Y = pd.read_csv("train_sim.csv")
#y = 1
def remove_quoats(sentence):
    x = sentence
    sentence = sentence.replace("\"", '')
    """
    quoats = ["\"", "\'"]
    if len(sentence) <= 2:
        return sentence
    if sentence[0] in quoats and sentence[-1] in quoats:
        sentence = sentence[1:-1]
    """
    return sentence

def replace_text_with_simplification_sentences():

    for p in tqdm(phases):
        data = pd.read_csv(path + f"/cvpr23.fairseq.i3d.{p}.how2sign.tsv", sep='\t')
        sim_sentences = pd.read_csv(sim_path+f"/{p}_simplify.txt", sep="/t/n", header=None )
        data["simplify"] = sim_sentences[0]
        data["simplify"] = data["simplify"].apply(remove_quoats)
        filename = f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_{p}.pt"
        loaded_data = torch.load(filename, map_location='cpu')
        for feature in tqdm(loaded_data):
            simplify_text = data[data["id"] == feature["name"]]["simplify"].values[0]
            feature["text"] = simplify_text
        torch.save(loaded_data, f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_{p}_simplify.pt",
                                    _use_new_zipfile_serialization=False)



for p in phases:
    data = pd.read_csv(path + f"/cvpr23.fairseq.i3d.{p}.how2sign.tsv", sep='\t')
    #sim_sentences = pd.read_csv(sim_path+f"/train_simplify.txt", sep="/t/n", header=None )
    sim_sentences = pd.read_csv(f"{p}_sim.csv")
    data = data.sort_values(by=["translation"])
    sim_sentences = sim_sentences.sort_values(by=["original"])
    sim_sentences["original"] = sim_sentences["original"].apply(remove_quoats)
    data["translation"] = data["translation"].apply(remove_quoats)

    original_sentences = list(sim_sentences["original"].values)
    simplify_sentences = list(sim_sentences["sim"].values)
    data["simplify"] = ["" for i in range(len(data))]

    for i, row in data.iterrows():

        if row["translation"] in original_sentences:
            index_df = original_sentences.index(row["translation"])
            sim_sentence = simplify_sentences[index_df]
            data.at[i, "simplify"] = sim_sentence
            #row["simplify"] = sim_sentence
    data["simplify"] = data["simplify"].apply(remove_quoats)
    data[["id", "translation", "simplify"]].to_csv(f"{p}_tranlation_simplify.csv")
    """
    filename = f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_{p}.pt"
    #loaded_data = torch.load(filename, map_location='cpu')
    sim_dict = torch.load(filename, map_location='cpu')
    final_output = []
    for feature in tqdm(sim_dict):
        simplify_text = data[data["id"] == feature["name"]]["simplify"].values[0]
        if simplify_text != "":
            feature["text"] = simplify_text
            feature["name"] = "sim_" + feature["name"]
            final_output.append(feature)
    #loaded_data.update(sim_dict)
    print(p, len(final_output) / len(sim_dict))
    torch.save(final_output, f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_{p}_comb_simplify.pt",
                                _use_new_zipfile_serialization=False)
    """

