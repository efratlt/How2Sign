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


def combine_original_sentence_with_simplification_sentence(phases):
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
def convert_string_tensor_to_float(str_tensor):
    return float(str_tensor.split("tensor([[")[1].split("]])")[0])

main_path = "/Users/eluzzon/efrat_private/How2Sign/simplification_sentence"
for p in phases:
    origina_simplify_features = []
    original_simplification_with_score = pd.read_csv(f"{main_path}/{p}_text_simplification_sematic_score.csv")
    original_simplification_with_score["score"] = original_simplification_with_score["score"].apply(convert_string_tensor_to_float)
    feature_data_path = f"/Users/eluzzon/efrat_private/fpt4slt/data/als/sub1_{p}.pt"
    loaded_data = torch.load(feature_data_path, map_location='cpu')
    i = 0
    for feature_data in tqdm(loaded_data):
        feature_id = feature_data["name"]
        row = original_simplification_with_score[original_simplification_with_score["id"] == feature_id]
        if not row.empty and row["score"].values[0] >= 0.95:
            i += 1
            copy_feature_data = feature_data.copy()
            copy_feature_data["name"] = "copy_" + copy_feature_data["name"]
            copy_feature_data["text"] = row["simplify"].values[0]
            origina_simplify_features.append(copy_feature_data)
        else:
            origina_simplify_features.append(feature_data)

    torch.save(origina_simplify_features, f"/Users/eluzzon/efrat_private/How2Sign/data/sub1_{p}_comb_simplify>0.8.pt",
        _use_new_zipfile_serialization=False)
    len_phase = len(loaded_data)
    print(f"{p}_time: {i}/{len_phase}")