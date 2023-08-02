import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split

train_annot = pd.read_csv("/Users/eluzzon/Downloads/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv", delimiter="|")
dev_annot = pd.read_csv("/Users/eluzzon/Downloads/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv", delimiter="|")
test_annot = pd.read_csv("/Users/eluzzon/Downloads/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv", delimiter="|")


train_annot["phase"] = ["train" for i in range(len(train_annot))]
dev_annot["phase"] = ["dev" for i in range(len(dev_annot))]
test_annot["phase"] = ["test" for i in range(len(test_annot))]

for df in [train_annot, dev_annot, test_annot]:
  df["orth"] = ["" for i in range(len(df))]

combine_annot = pd.concat([train_annot, dev_annot, test_annot])


features = []
path_main_dir = "/Users/eluzzon/Downloads/pt_feature/"
for file_name in os.listdir(path_main_dir ):
  if not file_name.startswith("PHOENIX"):
    loaded = torch.load(os.path.join(path_main_dir, file_name))
    features.append(pd.DataFrame(loaded.items(), columns=["name", "feature"]))
features_vectors = pd.concat(features)


indexes = []
for i in range(len(features_vectors["feature"])):
  if len(features_vectors.iloc[i]["feature"]) == 0:
    indexes.append(features_vectors.iloc[i]["name"])
features_vectors = features_vectors[~features_vectors["name"].isin(indexes)]


features_vectors["type"] = [feature_type.split("$$")[1] for feature_type in features_vectors["name"]]
base_files_name = [name[-1].split(".mp4")[0] for name in features_vectors["name"].str.split("/")]
features_vectors.insert(0, "base_name", base_files_name, True)

feature_vectors_combine = pd.merge(features_vectors, combine_annot, left_on='base_name', right_on='name', how="left")
feature_to_train = feature_vectors_combine[["base_name", "speaker", "orth", "translation","feature", "type"]]
feature_to_train = feature_to_train.rename(columns={"base_name": "name", "speaker" : "signer", "orth": "gloss", "translation": "text", "feature" : "sign"})
rgb_features = feature_to_train[feature_to_train["type"] == "rgb"][["name", "signer", "gloss", "text", "sign"]]

train_rgb, test = train_test_split(rgb_features, test_size=0.3)
dev_rgb, test_rgb = train_test_split(test, test_size=0.7)
train_rgb["name"] = [f"train/{feature_name}" for feature_name in train_rgb["name"].values]
dev_rgb["name"] = [f"dev/{feature_name}" for feature_name in dev_rgb["name"].values]
test_rgb["name"] = [f"test/{feature_name}" for feature_name in test_rgb["name"].values]

torch.save(train_rgb.values, f"data/flow/rgb_feature_train.pt")
torch.save(dev_rgb.values, f"data/flow/rgb_feature_dev.pt")
torch.save(test_rgb.values, f"data/flow/rgb_feature_test.pt")

flow_features = feature_to_train[feature_to_train["type"] == "flow"][["name", "signer", "gloss", "text", "sign"]]
train_flow, test = train_test_split(flow_features, test_size=0.3)
dev_flow, test_flow = train_test_split(test, test_size=0.7)
""""
train_flow["name"] = [f"train/{feature_name}" for feature_name in train_flow["name"].values]
dev_flow["name"] = [f"dev/{feature_name}" for feature_name in dev_flow["name"].values]
test_flow["name"] = [f"test/{feature_name}" for feature_name in test_flow["name"].values]

torch.save(train_flow.values, f"/content/drive/MyDrive/V2C/fpt4slt/data/flow/flow_feature_train.pt")
torch.save(dev_flow.values, f"/content/drive/MyDrive/V2C/fpt4slt/data/flow/flow_feature_dev.pt")
torch.save(test_flow.values, f"/content/drive/MyDrive/V2C/fpt4slt/data/flow/flow_feature_test.pt")
#torch.save(flow_features.values, f"flow_feature.pt")
#feature_to_train[feature_to_train["type"] == "flow"]
"""