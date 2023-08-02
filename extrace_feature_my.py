import os
"""
from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
"""
import torch
import pandas as pd
from tqdm import tqdm

x = torch.load("/Users/eluzzon/Downloads/train_1.pt", map_location="cpu")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name(0))

"""
def get_feature_vectors(feature_vectors_dict:dict, paths:list):
  # Select the feature type
  videos_failure = []
  feature_type = 'i3d'

  # Load and patch the config
  args = OmegaConf.load(build_cfg_path(feature_type))
  #args.video_paths = ["/content/drive/MyDrive/V2C/videos_/dev/06October_2012_Saturday_tagesschau-8744.mp4"]
  args.video_paths = paths #['/content/drive/MyDrive/V2C/videos_/dev/01April_2010_Thursday_heute-6697.mp4', "/content/drive/MyDrive/V2C/videos_/dev/01April_2010_Thursday_heute-6698.mp4", "/content/drive/MyDrive/V2C/videos_/dev/07October_2010_Thursday_tagesschau-4127.mp4"]
  #args.show_pred = True
  args.stack_size = 10
  args.step_size = 1
  args.extraction_fps = 15
  args.keep_tmp_files = True
  args.flow_type = 'raft' # 'pwc' is not supported on Google Colab (cupy version mismatch)
  # args.streams = 'flow'

  # Load the model
  extractor = ExtractI3D(args)

  # Extract features
  for video_path in tqdm(args.video_paths):
    try:
  #    print(f'Extracting for {video_path}')
      feature_dict = extractor.extract(video_path)
      feature_vectors_dict[f"{video_path}$$rgb"] = torch.from_numpy(feature_dict["rgb"])
      feature_vectors_dict[f"{video_path}$$flow"] = torch.from_numpy(feature_dict["flow"])
      #[(print(k), print(v.shape), print(v)) for k, v in feature_dict.items()]

    except:
      print("failed")
      videos_failure.append(video_path)
  return videos_failure



videos_path = "/content/drive/MyDrive/V2C/videos_/"
feature_vectors_dict = {}
features = []
failures = {}
for dir_phase in os.listdir(videos_path):
    if dir_phase == "train":
      dir_phase_paths = os.path.join(videos_path, dir_phase)
      paths = [os.path.join(dir_phase_paths, video_path) for video_path in os.listdir(dir_phase_paths)][:10]
      paths_failures = get_feature_vectors(feature_vectors_dict, paths)
      torch.save(feature_vectors_dict, f"{dir_phase}_.pt")
      if len(paths_failures):
          failures[dir_phase] = paths_failures

if len(failures):
    failures_df = pd.DataFrame(failures).T.reset_index()
    failures_df.columns = ["phase", "path"]
    failures_df.to_csv(f"failed_videos_test.csv")


torch.load("/content/drive/MyDrive/V2C/video_features/train_.pt")

"""