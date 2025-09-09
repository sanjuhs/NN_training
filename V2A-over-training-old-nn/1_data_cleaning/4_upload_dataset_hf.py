# now i need to use hugging face token to upload the training dataset to the hub

# first switch to using ipv4 only
# put this at the very top of your script/notebook
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only

# now the conection shoudl work well 
import os
from pathlib import Path
from tqdm import tqdm

root_path = Path.cwd()  # use current working directory in Colab
print(f"Root path: {root_path}")

print("Now uploading the training dataset to the hub")
print("The training Dataset is" , f"{root_path}/data/training_dataset")

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()


# from dotenv import load_dotenv
print("imported huggingface_hub")
# load the dotenv file

HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN" , HF_TOKEN)

api = HfApi(token=HF_TOKEN)
print("made the api object" , api)

print("starting upload ...")
api.upload_folder(
    folder_path=f"{root_path}/data/training_dataset",
    repo_id="sanjuhs/audio_to_blendshapes_test",
    repo_type="dataset",
    # multi_commits=True,
    # multi_commits_verbose=True,
)


print("uploaded the dataset to the hub!")