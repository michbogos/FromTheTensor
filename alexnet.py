from datasets import load_dataset
from huggingface_hub import list_datasets
ds = load_dataset(name="imagenet-1k", path="./datasets", split="train", use_auth_token=True)