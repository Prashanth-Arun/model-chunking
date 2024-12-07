from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="/home/dongfu/WorkSpace/model-chunking/saves/qwen2_chunking_mlp_freeze_uniform_with_shared_start_sft",
    repo_id="DongfuJiang/qwen2_chunking_mlp_freeze_uniform_with_shared_start_sft",
    repo_type="model",
)