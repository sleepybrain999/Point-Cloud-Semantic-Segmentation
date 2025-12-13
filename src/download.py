import os

def data_download(kaggle_username,kaggle_key , dest_path="./data/pandaset",download=True):
    """
    Download PandaSet dataset from Kaggle

    Args:
        kaggle_username (str): Your Kaggle username.
        kaggle_key (str): Your Kaggle API key.
        dest_path (str): Destination path for dataset.
    """
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    if download == True:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Set up Kaggle API
        os.makedirs(dest_path, exist_ok=True)
        api = KaggleApi()
        api.authenticate()

        # Download dataset to dest_path
        dataset_slug = "usharengaraju/pandaset-dataset"
        print("Downloading PandaSet...")
        api.dataset_download_files(
            dataset_slug,
            path=dest_path,
            unzip=True,
            quiet=False
        )
        print("Download complete — check:", dest_path)

    else:
        print("Already downloaded, no need to download again")

