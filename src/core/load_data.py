import io
import zipfile

import pandas as pd
import requests

zip_file_url = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/educationandchildcare/datasets/2011censusteachingfile/current/rft-teaching-file.zip"

def load_data(zip_file_url):
    """Load the zip data from the url and unpack them"""
    
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("../data/raw")
load_data(zip_file_url)

def initalize_result_df():
    """Initiliaze the dataframe which will save the result"""
    results = pd.DataFrame(
        columns=[
            "index",
            "Features",
            "Number_features",
            "Model_typ",
            "Accuracy_train",
            "Accuracy_val",
            "Model",
            "search_space",
            "search_res",
            "resampler"
        ]
    )
    results.to_csv("../data/processed/Comparison_models.csv", index=False)


initalize_result_df()
