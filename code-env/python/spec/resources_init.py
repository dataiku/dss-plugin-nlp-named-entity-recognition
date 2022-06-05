######################## Base imports #################################
from dataiku.code_env_resources import clear_all_env_vars
from dataiku.code_env_resources import set_env_path

######################## Download FLAIR Models ###########################
# Clear all environment variables defined by a previously run script
clear_all_env_vars()

# Set Flair cache directory
set_env_path("FLAIR_CACHE_ROOT", "flair")

from flair.models import SequenceTagger

# Download pretrained model: automatically managed by Flair,
# does not download anything if model is already in FLAIR_CACHE_ROOT
SequenceTagger.load('ner')
# Add any other models you want to download, check https://huggingface.co/flair for examples
# E.g. SequenceTagger.load('flair/ner-french')
# Make sure to modify the model used in recipe.py if you want to use a different model
