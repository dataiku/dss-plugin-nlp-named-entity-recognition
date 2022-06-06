# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config

from dku_io_utils import process_dataset_chunks

#############################
# Input & Output datasets
#############################

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

#############################
# Recipe Parameters
#############################

recipe_config = get_recipe_config()

text_column_name = recipe_config.get("text_column_name", None)
if not text_column_name:
    raise ValueError("Please choose a text column")

advanced_settings = recipe_config.get("advanced_settings", False)
if advanced_settings:
    output_single_json = recipe_config.get("output_single_json", False)
    ner_model = recipe_config.get("ner_model", "spacy")
else:
    output_single_json = False
    ner_model = "spacy"

if ner_model == "spacy":
    from ner_utils_spacy import extract_entities

    language = recipe_config.get("text_language_spacy", "en")
else:
    from flair.models import SequenceTagger
    from ner_utils_flair import extract_entities
    
    tagger = SequenceTagger.load("flair/ner-english-fast")

#############################
# Main Loop
#############################


def compute_entities_df(df):
    if ner_model == "spacy":
        out_df = extract_entities(df[text_column_name].fillna(" "), format=output_single_json, language=language)
    else:
        out_df = extract_entities(df[text_column_name].fillna(" "), format=output_single_json, tagger=tagger)
    df = df.reset_index(drop=True)
    out_df = out_df.reset_index(drop=True)
    out_df = df.merge(out_df, left_index=True, right_index=True)
    return out_df


process_dataset_chunks(
    input_dataset=input_dataset, output_dataset=output_dataset, func=compute_entities_df, chunksize=100
)
