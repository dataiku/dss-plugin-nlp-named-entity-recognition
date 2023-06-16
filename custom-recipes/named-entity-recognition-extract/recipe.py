# -*- coding: utf-8 -*-
import multiprocessing

import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config

from dku_io_utils import process_dataset_chunks

from ner.constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)

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
output_format = COLUMN_PER_ENTITY_FORMAT
output_single_json = False
ner_model = "spacy"
if advanced_settings:
    output_single_json = recipe_config.get("output_single_json", False)
    ner_model = recipe_config.get("ner_model", "spacy")
    if output_single_json:
        output_format = (
            JSON_KEY_PER_ENTITY_FORMAT
            if recipe_config.get("output_json_format", "standard") == "standard"
            else JSON_LABELING_FORMAT
        )

if ner_model == "spacy":
    from ner.spacy import extract_entities

    language = recipe_config.get("text_language_spacy", "en")
else:
    import flair
    from flair.models import SequenceTagger
    from ner.flair import extract_entities    
    flair.device = recipe_config.get("flair_device", "cpu")
    tagger = SequenceTagger.load("flair/ner-english-fast@3d3d35790f78a00ef319939b9004209d1d05f788")

#############################
# Main Loop
#############################


def compute_entities_df(df):
    if ner_model == "spacy":
        out_df = extract_entities(df[text_column_name].fillna(" "), format=output_format, language=language)
    else:
        out_df = extract_entities(df[text_column_name].fillna(" "), format=output_format, tagger=tagger)
    df = df.reset_index(drop=True)
    out_df = out_df.reset_index(drop=True)
    out_df = df.merge(out_df, left_index=True, right_index=True)
    return out_df

if __name__ == '__main__':
    if ner_model == "spacy":
        chunksize = 200 * multiprocessing.cpu_count()
    else:
        chunksize = 100

    process_dataset_chunks(
        input_dataset=input_dataset, output_dataset=output_dataset, func=compute_entities_df, chunksize=chunksize
    )
