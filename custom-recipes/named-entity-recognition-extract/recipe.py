# -*- coding: utf-8 -*-
import logging

from tqdm import tqdm

import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config

#############################
# Input & Output datasets
#############################

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

input_df = input_dataset.get_dataframe()

#############################
# Recipe Parameters
#############################

recipe_config = get_recipe_config()

text_column_name = recipe_config.get("text_column_name", None)
if not text_column_name:
    raise ValueError("You did not choose a text column.")

advanced_settings = recipe_config.get("advanced_settings", False)
if advanced_settings:
    output_single_json = recipe_config.get("output_single_json", False)
    ner_model = recipe_config.get("ner_model", "spacy")
else:
    output_single_json = False
    ner_model = "spacy"

if ner_model == "spacy":
    from ner_utils_spacy import extract_entities
else:
    from ner_utils_flair import extract_entities

    try:
        model_folder = get_input_names_for_role("model_folder")[0]
    except IndexError:
        raise Exception(
            "To use Flair, download the model using the macro and add the resulting folder as input to the recipe."
        )
    folder_path = dataiku.Folder(model_folder).get_path()

#############################
# Main Loop
#############################

CHUNK_SIZE = 100
n_lines = 0
logging.info("Started chunk-processing of input Dataset.")
for chunk_idx, df in enumerate(tqdm(input_dataset.iter_dataframes(chunksize=CHUNK_SIZE))):
    # Process chunk
    out_df = extract_entities(df[text_column_name].fillna(" "), format=output_single_json)
    df = df.reset_index(drop=True)
    out_df = out_df.reset_index(drop=True)
    out_df = df.merge(out_df, left_index=True, right_index=True)

    # Append dataframe to output Dataset
    if chunk_idx == 0:
        output_dataset.write_schema_from_dataframe(out_df)
        writer = output_dataset.get_writer()
        writer.write_dataframe(out_df)
    else:
        writer.write_dataframe(out_df)
    n_lines += len(df)
    logging.info("Finished processing {} lines".format(n_lines))
writer.close()
