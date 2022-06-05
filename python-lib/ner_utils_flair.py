# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import re

from flair.data import Sentence
import pandas as pd

FLAIR_ENTITIES = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]

#############################
# NER function
#############################

# Regex for matching either
PATTERN = r"({}|{})".format(
    # Single-word entities
    r"(?:\s*\S+ <S-[A-Z_]*>)",  # (<S-TAG> format)
    # Match multi-word entities
    r"{}{}{}".format(
        r"(?:\s*\S+ <B-[A-Z_]*>)",  # A first tag in <B-TAG> format
        r"(?:\s*\S+ <I-[A-Z_]*>)*",  # Zero or more tags in <I-TAG> format
        r"(?:\s*\S+ <E-[A-Z_]*>)",  # A final tag in <E-TAG> format
    ),
)
matcher = re.compile(PATTERN)


def extract_entities(text_column, format, tagger):
    # Create Sentences
    sentences = [Sentence(text, use_tokenizer=True) for text in text_column.values]

    # Tag Sentences
    tagger.predict(sentences)

    # Retrieve entities
    if format:
        entity_df = pd.DataFrame()
    else:
        entity_df = pd.DataFrame(columns=FLAIR_ENTITIES)

    for sentence in sentences:
        df_row = defaultdict(list)
        entities = matcher.findall(sentence.to_tagged_string())
        # Entities are in the following format: word1 <X-TAG> word2 <X-TAG> ...
        for entity in entities:
            # Extract entity text (word1, word2, ...)
            text = " ".join(entity.split()[::2])
            # Extract entity type (TAG)
            tag = re.search(r"<.-(.+?)>", entity).group(1)
            df_row[tag].append(text)

        if format:
            df_row = {"sentence": sentence.to_plain_string(), "entities": json.dumps(df_row)}
        else:
            for k, v in df_row.items():
                df_row[k] = json.dumps(v)
            df_row["sentence"] = sentence.to_plain_string()

        entity_df = entity_df.append(df_row, ignore_index=True)

    cols = [col for col in entity_df.columns.tolist() if col != "sentence"]
    entity_df = entity_df[cols]

    return entity_df
