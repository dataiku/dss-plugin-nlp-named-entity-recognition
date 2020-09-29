# -*- coding: utf-8 -*-
import json
from collections import defaultdict
import pandas as pd
import spacy

SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "nb_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "nb": "nb_core_news_sm",
}


def extract_entities(text_column, format: bool, language: str):
    # Tag sentences
    nlp = spacy.load(SPACY_LANGUAGE_MODELS[language])
    docs = nlp.pipe(text_column.values)

    # Extract entities
    entity_df = pd.DataFrame()
    for doc in docs:
        df_row = defaultdict(list)
        for entity in doc.ents:
            df_row[entity.label_].append(entity.text)

        if format:
            df_row = {"sentence": doc.text, "entities": json.dumps(df_row)}
        else:
            for k, v in df_row.items():
                df_row[k] = json.dumps(v)
            df_row["sentence"] = doc.text

        entity_df = entity_df.append(df_row, ignore_index=True)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]

    return entity_df
