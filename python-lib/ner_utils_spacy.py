# -*- coding: utf-8 -*-
from collections import defaultdict
import json

import pandas as pd
import spacy

SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "nb_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "nb": "nb_core_news_sm",
}

def get_spacy_model(language: str):
    language_model = SPACY_LANGUAGE_MODELS.get(language, None)
    if language_model is None:
        raise ValueError(f"The language {language} is not available. \
                        You can add the language & corresponding model name by editing the code.")
    try:
        nlp = spacy.load(language_model)
    except OSError:
        # Raising ValueError instead of OSError so it shows up at the top of the log
        raise ValueError(f"Could not find spaCy model for the language {language}. \
                        Maybe you need to edit the requirements.txt file to enable it.")
    return nlp

def extract_entities(text_column, format: bool, language: str):
    # Tag sentences
    nlp = get_spacy_model(language=language)
    docs = nlp.pipe(text_column.values)

    # Extract entities
    rows = []
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

        rows.append(df_row)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df
