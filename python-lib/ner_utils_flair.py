# -*- coding: utf-8 -*-
from collections import defaultdict
import json

from flair.data import Sentence
import pandas as pd


def extract_entities(text_column, format, tagger):
    # Create Sentences
    sentences = [Sentence(text, use_tokenizer=True) for text in text_column.values]

    # Tag Sentences
    tagger.predict(sentences)

    # Extract entities
    rows = []
    for sentence in sentences:
        df_row = defaultdict(list)
        for entity in sentence.get_spans('ner'):
            tag = entity.get_label("ner").value
            text = entity.text
            df_row[tag].append(text)
        if format:
            df_row = {"sentence": sentence.to_plain_string(), "entities": json.dumps(df_row)}
        else:
            for k, v in df_row.items():
                df_row[k] = json.dumps(v)
            df_row["sentence"] = sentence.to_plain_string()

        rows.append(df_row)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df
