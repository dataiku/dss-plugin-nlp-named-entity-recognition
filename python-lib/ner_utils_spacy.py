# -*- coding: utf-8 -*-
import json

import pandas as pd
import spacy

# backward compatibility
model_provider = None

try:
    from dataiku.core.model_provider import get_model_provider
    model_provider = get_model_provider()
except ImportError:
    pass


SPACY_LANGUAGE_MODELS_LEGACY_MAPPING = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "nb_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "nb": "nb_core_news_sm",
}

SPACY_MODEL_PROVIDER_MAPPING = {
    "en_core_web_trf": "spacy/en_core_web_trf@de00f6d68ceec2864448ffa2e00bda7f05605d2e",
}

def get_model(model_id: str):
    language_model = SPACY_LANGUAGE_MODELS_LEGACY_MAPPING.get(model_id, None)
    if language_model is not None:
        # those models are downloaded on resources init
        model_path = language_model
    elif model_provider is not None:
        # dl model with provider at hugging face proper location, return pytorch bin path
        model_path = model_provider.get_or_download_model(SPACY_MODEL_PROVIDER_MAPPING[model_id])
    return spacy.load(model_path, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
def extract_entities(text_column, format, model_id, n_process=-1):
    # Tag sentences
    nlp = get_model(model_id=model_id)
    docs = nlp.pipe(text_column.values, n_process=n_process, batch_size=100)
    # Extract entities
    rows = {
        "standard": get_standard_rows,
        "labeling": get_labeling_rows
    }.get(format, get_default_rows)(docs)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df

def get_standard_rows(docs):
    rows = []
    for doc in docs:
        entities = {}
        for span in doc.ents:
            if span.label_ not in entities:
                entities[span.label_] = []
            entities[span.label_].append(span.text)
        rows.append({"sentence": doc.text, "entities": json.dumps(entities)})
    return rows

def get_default_rows(docs):
    rows = []
    for doc in docs:
        labels = {}
        for span in doc.ents:
            if span.label_ not in labels:
                labels[span.label_] = []
            labels[span.label_].append(span.text)
        row = {"sentence": doc.text}
        for k, v in labels.items():
            row[k] = json.dumps(v)
        rows.append(row)
    return rows

def get_labeling_rows(docs):
    rows = []
    for doc in docs:
        entities = []
        for span in doc.ents:
            entities.append({
                "text": span.text,
                "beginningIndex": span.start_char,
                "endIndex": span.end_char,
                "category": span.label_
            })
        rows.append({"sentence": doc.text, "entities": json.dumps(entities)})
    return rows
