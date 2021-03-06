# -*- coding: utf-8 -*-
import os
import logging
import re
import json
import requests
from urllib.parse import urlparse
from collections import defaultdict

from flair.data import Sentence
from flair.models.sequence_tagger_model import SequenceTagger
import pandas as pd
from tqdm import tqdm

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


def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    os.makedirs(cache_dir, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        logging.info("File {} found in cache".format(filename))
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url)
    if response.status_code != 200:
        raise IOError("HEAD request failed for url {}".format(url))

    if not os.path.exists(cache_path):
        logging.info("File {} not found in cache, downloading from URL {}...".format(filename, url))
        req = requests.get(url, stream=True)
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)
        with open(cache_path, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
        progress.close()

    return cache_path


def cached_path(url_or_filename: str, cache_path: str, cache_dir: str) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    dataset_cache = os.path.join(cache_path, cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


class CustomSequenceTagger(SequenceTagger):
    @staticmethod
    def load(model: str, cache_path: str):
        model_file = None
        aws_resource_path = "https://nlp.informatik.hu-berlin.de/resources/models"

        if model.lower() == "ner":
            base_path = "/".join([aws_resource_path, "ner", "en-ner-conll03-v0.4.pt"])
            model_file = cached_path(base_path, cache_path, cache_dir="models")

        if model.lower() == "ner-ontonotes-fast":
            base_path = "/".join([aws_resource_path, "ner-ontonotes-fast", "en-ner-ontonotes-fast-v0.4.pt"])
            model_file = cached_path(base_path, cache_path, cache_dir="models")

        if model_file is not None:
            tagger = SequenceTagger.load(model_file)
            return tagger


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
