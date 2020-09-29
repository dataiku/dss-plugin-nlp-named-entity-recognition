import json
from flask import request
import spacy
from spacy import displacy

from ner_utils_spacy import SPACY_LANGUAGE_MODELS


@app.route("/run_NER")  # noqa
def run_NER():
    text = request.args.get("input", "")
    language = request.args.get("language", "en")
    print("Processing text '{}' in language '{}'...".format(text, language))
    nlp = spacy.load(SPACY_LANGUAGE_MODELS[language])
    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=False)
    return json.dumps(html)
