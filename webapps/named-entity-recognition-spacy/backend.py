import json

from flask import request
from spacy import displacy

from ner_utils_spacy import get_spacy_model


@app.route("/run_NER")  # noqa
def run_NER():
    text = request.args.get("input", "")
    language = request.args.get("language", "en")
    print("Processing text '{}' in language '{}'...".format(text, language))
    nlp = get_spacy_model(language=language)
    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=False)
    return json.dumps(html)
