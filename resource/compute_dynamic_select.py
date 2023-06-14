# display options depending on model provider availability
model_provider = None

try:
    from dataiku.core.model_provider import get_model_provider
    model_provider = get_model_provider()
except ImportError:
    pass

def do(payload, *args, **kwargs):
    choices = [
        {
            "value": "en",
            "label": "English - 4 categories"
        }
    ]
    if model_provider is not None:
        choices.append(
            {
                "value": "ner_english_ontonotes_fast",
                "label": "English - 18 categories"
            }
        )
    return {"choices": choices}
