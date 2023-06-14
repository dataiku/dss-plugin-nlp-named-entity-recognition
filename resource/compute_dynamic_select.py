# display options depending on model provider availability
model_provider = None

try:
    from dataiku.core.model_provider import get_model_provider
    model_provider = get_model_provider()
except ImportError:
    pass

def do(*args, **kwargs):
    choices = [
        {
            "value": "zh",
            "label": "Chinese"
        },
        {
            "value": "en",
            "label": "English"
        },
        {
            "value": "fr",
            "label": "French"
        },
        {
            "value": "de",
            "label": "German"
        },
        {
            "value": "ja",
            "label": "Japanese"
        },
        {
            "value": "nb",
            "label": "Norwegian Bokmål"
        },
        {
            "value": "pl",
            "label": "Polish"
        },
        {
            "value": "es",
            "label": "Spanish"
        }
    ]
    if model_provider is not None:
        choices.append(
            {
                "value": "en_core_web_trf",
                "label": "English - Transformer"
            }
        )
    return {"choices": choices}
