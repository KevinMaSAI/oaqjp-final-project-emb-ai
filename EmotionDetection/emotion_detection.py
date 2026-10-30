import requests


def emotion_detector(text_to_analyse):
    """Analyze the emotions expressed in the supplied text."""
    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyse}}

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    emotion_scores = response.json()["emotionPredictions"][0]["emotion"]
    result = {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
    }
    result["dominant_emotion"] = max(result, key=result.get)
    return result
