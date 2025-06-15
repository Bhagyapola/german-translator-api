import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import pipeline
import random

# Translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

app = FastAPI()

# ---------- Request Model ----------
class SentenceInput(BaseModel):
    sentence: str

# ---------- Response Model ----------
class LearningResponse(BaseModel):
    german_translation: str
    vocabulary: List[Dict[str, str]]
    grammar_tips: str
    example_sentences: str

# ---------- Dummy NLP logic for learning (to simulate dynamic behavior) ----------
def generate_learning_insights(sentence: str, translation: str):
    # Naive vocab pairing (for demo, realistic logic would use NLP)
    eng_words = sentence.split()
    ger_words = translation.replace(".", "").split()

    vocab = []
    for i in range(min(len(eng_words), len(ger_words))):
        vocab.append({"english": eng_words[i], "german": ger_words[i]})

    # Static + pattern-based grammar notes (can use a GPT model later)
    grammar_tips = [
        "In German, the verb usually comes in the second position.",
        "Nouns are capitalized in German.",
        "German word order may differ significantly from English."
    ]

    example_sentences = [
        f"EN: {sentence}",
        f"DE: {translation}",
        "EN: I speak German well. / DE: Ich spreche gut Deutsch.",
        "EN: She is learning German. / DE: Sie lernt Deutsch."
    ]

    return vocab, random.choice(grammar_tips), random.choice(example_sentences)

# ---------- Main Endpoint ----------
@app.post("/learn-german", response_model=LearningResponse)
def learn_german(data: SentenceInput):
    translation = translator(data.sentence)[0]['translation_text']

    vocab, tips, examples = generate_learning_insights(data.sentence, translation)

    return LearningResponse(
        german_translation=translation,
        vocabulary=vocab,
        grammar_tips=tips,
        example_sentences=examples
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
