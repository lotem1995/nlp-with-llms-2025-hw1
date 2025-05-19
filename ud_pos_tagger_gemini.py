# --- Imports ---
import os
from google import genai
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Optional
from enum import Enum
from tqdm import tqdm
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"      # Adjective
    ADP = "ADP"      # Adposition
    ADV = "ADV"      # Adverb
    AUX = "AUX"      # Auxiliary
    CCONJ = "CCONJ"  # Coordinating conjunction
    DET = "DET"      # Determiner
    INTJ = "INTJ"    # Interjection
    NOUN = "NOUN"    # Noun
    NUM = "NUM"      # Numeral
    PART = "PART"    # Particle
    PRON = "PRON"    # Pronoun
    PROPN = "PROPN"  # Proper noun
    PUNCT = "PUNCT"  # Punctuation
    SCONJ = "SCONJ"  # Subordinating conjunction
    SYM = "SYM"      # Symbol
    VERB = "VERB"    # Verb
    X = "X"          # Other
class SentencePOS(BaseModel):
    sentence_words:List[str]
    sentence_tags:List[UDPosTag]

# TODO Define more Pydantic models for structured output
class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

class TagExplanation(BaseModel):
    word: str
    correct_tag: UDPosTag
    predicted_tag: UDPosTag
    explanation: str
    category: str

class TagExplanations(BaseModel):
    explanations: List[TagExplanation]

class SentenceSegmentation(BaseModel):
    sentence: str
    tokens: List[str]
class SentenceSegmentations(BaseModel):
    segmentations: List[SentenceSegmentation] = Field(description="A list of segmented sentences.")
# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    # genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""
You are an expert in Universal Dependencies (UD) Part-of-Speech tagging. 
Your task is to provide accurate UD POS tags for each word in the list of sentences below.

Instructions:
1. Each sentence is on a separate line.
2. Tag each word in the sentence with its corresponding UD POS tag.
3. Ensure the order of the words is preserved.
4. Follow the standard UD tagset conventions.


Now tag the following sentences:
{text_to_tag}
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    if response and response.parsed:
        return response.parsed
    else:
        print("❌ Parsing failed. Check the raw response below:")
        print(response.text if response else "No response received.")
        return None

def explain_tagging_error(sentence: str, failed_words: list[str], correct_tags:list[str],predicted_tags:list[str])->Optional[TagExplanations]:
    error_descriptions = "\n".join([
        f"- Word: '{word}', Expected: '{correct_tag}', Predicted: '{predicted_tag}'"
        for word, correct_tag, predicted_tag in zip(failed_words, correct_tags, predicted_tags)
    ])
    # Construct the prompt
    prompt = f"""
Analyze the following tagging errors in the provided sentence:
Sentence: "{sentence}"

Errors:
{error_descriptions}

Here is an example:
WE HAVE A DATE FOR THE RELEASE OF RAGNAROK ONLINE 2 ( beta anyway )
September 16 - 18 , this was announced by Gravity CEO Kim Jung - Ryool on
either 16th or 17th of july and as i don't want to take someone's
credits i got it here ^^ GameSpot

Number of errors: 7
WE
PRON
HAVE
VERB
A
DET
DATE
NOUN
FOR
ADP
THE
DET
RELEASE
NOUN
OF
ADP
RAGNAROK
PROPN
ONLINE
C: ADJ
P: NOUN
**** Error
2
NUM
(
PUNCT
beta
NOUN
anyway
ADV
)
PUNCT
September
PROPN
16
NUM
-
C: SYM
P: PUNCT
**** Error
18
NUM
,
PUNCT
this
PRON
was
AUX
announced
VERB
by
ADP
Gravity
C: PROPN
P: NOUN
**** Error
CEO
NOUN
Kim
PROPN
Jung
PROPN
-
PUNCT
Ryool
PROPN
on
ADP
either
CCONJ
16th
C: NOUN
P: ADJ
**** Error
or
CCONJ
17th
C: NOUN
P: ADJ
**** Error
of
ADP
july
C: PROPN
P: ADV
**** Error
and
CCONJ
as
SCONJ
i
PRON
do
AUX
n't
PART
want
VERB
to
PART
take
VERB
someone
PRON
's
PART
credits
NOUN
i
PRON
got
VERB
it
PRON
here
ADV
^^
C: SYM
P: PUNCT
**** Error

output
[
  {{
    "word": "ONLINE",
    "correct_tag": "ADJ",
    "predicted_tag": "NOUN",
    "explanation": "The word 'ONLINE' is often used as an adjective modifying a noun like RAGNAROK. However, it can also function as a noun in other contexts.",
    "category": "Ambiguity (ADJ/NOUN)"
  }},
  {{
    "word": "-",
    "correct_tag": "SYM",
    "predicted_tag": "PUNCT",
    "explanation": "The hyphen here functions as a symbol indicating a range rather than punctuation.",
    "category": "Punctuation vs Symbol"
  }},
  {{
    "word": "Gravity",
    "correct_tag": "PROPN",
    "predicted_tag": "NOUN",
    "explanation": "Gravity here is the name of a specific company, thus a proper noun.",
    "category": "Proper Noun vs Common Noun"
  }},
  {{
    "word": "16th",
    "correct_tag": "NOUN",
    "predicted_tag": "ADJ",
    "explanation": "Ordinal numbers can function as adjectives or nouns depending on context.",
    "category": "Numeral/Ordinal Ambiguity"
  }},
  {{
    "word": "17th",
    "correct_tag": "NOUN",
    "predicted_tag": "ADJ",
    "explanation": "Similar to '16th', this ordinal number can act as either an adjective or a noun.",
    "category": "Numeral/Ordinal Ambiguity"
  }},
  {{
    "word": "july",
    "correct_tag": "PROPN",
    "predicted_tag": "ADV",
    "explanation": "July is a proper noun referring to a month; misclassification may stem from lowercase usage.",
    "category": "Proper Noun Misclassification"
  }},
  {{
    "word": "^^",
    "correct_tag": "SYM",
    "predicted_tag": "PUNCT",
    "explanation": "Emoticons like '^^' should be categorized as symbols rather than punctuation.",
    "category": "Punctuation vs Symbol"
  }}
]
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TagExplanations,
        },
    )
    if response and response.parsed:
        return response.parsed
    else:
        print("❌ Parsing failed. Check the raw response below:")
        print(response.text if response else "No response received.")
        return None

def sentence_segmentation(few_shots:str, sentence_to_segment)->Optional[SentenceSegmentation]:
    prompt = f"""
    Here is an example for a sentence and it's segmentation:
    {few_shots}
    • Split every punctuation mark (comma, period, hyphen, slash, parentheses…)
  into its own token.
• Split email addresses into: local-part, @, domain-parts, . , top-level-domain.
• Split URLs on every “://”, “/”, “.” and query symbol.
• Keep contractions exactly as in the input; do NOT invent extra tokens.
    Segment the following sentence into words:
    Sentence: "{sentence_to_segment}"
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': SentenceSegmentation,
        },
    )
    if response and response.parsed:
        return response.parsed
    else:
        print("❌ Parsing failed. Check the raw response below:")
        print(response.text if response else "No response received.")
        return None


def batch_segment(sentences: List[str], few_shots: str, batch_size=5) -> Optional[SentenceSegmentations]:
    """
    Segments sentences in batches using LLM calls.
    """
    segmented_sentences = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="🪓 segment"):
        batch = sentences[i:i + batch_size]
        batch_str = '\n'.join(batch)
        prompt = f"""
        Here are some examples of sentence segmentation:
        {few_shots}

        Now, segment the following sentences:
        {batch_str}
        """
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': SentenceSegmentations,
            },
        )

        if response and response.parsed:
            segmented_sentences.extend(response.parsed.segmentations)
        else:
            print(f"❌ Segmentation failed for batch starting at index {i}")
    return segmented_sentences


def batch_pos_tagging(segmented_sentences: List[List[str]], batch_size=5) -> Optional[List[TaggedSentences]]:
    """
    Tags segmented sentences with POS tags in batches.
    """
    tagged_sentences = []
    for i in tqdm(range(0, len(segmented_sentences), batch_size), desc="🏷️ tag"):
        batch = [" ".join(s.tokens) for s in segmented_sentences[i:i + batch_size]]
        batch_str = '\n'.join(batch)
        prompt = f"""
        Tag the following sentences with the UD POS tagset:
        {batch_str}
        """

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': TaggedSentences,
            },
        )

        if response and response.parsed:
            tagged_sentences.extend(response.parsed)
        else:
            print(f"❌ POS tagging failed for batch starting at index {i}")
    return tagged_sentences

# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?\n"
    example_text += "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            # TODO: Retrieve tokens and tags from each sentence:
            for word,tag in zip(s.sentence_words,s.sentence_tags):
                token = word  # TODO
                tag = tag    # TODO
                # Handle potential None for pos_tag if the model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")