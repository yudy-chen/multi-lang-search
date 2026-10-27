"""Scratch: what the model named in `PICTURE_MODEL` does with the app's own body.

Run it whenever `PICTURE_MODEL` is doubted or changed. `PICTURE_MODEL`'s name is the one
thing that decides whether the picture button works at all, and it is not a name that can
be guessed from the family: `gemini-flash-lite-latest` is on the same listing as the
models that return pictures and `generateContent` is on both, but asked for an image it
answers with prose about the picture it was sent -- measured, and it is what this prints.
Every name that does return one carries `-image`, so the listing is printed filtered to
those names, the `-lite` names beside them for contrast, and then one small picture is
sent to the name the app actually holds.

The name, the base URL, the prompt and the body are read out of `main.py` rather than
imported, and the source picture is one the app itself kept, so nothing here can drift away
from what the app does -- the prompt especially, because the same model and the same picture
answer with prose instead of a picture for a short "translate this" line, which is what this
probe first sent and what made it read as a model that cannot edit pictures. (Importing the
app would also pull in gradio, which costs about nineteen seconds.)
"""

import ast
import base64
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
KEY = os.getenv("GEMINI_KEY", "").strip()


def app_strings():
    """`main.py`'s module-level string settings, read without importing the app."""
    module = ast.parse((HERE / "main.py").read_text(encoding="utf-8"))
    names = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return names


SETTINGS = app_strings()
NAMED = SETTINGS["PICTURE_MODEL"]
BASE = SETTINGS["PICTURE_BASE_URL"]
# `python _model_check.py <picture>` sends that one instead: the app's cards carry a
# 320x320 `_OL.jpg` and its detail panel a bigger `og:image`, and which of the two the
# model will edit is worth asking about separately when one of them is doubted.
SOURCES = sorted((HERE / SETTINGS["PICTURE_DIR"]).glob("*-source.*"))
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCES[0]

print(f"PICTURE_MODEL in main.py: {NAMED}", flush=True)
print("key:", "set" if KEY else "MISSING", flush=True)
listing = requests.get(f"{BASE}/models", headers={"x-goog-api-key": KEY}, timeout=60)
print("GET /models:", listing.status_code, flush=True)
if listing.ok:
    for model in listing.json().get("models", []):
        name = model["name"].removeprefix("models/")
        if "lite" in name or "image" in name or "flash-latest" in name:
            print(
                f"  {name:<34} {','.join(model.get('supportedGenerationMethods', []))}",
                flush=True,
            )

content = SOURCE.read_bytes()
PROMPT = SETTINGS["IMAGE_TRANSLATE_PROMPT"].format(language="English")
print(f"\nsource: {SOURCE.name}, {len(content):,} bytes", flush=True)
call = requests.post(
    f"{BASE}/models/{NAMED}:generateContent",
    headers={"x-goog-api-key": KEY},
    json={
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(content).decode(),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    },
    timeout=240,
)
print(f"POST {NAMED}:generateContent: {call.status_code}", flush=True)
try:
    body = call.json()
except ValueError:
    print(call.text[:400], flush=True)
    raise SystemExit(1)
if call.status_code != 200:
    print(json.dumps(body, ensure_ascii=False)[:400], flush=True)
    raise SystemExit(1)
picture = None
candidates = body.get("candidates") or []
if not candidates:
    # A blocked prompt comes back with no candidate at all and a `promptFeedback` instead.
    print("  no candidates in the reply; the whole answer:", flush=True)
    print(f"  {json.dumps(body, ensure_ascii=False)[:900]}", flush=True)
parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
if candidates and not parts:
    # An answer with no parts is a refusal or a safety stop, and the body is the only place
    # that says which: print it whole rather than raising a KeyError on `parts`.
    print("  no parts in the reply; the whole answer:", flush=True)
    print(f"  {json.dumps(body, ensure_ascii=False)[:900]}", flush=True)
for part in parts:
    if "inlineData" in part:
        data = base64.b64decode(part["inlineData"]["data"])
        picture = len(data)
        print(f"  inlineData: {part['inlineData']['mimeType']}, {len(data):,} bytes", flush=True)
    if part.get("text"):
        print(f"  text: {part['text'][:300]}", flush=True)
print("finishReason:", (candidates[0] if candidates else {}).get("finishReason"), flush=True)
if picture:
    print(f"\n{NAMED} returns pictures with this body.", flush=True)
elif parts:
    print(
        f"\nno picture came back: {NAMED} answers about the picture instead of editing it, "
        "so the overlay would read `returned no picture`. Name an `-image` model in "
        "`PICTURE_MODEL` (the listing above says which those are).",
        flush=True,
    )
else:
    print(
        f"\nno picture came back and nothing else either: {NAMED} answered this picture with "
        "an empty candidate, so the overlay would read `returned no picture` and the body "
        "above is the whole of what it said. An `-image` name is not a promise for every "
        "picture -- try `python _model_check.py <another picture>` before blaming the name.",
        flush=True,
    )
