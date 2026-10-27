"""Scratch: is the endpoint answering at all, and which key is the app holding?

Run this when the panel stays Chinese and the console says a model was rate limited. It
answers the two questions that decide what to do next, and nothing else: whether the token
is being refused right now (a 429 for every model, or for one) and whether the key the
app holds is the key that works elsewhere.

It reads the same four names `main.py` reads -- `AI_BASE_URL`, `AI_API_KEY` (or the older
`AMD_AI_TOKEN`), `QUERY_MODEL`, `RESULT_MODEL` -- and falls back to the Radeon defaults, so
it follows the app to another endpoint rather than asking the one it has left.

One tiny request per model, because a rate limit is a property of the token and the moment:
`DeepSeek-V4-Flash` answering `ok` here means the panel's Chinese was a moment that has
passed, and a 429 here means the token is still being refused and no rewrite of the app's
prompts will help.
"""

import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

BASE = (os.getenv("AI_BASE_URL") or "").strip() or "https://developer.amd.com.cn/radeon/api/v1"
ASKED = list(
    dict.fromkeys(
        (
            (os.getenv("QUERY_MODEL") or "").strip() or "DeepSeek-V4-Flash",
            (os.getenv("RESULT_MODEL") or "").strip() or "Gemma-4-31B-IT",
        )
    )
)


def masked(secret):
    """Enough of a key to tell two apart, and not enough to be a key."""
    if not secret:
        return "MISSING"
    return f"{secret[:6]}...{secret[-4:]} ({len(secret)} chars)"


print("--- .env, names only ---", flush=True)
for line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        print(f"  {line.split('=', 1)[0].strip()}", flush=True)

app_key = os.getenv("GEMINI_KEY", "").strip()
print("\n--- the Gemini key the app holds ---", flush=True)
print("GEMINI_KEY:", masked(app_key), flush=True)
script = HERE.parent.parent / "test_convert.py"
if script.exists():
    literal = re.search(r'api_key="([^"]+)"', script.read_text(encoding="utf-8"))
    if literal:
        print(f"{script.name}:", masked(literal.group(1)), flush=True)
        print("same key:", app_key == literal.group(1), flush=True)

token = (os.getenv("AI_API_KEY") or os.getenv("AMD_AI_TOKEN") or "").strip()
print(f"\n--- the token for {BASE} ---", flush=True)
print("AI_API_KEY / AMD_AI_TOKEN:", masked(token), flush=True)
try:
    listing = requests.get(
        f"{BASE}/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    print(f"GET /models: {listing.status_code}", flush=True)
    if listing.ok:
        names = [model.get("id") for model in listing.json().get("data", [])]
        print(f"  {len(names)} model(s): {', '.join(str(name) for name in names)}", flush=True)
except Exception as error:  # noqa: BLE001 - a probe reports, it does not raise
    print(f"GET /models failed: {type(error).__name__}: {str(error)[:200]}", flush=True)

print("\n--- one small request per model, right now ---", flush=True)
for model in ASKED:
    client = OpenAI(base_url=BASE, api_key=token, timeout=60, max_retries=0)
    started = time.time()
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word ok."}],
        )
        said = (reply.choices[0].message.content or "").strip()
        print(f"{model}: {time.time() - started:.1f}s -> {said[:60]!r}", flush=True)
    except Exception as error:  # noqa: BLE001 - a refusal is the answer being looked for
        print(
            f"{model}: {type(error).__name__} after {time.time() - started:.1f}s -> "
            f"{str(error)[:300]}",
            flush=True,
        )
