# multi-lang-search

Search [Momoshop](https://www.momoshop.com.tw) -- one of Taiwan's largest online malls --
in **English** or **Indonesian**. Every listing is translated back into the language you
searched in, card by card, as the answers land.

Built for the AMD Radeon Cloud hackathon: the chat models are served from the Radeon
endpoint (ROCm), the picture step from Gemma. Gradio, Python 3.12.

## What it does

- **Your query, in Chinese.** `kacang tanah` becomes `花生`, and Momoshop's own search runs
  on that.
- **Cards, translated and checked.** One request per product, so cards appear as their
  translations land. Every reply is scanned for leftover Chinese and only the failures
  are asked again.
- **The product page too.** Click a card: the panel is rebuilt from the page's own JSON
  payload and translated line by line under a `Translating ...` veil, with a button to
  ask again for the lines that stayed Chinese.
- **The wording in the photos.** Click a picture, then `translate the wording into
  Indonesian`. With a `GEMINI_KEY` an image model edits the picture in place; without
  one, Gemma reads the wording off it and Pillow letters the translation back on. Those
  pictures land in `translated_pictures/`.

## Quick start

From `multi-lang-search/`, with Python 3.12 and network access to Momoshop and whichever
chat endpoint `.env` names (the Radeon one by default):

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate  elsewhere
pip install -r requirements.txt
```

Then the keys:

```bash
copy .env.example .env           # Windows;  cp .env.example .env  elsewhere
# open .env and put the token from the Radeon Cloud console in AMD_AI_TOKEN
python main.py
```

`AMD_AI_TOKEN` is the only value you have to fill in. `GEMINI_KEY` is optional and only
the picture step uses it. Open the URL Gradio prints -- usually http://127.0.0.1:7860.

`python main.py` is the plain Gradio server. A hosted runner does not run the file: it
imports it and serves the top-level ASGI `app` exported at the bottom, so
`uvicorn main:app --port 7860` serves exactly the same app (`application` is the same
object, for runners that look for that name).

## Any other endpoint

Every chat model goes through one OpenAI-compatible client, so the endpoint is
configuration, not code: fill these in `.env` and no line of `main.py` moves. Unset
means the Radeon endpoint the numbers in [NOTES.md](NOTES.md) were measured on.

| where | `AI_BASE_URL` | key | `QUERY_MODEL` / `RESULT_MODEL` |
| --- | --- | --- | --- |
| Radeon (default) | *(unset)* | `AMD_AI_TOKEN` | the defaults, `DeepSeek-V4-Flash` / `Gemma-4-31B-IT` |
| DeepSeek | `https://api.deepseek.com` | `AI_API_KEY` | `deepseek-flash` for both |
| Gemini, free tier | `https://generativelanguage.googleapis.com/v1beta/openai/` | `AI_API_KEY`, and `GEMINI_KEY` for the picture step | a flash name from `GET .../openai/models` |
| OpenRouter | `https://openrouter.ai/api/v1` | `AI_API_KEY` | a `:free` text model id |
| Ollama, on this machine | `http://localhost:11434/v1` | any non-empty string | the model you pulled |

`DETAIL_MODEL` follows `QUERY_MODEL` unless it is set too. One caveat: DeepSeek's API
takes text only, so without a `GEMINI_KEY` nothing there can read a picture -- that step
then says it could not, rather than guessing. Set `IMAGE_MODEL` to a vision model on the
same endpoint, or keep the Gemini key (which also covers text, if the endpoint is
Gemini's). `python _endpoint_check.py` asks whichever endpoint `.env` names, one request
per model, and prints what came back. `python _env_check.py` asks the same question
offline: it stands a fake endpoint up on a local port, points `main.py` at it and calls
every path, so a switch can be checked without spending a key.

## Test it in two minutes

1. Query `peanut`, Language **English**, **Search**. The query is translated first, then
   cards fill in one by one -- a minute or so for the five the testing cap allows.
2. Switch to **Indonesia**, search `kacang tanah`. The same products come back in
   Indonesian.
3. **Click any card.** The panel opens under a veil reading
   `Translating ... into Indonesian (3 of 13 line(s))` and fills in as the requests land.
4. **Click a picture** in the panel, then `translate the wording into Indonesian`. The note
   under the button says which model does the job -- with a `GEMINI_KEY` the image model
   returns the same picture in Indonesian, without one Gemma reads the wording and the
   picture is lettered again here.
5. Language **Chinese**, query `花生`: nothing is translated -- the page is shown as
   scraped.

Measured on the live endpoint, so you know what "working" looks like: one listing 4-40s
depending on the model and the token's load, a detail panel 23-42s for 12-13 lines, a
picture read in 9-76s and lettered in 4-11s.

## Three things worth knowing before you judge it

- **Only five products are translated.** `TRANSLATE_LIMIT = 5` at the top of `main.py` is
  a testing cap; the page says how many were left as scraped. Set it to `0` for the whole
  page.
- **A busy endpoint is slow, not broken.** Three requests in flight, a 60s ceiling per
  request, retries per product, and a 20s back-off when the token answers `429`. The app
  keeps going and narrates it in the console.
- **Nothing is trusted.** A reply that still contains Chinese counts as untranslated: it
  is asked again, and whatever survives is counted on the panel instead of passing
  silently.

## Troubleshooting

| What you see | What it is |
| --- | --- |
| Cards or the panel stay Chinese, console mentions `429 ... token_concurrency_rate_limit_exceeded` | The token is at its concurrency limit. The app backs off 20s and retries; `python _endpoint_check.py` tells you whether it is still refusing. |
| The picture step answers `402 ... prepayment credits are depleted` | The Gemini project behind `GEMINI_KEY` has no balance. Delete the key -- the app still re-letters pictures with Gemma and Pillow. |
| A card or a line stays Chinese after the retries | The console counts it; the panel's retry button asks again for exactly those lines. A second search is the blunt fix. |
| `ModuleNotFoundError: gradio` | The venv is not active, or `pip install` ran against another interpreter. |
| A runner refuses with `PYTHON_ENTRYPOINT_NOT_FOUND` | It imports `main.py` and wants a top-level ASGI app. `app` and `application` are exported at the bottom of the file; `uvicorn main:app` is that app. |

## Files

| | |
| --- | --- |
| `main.py` | The app: search, cards, detail panel, picture step -- and, at the bottom, the ASGI `app` / `application` a hosted runner serves instead of running the file. Models, the endpoint, timeouts and concurrency are constants at the top; the endpoint, its key and the model names can be overridden in `.env`. |
| `requirements.txt` | The six packages, pinned to what it ran on. |
| `.env.example` | The endpoint, its key and the model names, with what each one is for. Copy to `.env`; `.env` itself is git-ignored. |
| `NOTES.md` | The long version: every measurement, the endpoint's behaviour, and why the design is what it is. |
| `_panel_check.py` | Offline check of the panel's frames -- `python _panel_check.py`, no network. |
| `_endpoint_check.py` | Is the token in `.env` answering, and which models does that endpoint list. |
| `_env_check.py` | The endpoint switch without a provider: a fake endpoint on a local port, every path called against it (its fixture is `_mock_provider.py`). |
| `_model_check.py` | What the image model does with the app's own request body. |
| `_picture_dbg.py` | One picture through the picture step, with the shape of the reply. |

## How it works, in one paragraph

`DeepSeek-V4-Flash` translates the query into Traditional Chinese, `requests` fetches
Momoshop's search page, and `Gemma-4-31B-IT` translates each listing in a request of its
own. The detail panel is read out of the product page's own JSON payload, chunked to
`DETAIL_CHUNK_CHARS` and translated by `DeepSeek-V4-Flash` as it streams. The picture
step is `gemini-3.1-flash-image-preview` on `GEMINI_KEY`, or Gemma plus Pillow without
one. All of that, including the numbers behind it, is in [NOTES.md](NOTES.md).
