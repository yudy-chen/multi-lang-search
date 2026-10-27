"""Multilanguage e-commerce search: type a query, get product cards back."""

import base64
import html
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urljoin

import gradio as gr
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageFont

load_dotenv(Path(__file__).resolve().parent / ".env")

momo_search = "https://www.momoshop.com.tw/search/"

# The endpoint this was built on; what an empty `AI_BASE_URL` means.
_DEFAULT_AI_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"

# The endpoint every chat model here is served from. Any server that speaks
# `/chat/completions` will do -- the Radeon endpoint this was built and measured on,
# DeepSeek (`https://api.deepseek.com`), Gemini's OpenAI-compatible layer
# (`https://generativelanguage.googleapis.com/v1beta/openai/`), OpenRouter
# (`https://openrouter.ai/api/v1`), a local Ollama (`http://localhost:11434/v1`). It is
# read from `.env` so the app can outlive one provider's token; an empty or missing
# value is the Radeon endpoint, so a `.env` that only carries the old token behaves
# exactly as it did before this was movable.
AI_BASE_URL = (os.getenv("AI_BASE_URL") or "").strip() or _DEFAULT_AI_BASE_URL

# The key for `AI_BASE_URL`. `AI_API_KEY` is the name that fits any endpoint;
# `AMD_AI_TOKEN` is what the Radeon console calls it and is read second, so the older
# `.env` keeps working unchanged.
AI_API_KEY = (os.getenv("AI_API_KEY") or os.getenv("AMD_AI_TOKEN") or "").strip()

# For these source languages the query is translated to Traditional Chinese
# before it is sent to the Momoshop search endpoint.
TRANSLATE_TO_TRADITIONAL_CHINESE = {"English", "Indonesia"}
TARGET_LANGUAGE = "Traditional Chinese"

# Chat models, one per job: the search query is a single short string that the user
# waits on with an empty page, while the results are a whole page of listings
# translated one request at a time as they arrive. Both are served from the same token,
# so neither is a spare for the other: they share the token's rate limit and get
# throttled together -- which is why changing the provider changes all of them at once.
#
# The defaults below are the names the Radeon endpoint serves, and the names every
# number in NOTES.md was measured on. Another endpoint serves other names, so all four
# are read from `.env` as well; an empty or missing value is the default.
QUERY_MODEL = (os.getenv("QUERY_MODEL") or "").strip() or "DeepSeek-V4-Flash"

RESULT_MODEL = (os.getenv("RESULT_MODEL") or "").strip() or "Gemma-4-31B-IT"

# The detail panel is translated with the query model rather than the result model:
# its text is a handful of blocks and the user is sitting on a click, so it wants the
# model that answers in seconds -- one listing measured 4.3s and 11.3s on this model
# against 38.8s on `RESULT_MODEL`, and 2.9s against 51.1s when `RESULT_MODEL` was
# `MiMo-V2.6-Flash`, which is the span the split was drawn across. Unset, it follows
# whatever `QUERY_MODEL` was set to.
DETAIL_MODEL = (os.getenv("DETAIL_MODEL") or "").strip() or QUERY_MODEL

# "The same picture with the wording changed" is one request to an image model: the
# picture goes out beside a prompt, and the picture with its Chinese wording translated
# comes back, its layout and product graphics included. The Radeon endpoint serves no
# such model -- measured: no model it lists outputs a picture (the modalities on its
# listing are text, decision, ocr and audio),
# /v1/images/generations refuses a chat model by name ("Model Gemma-4-31B-IT is a chat
# model and cannot be used with /v1/images/generations") and /v1/images/edits validates
# its request against an image model that is not served -- so the picture step is served
# from Google's image models, on a key of the user's own. `PICTURE_KEY` is
# read once here because it is a property of the process, and a process without one has
# no picture model at all: `image_translated` then falls back to the endpoint's own
# Gemma, and the panel says so.
#
# `PICTURE_MODEL` has to name a model that returns pictures, and the family is not a
# guide to that: on this key's own listing, `gemini-flash-lite-latest` sits beside the
# image models with `generateContent` on it as well, and asked for a picture with the body
# below it answers with prose about the picture it was given and no image part at all --
# measured, and what `_model_check.py` prints. Every name that does return one carries
# `-image`, though an `-image` name is not a promise for a particular picture (below); see
# the README for what the listing carried when this was written.
#
# What ships is the model measured end to end on the picture this app is handed: a 320x320
# `_OL.jpg` off a card, the prompt below, and the body below, answered `200` with a
# `1024x1024` PNG in `inlineData` and the badge on the product re-lettered (see the README
# for that run). The newer `-image` names are not interchangeable with it even though they
# are on the same listing: `gemini-3.1-flash-image-preview` answers this same picture with a
# `200` whose candidate has no parts at all -- no picture, no text, and nothing in the body
# to explain it (`_model_check.py` prints that body) -- while it edits a larger picture
# happily. So a name goes in here only after `_model_check.py` has returned a picture for
# *this* picture, which is the check the probe takes a path for.
PICTURE_MODEL = "gemini-3.1-flash-image-preview"
PICTURE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PICTURE_KEY = os.getenv("GEMINI_KEY", "").strip()

# Gemma can be shown a picture, which is the half of that job a chat endpoint can do,
# and it is what the picture step becomes without a `GEMINI_KEY`: it takes a
# picture as *input*, so it can read the Chinese wording off one and where each line
# sits, but it answers in text, so the translation is lettered back onto the same
# picture with Pillow afterwards. It is not the only model the Radeon endpoint lists that
# takes a picture (`DeepSeek-V4-Flash-Vision-Exp` and the two `MiniCPM-V` names do too),
# but it is the one this step was measured on. See `reletter_image`.
#
# This one has to name a model that accepts a picture, and that is the single thing
# another endpoint does not promise: DeepSeek's API takes text only (its `-vision-exp`
# name is retired and served by a text model), so pointing `AI_BASE_URL` there leaves
# this step with a model that refuses the request. It says so in the lightbox and does
# not guess -- and it is a reason to set `IMAGE_MODEL` to a vision model on the same
# endpoint, or to keep a `GEMINI_KEY`, when the chat endpoint moves.
IMAGE_MODEL = (os.getenv("IMAGE_MODEL") or "").strip() or "Gemma-4-31B-IT"

# A picture is a much bigger request than a listing is (the one measured was 366 KB
# of base64 to read), so picture requests get a ceiling of their own, whether they go to
# the picture model or are read and re-lettered here.
IMAGE_TIMEOUT = 240.0

# How long one request may take before the client gives up on it. A one-product
# request answers in about thirty seconds on a loaded endpoint (measured 31.0s for a
# single listing, and 31.0s / 33.2s / 59.9s for three of them in flight), so the
# ceiling sits above that: the request that landed at 59.9s was a real answer that a
# 45s ceiling would have thrown away and asked for again. A ceiling is not free -- a
# request we gave up on is still generated on the endpoint and keeps holding its
# slot -- which is why the page asks for one product at a time now: a small request
# is generated in seconds, so little of the endpoint is left working on ghost
# answers. A 30s ceiling used to kill the answer behind a reported Chinese card
# moments before it arrived (it took 32.3s), which is what left cards untranslated.
REQUEST_TIMEOUT = 60.0

# For these languages the *results* are translated back for the user.
# "Chinese" is missing on purpose: the Momoshop listings are already Chinese.
RESULT_LANGUAGE = {"English": "English", "Indonesia": "Indonesian"}

# The endpoint answers 429 "rate limit exceeded; please retry later" once too many
# requests share the token -- which a request that is still holding its slot after
# we gave up on it also causes. Retrying into that with more concurrent requests
# only makes the limit worse, so a request that sees it raises this flag: the
# retries then go out one request at a time instead of walking into the same wall.
RATE_LIMITED = threading.Event()

# One product per request, which is how this page is tuned. The endpoint's cost is
# mostly per request rather than per product: measured back to back on the live
# endpoint, a request holding one listing answered in 31.0s, a second one in 33.2s,
# while a request holding three listings was still generating after 150s and one
# holding six did not answer either. So the page asks for its products one at a
# time and shows them one by one. Raise this to 6 when the endpoint has headroom and
# the page has to come back in a fifth of the requests: the cards then arrive in
# waves instead of one by one, and a stall costs six cards instead of one.
TRANSLATE_BATCH = 1

# How many requests are in flight at once. Three requests of one listing each were
# sent together to the loaded endpoint and all three landed (31.0s, 33.2s, 59.9s),
# so the page gained a card about every twenty seconds instead of every thirty. That
# is the trade this number makes: the endpoint shares its capacity between the
# requests generating on it, so beyond a few, concurrency buys no throughput and
# only pushes every request closer to the ceiling, where it is abandoned while still
# generating -- and those ghosts hold the slots the retries need. Measured: three
# concurrent *six-listing* batches each ran past 45s while the one-listing ones came
# back, so overlap what is small and few, not what is big.
TRANSLATE_WORKERS = 3

# A reply can also come back with only part of its text translated, or with the
# listing still in Chinese, and a request that times out leaves its product in
# Chinese. `translate_items` therefore asks again for exactly those products and
# only those -- a product that already arrived is never sent a second time. Three
# attempts are enough for a request that timed out once to land on a retry; whatever
# is still in Chinese after the last attempt is counted in the console rather than
# passing silently.
TRANSLATE_ATTEMPTS = 3

# How long the endpoint is left alone before a retry wave goes out. The detail half of
# the flow has always waited between its attempts (`DETAIL_RETRY_WAIT`); the listing half
# did not, which is the wrong shape for the expensive request: a listing that ran past
# the ceiling is still being generated on the endpoint, so a retry that leaves
# immediately runs beside its own ghost and asks for a slot the endpoint has already
# given away -- the wave then reads `429 ... token_concurrency_rate_limit_exceeded` and
# asks again, and a five-product search becomes fifteen requests where five were needed
# (`TRANSLATE_LIMIT` 5 x `TRANSLATE_ATTEMPTS` 3, counted with the endpoint faked out).
# The wait grows with the attempt number, which gives the endpoint twice as long to
# finish what we walked away from before the next wave; after a 429 the longer one is
# used, because there the token itself is refusing the wave's concurrency.
TRANSLATE_RETRY_WAIT = 5.0
TRANSLATE_RATE_LIMIT_WAIT = 20.0

# A page of listings is more requests than the flow needs while it is being tested:
# only the first few products are sent to the endpoint, and the rest are put on the
# page in Chinese rather than dropped, so the count of what the search found is still
# the truth. 0 translates the whole page, which is what the finished app should do.
TRANSLATE_LIMIT = 5

# Result cards are only put on the page once their text is translated, so the page
# never shows the listings in Chinese and swaps them for the translation afterwards:
# every product that lands appends its own card. Set this to False to show the whole
# page in Chinese straight away and fill the translations in as they arrive, which
# is more useful when a page takes minutes and the listings are recognisable from
# their images.
REVEAL_TRANSLATED_ONLY = True

# The query translation is the one thing the user waits for with an empty page, so
# its deadline sits below the client's own ceiling: if the endpoint has not
# answered a single short line in 30s it is saturated, and the search is better
# off going out with the query as typed.
QUERY_DEADLINE = 30.0

# How long one Momoshop page may take to come back: the search page, and the
# product page the detail panel fetches when a card is clicked.
PAGE_TIMEOUT = 30.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}

# The search page embeds its results as schema.org JSON-LD inside a <script> tag.
JSON_LD_SCRIPT = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
    re.DOTALL,
)

# The listings are Chinese, so leftover Chinese is how a product that did not
# actually get translated is spotted.
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")

# A product page is a Next.js app, and the browser renders it from data the server
# already put in the page it sent: the React flight payload, one
# `self.__next_f.push([1, "..."])` per chunk, holding JSON whose quotes the
# surrounding script string has escaped a second time. The blocks the page shows as
# 「商品特色」, 「商品規格」 and 「商品說明」 are in there, so they can be read out
# without running the page -- which matters, because the richest part of 「商品特色」
# is an iframe on a host that answers 403 to anything but a browsing browser (see
# FEATURE_IFRAME_BLOCKED_HOST below).
RSC_PUSH_PATTERN = re.compile(
    r'self\.__next_f\.push\(\[\s*1\s*,\s*("(?:[^"\\]|\\.)*")\s*\]\)'
)

# One level of that escaping: \n, \" and \uXXXX. The payload mixes literal Chinese
# with \uXXXX escapes depending on the product, so both have to come out the same.
JSON_ESCAPE_PATTERN = re.compile(r"\\(u[0-9a-fA-F]{4}|[nrt\"\\/])")
JSON_ESCAPE_SIMPLE = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/"}

# The content blocks the page renders, in the order the payload carries them: a
# heading, a bullet list, a paragraph. A product page repeats itself (the same
# block shows in more than one part of the tree), so the reader drops duplicates.
PAYLOAD_TITLE = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')
PAYLOAD_FORMAT = re.compile(r'"format"\s*:\s*\[([^\]]*)\]')
PAYLOAD_TEXT = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')
PAYLOAD_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')

# The same page on the legacy endpoint serves 商品規格 and 商品說明 as plain HTML, and
# it is on the host that answers normally from here. It fills the panel in for the
# products whose own page came back without its payload (measured: some pages are a
# 40 KB shell with no data in them).
GOODS_INFO_URL = "https://www.momoshop.com.tw/goods/dsgoodsinfo.jsp"
PRODUCT_CODE_PATTERN = re.compile(
    r"(?:i_code=|goodsCode=|/product/|/goodsDetail/)([A-Za-z0-9]+)"
)
# A fragment this short is the site's "no such product" answer rather than content.
GOODS_INFO_MIN_CHARS = 1_000

# 「商品特色」 is the one section whose content is not in the product page at all: the
# page renders an iframe, and that iframe shows a static page the supplier wrote, at
# /expertimg/<4>/<3>/<3>/mobile/<code>.html -- the groups being the product code zero
# padded to ten digits (`changeGoodsCode` in the iframe's own shell pads it, and
# `changeGoodsCodePath` splits it 4/3/3). That shell is what the path was read off, but
# it is not what gets fetched: the file is on the content hosts that serve the rest of
# the site's images, which answer normal requests, while the shell's own host
# (m.momoshop.com.tw) answers 403 to them as often as it answers 200.
FEATURE_PAGE_SHAPE = "/expertimg/{path}/mobile/{code}.html"
FEATURE_PAGE_HOSTS = ("www.momoshop.com.tw", "image.momoshop.com.tw")
# A page this short is the site saying there is nothing for that code rather than
# content (the "nothing here" answers measured 117 characters, a real one 759+).
FEATURE_PAGE_MIN_CHARS = 200
# The rich pages are mostly pictures, so the panel shows a strip of them instead of
# printing a wall of image tags.
FEATURE_IMAGE_LIMIT = 6
FEATURE_IMAGE_PATTERN = re.compile(r"""src=["']([^"']+)["']""", re.IGNORECASE)

# A picture in the panel can be opened bigger, and its wording re-lettered in the
# target language: the server fetches that one picture to read it, so the download has
# a ceiling, and the picture it hands back is capped so the panel does not carry
# megabytes of base64. The re-lettered picture keeps the original's shape.
IMAGE_MAX_BYTES = 12_000_000
IMAGE_MAX_WIDTH = 1200
# A line the model places smaller than this is too small to letter legibly, so it is
# left as it is rather than drawn over, and a line it places taller or wider than this
# is not a line at all but a guess -- see `line_box`.
IMAGE_MIN_BOX = (40, 12)
IMAGE_MAX_BOX_HEIGHT = 0.30
IMAGE_MAX_BOX_WIDTH = 0.98
# Wording written vertically is common on these pictures, and a line as tall as this
# many times its width is that rather than a line of Latin text: the translation would
# have to be drawn sideways to fit it, so the line is left as it is.
IMAGE_MAX_BOX_ASPECT = 2.2
# The type a block is lettered in comes down to a fraction of the picture's own height
# before the block is left alone: a fixed size is unreadable on a large picture and
# cramped on a small one.
IMAGE_MIN_FONT_RATIO = 0.022
IMAGE_MIN_FONT = 12
# The re-lettered picture comes back as a JPEG: it is a photograph of a product rather
# than a diagram, and the panel carries it inline as base64, where the size matters.
IMAGE_QUALITY = 88

# Where the picture step leaves what it did. The lightbox shows one picture and the
# browser does not say which one it is: an answer that is the picture it was given, or the
# same picture at a larger size with the Chinese still on it, is a 200 carrying an image,
# and on the panel that looks exactly like a translation. So the picture that went out and
# the picture that came back are both written to this folder, and one block of console
# says what the model was asked, what it said about its own answer, how long it took and
# how much of the picture it changed. See `picture_report`.
#
# It is the name of a folder beside `main.py`, or an absolute path from `.env`, or empty
# for that name -- and it is only the first place tried. A hosted runner unpacks the app
# read-only (Vercel serves the files from `/var/task` and writes only under `/tmp`), so the
# same folder is also tried under the temp directory, and a host where neither works still
# runs the step with the console block as its record. `picture_folder` does the choosing.
PICTURE_DIR = (os.getenv("PICTURE_DIR") or "").strip() or "translated_pictures"

# The extension for each type `image_mime` can report: what is kept on disk is opened by
# name rather than sniffed, unlike the bytes going out to the model and to the panel.
PICTURE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# The reading prompt. Gemma is asked for the wording and where it sits on the picture
# as fractions of its size, so the box survives any resize between reading and
# drawing, and for plain JSON, because the answer is parsed rather than read.
IMAGE_READ_PROMPT = (
    "This is one product picture from a Taiwanese online shop. Read the wording that "
    "is written into the picture itself (usually Traditional Chinese).\n"
    "Reply with JSON and nothing else, like:\n"
    '{"lines":[{"text":"the words as written","x":0.05,"y":0.1,"w":0.9,"h":0.08}]}\n'
    "x and y are the top-left corner of that line as a fraction of the picture's width "
    "and height, w and h its width and height the same way (0 to 1). One entry per "
    "line of wording, in reading order. If the picture carries no wording, reply "
    '{"lines":[]}.\n'
)

# The picture prompt: the whole of what goes with the picture to `PICTURE_MODEL`, which
# is asked for the picture back with its wording changed and for nothing else on it to
# move. The language is filled in from the search, so the one prompt serves every target
# this app offers, and it is the wording the panel promises the user under the picture.
#
# It asks for the Chinese to be replaced rather than for the picture to be translated, and
# that difference was measured on one product picture: "Translate all Chinese text in the
# provided image into clear English. Retain the exact visual layout, product graphics,
# diagram elements, arrows, numbers, and background styling without modifying any physical
# object details" came back as a 200 with a picture whose wording was unchanged -- the
# round badge still read 任選4, with an "ANY" added under it -- where "translate ... every
# piece of Chinese wording ... including the wording printed on the product packets ... so
# that no Chinese character is left in the picture" came back with the badge lettered
# "CHOOSE ANY 4" and the flavour tabs as MATCHA, CHOCOLATE and STRAWBERRY. A picture of a
# product says "do not modify the product", and on one the Chinese is printed on the
# product, so the wording has to say which of the two is the thing being changed.
#
# Measured limit: the print on the packets themselves is still Chinese after a prompt that
# asks for it, on the 640x640 picture of the product as well as on the 320x320 one the
# cards carry -- the model re-letters the large wording and leaves the small print, which
# is a dozen pixels tall on the picture this app is handed. What it does translate is worth
# having and what it leaves is visible in the lightbox, so the reply is taken as it comes.
IMAGE_TRANSLATE_PROMPT = (
    "Translate into {language} every piece of Chinese wording in this picture -- the "
    "wording in the badges, the wording printed on the product packets, and the small "
    "print -- and put the {language} wording where the Chinese was, in the same place, at "
    "the same size, in the same colour and slant, so that no Chinese character is left in "
    "the picture. The products themselves stay as they are: their shapes, colours, "
    "photographs, graphics, arrows and numbers are not to be changed, only the wording on "
    "them replaced."
)

# The panel's own text is translated in chunks: short lines go out numbered, a long
# block goes out on its own so its line breaks survive. The chunk size is what keeps
# one request from growing into a several-minute one on a loaded endpoint, and the cap
# is what keeps a page with a dozen paragraphs from turning one click into a queue of
# requests -- what the cap leaves out is asked for by clicking the panel's button, which
# says how many lines are still in Chinese.
DETAIL_CHUNK_CHARS = 2_400
DETAIL_LONG_CHARS = 400
DETAIL_REQUEST_CAP = 12
# A request the endpoint answered with nothing at all is asked for again this many
# times, with a wait in between, before the Chinese it was to replace is what the panel
# keeps: an empty answer is a timeout or a 429, and on a loaded endpoint the panel's one
# request per piece of the page met both (measured: one timeout left a whole page in
# Chinese with only a line in the console to say why). Asking a third time is the
# button's job.
#
# The wait is the endpoint's to decide, not the panel's: a wait of a few seconds is right
# for an answer that never came, and it is exactly wrong for a 429, because that is the
# token refusing the concurrency this retry is about to ask for again -- measured on a
# click that landed while the search's own translations were still in flight: the panel
# read `429 ... token_concurrency_rate_limit_exceeded` and came back with the same
# Chinese three seconds later. So a refusal waits `TRANSLATE_RATE_LIMIT_WAIT` like the
# result retries do, and only a plain empty answer waits the short one.
DETAIL_ATTEMPTS = 2
DETAIL_RETRY_WAIT = 3.0


def complete(prompt, model=QUERY_MODEL, timeout=REQUEST_TIMEOUT):
    """Send one user message to the endpoint `AI_BASE_URL` names and return the reply.

    `prompt` is either the text of the message or the parts of a multimodal one
    (a text part plus a picture, which is how `IMAGE_MODEL` is asked to read a
    picture). An empty string comes back when the endpoint cannot be reached in
    time, so callers can fall back to the untranslated text instead of hanging.
    """
    client = OpenAI(
        base_url=AI_BASE_URL,
        api_key=AI_API_KEY,
        timeout=timeout,
        max_retries=0,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as error:  # noqa: BLE001 - a stuck request must not hang the app
        print(f"{model} request failed: {error}", flush=True)
        if "429" in str(error):
            RATE_LIMITED.set()
        return ""
    # A reply means the endpoint took a request from this token, so it is handing out
    # concurrency again: the flag is cleared by the next answer rather than waiting for
    # the next search to clear it. A 429 that arrives beside a reply is a wave that is
    # partly over its limit, which is worth testing again -- the flag is a back-off, not
    # a latch.
    RATE_LIMITED.clear()
    return (resp.choices[0].message.content or "").strip()


def complete_again(
    prompt, model=DETAIL_MODEL, timeout=REQUEST_TIMEOUT, attempts=DETAIL_ATTEMPTS
):
    """`complete`, asked again while the endpoint answers nothing at all.

    One request is one piece of the panel, so a piece the endpoint dropped -- a timeout,
    or a 429 while the search's own translations are in flight -- used to leave that part
    of the page in Chinese with nothing but a line in the console to say why. Empty is
    the one answer that is never useful here, so it is the one worth waiting out and
    asking for again, and how long it waits is the endpoint's answer to decide: the short
    `DETAIL_RETRY_WAIT` for an answer that never came, `TRANSLATE_RATE_LIMIT_WAIT` when
    what came back was a refusal (see the comment on `DETAIL_ATTEMPTS`).
    """
    for attempt in range(1, attempts + 1):
        reply = complete(prompt, model=model, timeout=timeout)
        if reply:
            return reply
        if attempt < attempts:
            # `complete` raised the rate-limit flag if this was a 429, and cleared it on
            # any reply, so the flag says whether the wait is for a refusal or for an
            # answer that is merely late.
            refused = RATE_LIMITED.is_set()
            wait = (
                TRANSLATE_RATE_LIMIT_WAIT if refused else DETAIL_RETRY_WAIT * attempt
            )
            print(
                f"{model} answered nothing (attempt {attempt} of {attempts}); "
                f"asking again in {wait:.0f}s"
                + (
                    " -- the token refused that request, so the retry waits out the "
                    "refusal instead of asking into it"
                    if refused
                    else ""
                ),
                flush=True,
            )
            time.sleep(wait)
    # Whatever is still Chinese after the last attempt is the panel's to offer again, and
    # the console says so here rather than leaving the Chinese to be read as a translation
    # that came back the same.
    print(
        f"{model} answered nothing for a piece of this panel after {attempts} attempt(s); "
        "it keeps its Chinese and the panel offers the lines again",
        flush=True,
    )
    return ""


def translate(query, language):
    """Translate the user's search query into `language`.

    The endpoint sometimes stalls without the request ever timing out, and this
    is the one step where the user is staring at an empty page, so the answer is
    only waited for up to a deadline: past it the search runs with the query as
    it was typed instead of leaving the page stuck on the progress bar.
    """
    prompt = f"Translate '{query}' to {language}. Reply with only the translation."
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        reply = pool.submit(complete, prompt).result(timeout=QUERY_DEADLINE)
    except TimeoutError:
        print("query translation stalled, searching with the query as typed", flush=True)
        return query
    except Exception as error:  # noqa: BLE001 - the search still has to run
        print(f"query translation failed: {error}", flush=True)
        return query
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    # Some models like to wrap the answer in quotes.
    return reply.strip().strip("\"'“”").strip() or query


def search_momo(keyword):
    """GET a Momoshop search page and return its raw HTML."""
    url = momo_search + quote(keyword)
    resp = requests.get(url, headers=HEADERS, timeout=PAGE_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def fetch_product_page(url):
    """GET a product page and return its raw HTML, exactly as it came back.

    Nothing is parsed out of it here: this is what `product_detail` reads the
    content out of, and what the panel keeps behind its `raw page source` fold.
    """
    resp = requests.get(url, headers=HEADERS, timeout=PAGE_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def unescape_json(text):
    """Undo one level of JSON string escaping: \\n, \\" and \\uXXXX."""

    def replace(match):
        body = match.group(1)
        if body in JSON_ESCAPE_SIMPLE:
            return JSON_ESCAPE_SIMPLE[body]
        return chr(int(body[1:], 16))

    return JSON_ESCAPE_PATTERN.sub(replace, text)


def rsc_payload(page_html):
    """The React flight payload of a product page, with its script strings decoded.

    Each chunk is a JavaScript string literal holding JSON, so `json.loads` reads
    it wherever the escapes are JSON's own; a literal it will not take (JavaScript
    allows escapes JSON does not) is still readable text, so its quotes are
    stripped and its escapes decoded by hand rather than dropping the chunk.
    """
    chunks = []
    for literal in RSC_PUSH_PATTERN.findall(page_html):
        try:
            chunks.append(json.loads(literal))
        except json.JSONDecodeError:
            chunks.append(unescape_json(literal[1:-1]))
    return "".join(chunks)


def payload_blocks(payload):
    """The content blocks the page renders: headings, bullet lists, paragraphs.

    Read out of the payload as text rather than by walking its JSON, because the
    payload is a component tree whose shape differs per page while the blocks
    themselves (`title`, `format`, `text`) are the same everywhere. Duplicates are
    dropped: the same block appears in more than one branch of the tree.
    """
    plain = unescape_json(payload)
    found = []
    for pattern, kind in ((PAYLOAD_TITLE, "title"), (PAYLOAD_TEXT, "text")):
        for match in pattern.finditer(plain):
            value = unescape_json(match.group(1)).strip()
            if value:
                found.append((match.start(), kind, value))
    for match in PAYLOAD_FORMAT.finditer(plain):
        bullets = [
            unescape_json(item).strip()
            for item in PAYLOAD_QUOTED.findall(match.group(1))
        ]
        bullets = [bullet for bullet in bullets if bullet]
        if bullets:
            found.append((match.start(), "format", bullets))
    found.sort(key=lambda block: block[0])

    blocks = []
    seen = set()
    for _, kind, value in found:
        key = (kind, tuple(value) if isinstance(value, list) else value)
        if key in seen:
            continue
        seen.add(key)
        blocks.append((kind, value))
    return blocks


def payload_text(payload, key):
    """One scalar value out of the payload, e.g. the 「商品特色」 headline."""
    plain = unescape_json(payload)
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', plain)
    return unescape_json(match.group(1)).strip() if match else ""


def meta_content(page_html, key):
    """The content of a `<meta name=...>` or `<meta property=...>` tag."""
    for pattern in (
        r'<meta[^>]*(?:name|property)="%s"[^>]*content="([^"]*)"',
        r'<meta[^>]*content="([^"]*)"[^>]*(?:name|property)="%s"',
    ):
        match = re.search(
            pattern % re.escape(key), page_html, re.IGNORECASE | re.DOTALL
        )
        if match:
            return html.unescape(match.group(1)).strip()
    return ""


def fetch_goods_info(code):
    """The legacy product-info fragment (商品規格 and 商品說明) for one product code.

    An empty string comes back for a code the endpoint has no content for -- a
    third-party listing, say -- because the panel is happy to show the page's own
    blocks alone.
    """
    if not code:
        return ""
    resp = requests.get(
        GOODS_INFO_URL, params={"i_code": code}, headers=HEADERS, timeout=PAGE_TIMEOUT
    )
    resp.raise_for_status()
    return resp.text if len(resp.text) >= GOODS_INFO_MIN_CHARS else ""


def fragment_text(fragment):
    """The readable text of an HTML fragment: tags gone, line breaks kept."""
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        fragment,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|tr|li|table|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def feature_urls(code):
    """Where a product's rich 「商品特色」 page is, in the order to try it.

    The iframe's own shell builds this path itself -- code padded to ten digits, split
    4/3/3, so 14028360 becomes /expertimg/0014/028/360/mobile/14028360.html -- and that
    file is served from the content hosts too, which answer the request the shell's own
    host refuses. A code that is not a plain number is a listing type this does not
    apply to, so it gets nothing.
    """
    if not code.isdigit():
        return []
    padded = code.rjust(10, "0")
    path = f"{padded[0:4]}/{padded[4:7]}/{padded[7:10]}"
    return [
        f"https://{host}{FEATURE_PAGE_SHAPE.format(path=path, code=code)}"
        for host in FEATURE_PAGE_HOSTS
    ]


def feature_images(page_html, base_url):
    """The pictures a rich 「商品特色」 page shows, in page order.

    The supplier's content is usually pictures rather than a paragraph, so they are
    the section: taken as they appear, made absolute, without repeats and capped --
    a strip of them says more than a wall of image tags would.
    """
    found = []
    for source in FEATURE_IMAGE_PATTERN.findall(page_html):
        source = html.unescape(source).strip()
        if not source or source.startswith("data:"):
            continue
        url = urljoin(base_url, source)
        if url not in found:
            found.append(url)
        if len(found) >= FEATURE_IMAGE_LIMIT:
            break
    return found


def fetch_feature(code):
    """Read the 「商品特色」 the product page only embeds: its page, text and pictures.

    One request per host, taking the first that answers with content (see
    `feature_urls` for where and why). A product the site has nothing behind, or a
    host that refuses, reads as empty rather than as an error: the panel keeps the
    headline and the bullets the product page itself carries either way.
    """
    for url in feature_urls(code):
        try:
            reply = requests.get(url, headers=HEADERS, timeout=PAGE_TIMEOUT)
        except requests.RequestException as error:
            print(f"product highlights: {url} failed: {error}", flush=True)
            continue
        if reply.status_code != 200 or len(reply.text) < FEATURE_PAGE_MIN_CHARS:
            print(
                f"product highlights: {url} answered {reply.status_code} with "
                f"{len(reply.text):,} characters",
                flush=True,
            )
            continue
        return {
            "url": url,
            "text": fragment_text(reply.text),
            "images": feature_images(reply.text, url),
        }
    return {"url": "", "text": "", "images": []}


def image_mime(content):
    """The media type of a picture, read off its own first bytes."""
    if content.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"GIF8"):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def image_url(content):
    """A picture as a data URL, which is how it goes out to the model and comes back
    to the browser: the panel has to carry it without a file of its own to serve."""
    return f"data:{image_mime(content)};base64,{base64.b64encode(content).decode()}"


def download_image(url):
    """One picture, fetched by the server so the wording on it can be read.

    The panel shows its pictures straight from Momoshop, but re-lettering one means
    reading its pixels, so the same host rule as the product page applies here (the
    URL came from the page, and a panel is not a licence to fetch any site), plus a
    ceiling on how much is downloaded.
    """
    if "momoshop.com.tw" not in url:
        raise ValueError("not a Momoshop picture")
    reply = requests.get(url, headers=HEADERS, timeout=PAGE_TIMEOUT)
    reply.raise_for_status()
    if len(reply.content) > IMAGE_MAX_BYTES:
        raise ValueError(f"picture is {len(reply.content):,} bytes")
    return reply.content


def picture_reason(call):
    """What a failed call says went wrong, for the lightbox that shows the failure.

    Google's endpoint answers a bad key, a bad model name or a blocked prompt with a
    JSON error object, and a proxy in the way answers with something that is not JSON at
    all, so what is taken from the reply is the message if there is one and the body
    itself if there is not.
    """
    try:
        body = call.json()
    except ValueError:
        return str(call.text)[:200]
    error = body.get("error") if isinstance(body, dict) else None
    message = (error or {}).get("message") if isinstance(error, dict) else None
    return str(message or call.text)[:200]


def picture_bytes(reply):
    """The picture out of a `generateContent` reply, and what the model said with it.

    A reply carries one part per thing the model produced and the picture is the part
    that carries inline data (the REST endpoint names it camelCase, `inlineData`). A
    reply without one is a refusal, a blocked prompt or a text-only answer, and the
    reason for it is turned into the message, because that is what the person looking at
    the panel gets to see instead of a picture.

    The text parts come back too: an image model writes a line about the edit it made
    ("here is the image with the Chinese characters replaced by their translations"),
    which is the only account of its own answer there is, and `picture_report` prints it.
    """
    candidate = (reply.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data") or {}
        if inline.get("data"):
            said = " ".join(str(part.get("text") or "") for part in parts).strip()
            return base64.b64decode(inline["data"]), said
    said = " ".join(str(part.get("text") or "") for part in parts).strip()
    reason = (
        candidate.get("finishReason")
        or (reply.get("promptFeedback") or {}).get("blockReason")
        or (reply.get("error") or {}).get("message")
        or said
        or "no picture in the reply"
    )
    raise RuntimeError(f"{PICTURE_MODEL} returned no picture ({str(reason)[:200]})")


def picture_stem(url):
    """A name for a picture out of its URL, for the files the picture step keeps.

    A Momoshop picture URL ends in the name of the file (`.../14473107_OL.jpg`), which is
    the closest thing one of these pictures has to a name of its own: it is what the kept
    files and the console line are called by, so a picture that came back wrong can be
    found again without a list of products to look it up in. The extension is dropped,
    because the files are named for what they hold (`-source` or `-translated`) and the
    model's answer is not always in the format of the picture that went out.
    """
    name = url.split("?", 1)[0].rsplit("/", 1)[-1]
    name = name.rsplit(".", 1)[0] if "." in name else name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "picture"


def picture_shape(content):
    """A picture's pixel size as "1024x1024" for the console, or "?" if it will not open."""
    try:
        width, height = Image.open(BytesIO(content)).size
    except Exception:  # noqa: BLE001 - the report is worth more than the size is
        return "?"
    return f"{width}x{height}"


def picture_changed(before_content, picture):
    """How much of the picture the model changed, as a percentage of its pixels.

    The two are not the same size -- the model answers at a size of its own, and a
    320x320 picture came back as 1024x1024 -- so the returned one is scaled to the size of
    the one that went out before the two are compared. Only differences past 8 of 255 are
    counted: re-encoding a picture moves every pixel a little without anything on it
    having changed, and a picture that was not edited at all is what this number is for.
    """
    try:
        before = Image.open(BytesIO(before_content)).convert("RGB")
        after = Image.open(BytesIO(picture)).convert("RGB")
        if after.size != before.size:
            after = after.resize(before.size)
        difference = ImageChops.difference(before, after).convert("L")
    except Exception:  # noqa: BLE001 - an unreadable picture is what the report says
        return None
    changed = sum(difference.histogram()[8:])
    return 100 * changed / (before.size[0] * before.size[1])


# The folder `picture_folder` settled on, and whether it has looked yet: the answer does not
# change inside one process, and looking costs a directory and a throwaway file.
_PICTURE_FOLDER = None
_PICTURE_FOLDER_LOOKED = False


def picture_folder():
    """A folder the picture step's two files can be written to, or `None` if there is none.

    `PICTURE_DIR` beside `main.py` is where this was written and what the README promises,
    and on a machine that runs the app from a checkout it is the answer. A hosted runner
    does not run it from a checkout: Vercel serves the files from a read-only `/var/task`,
    where `mkdir` fails with `[Errno 30] Read-only file system`, and `/tmp` is the one place
    a function may write. So the candidates are the configured path (alone, when it is
    absolute, because that is an instruction rather than a name), then the same name beside
    the app, then the same name under `tempfile.gettempdir()` -- and each one has to accept
    a real file before it is trusted, because a mount can report itself writable and refuse
    the write, and a `mkdir(exist_ok=True)` on an existing read-only folder succeeds.

    `None` is not a failure of the picture step: the console block is the account of what
    the model answered, and the two files are only how that answer can be looked at again.
    See `picture_report`.
    """
    global _PICTURE_FOLDER, _PICTURE_FOLDER_LOOKED
    if _PICTURE_FOLDER_LOOKED:
        return _PICTURE_FOLDER
    _PICTURE_FOLDER_LOOKED = True
    wanted = Path(PICTURE_DIR)
    candidates = (
        [wanted]
        if wanted.is_absolute()
        else [
            Path(__file__).resolve().parent / wanted,
            Path(tempfile.gettempdir()) / wanted,
        ]
    )
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".mm-", delete=True):
                pass
        except OSError:
            continue
        _PICTURE_FOLDER = folder
        break
    return _PICTURE_FOLDER


def picture_report(stem, language, source, picture, said, seconds):
    """The picture that came back kept on disk, and one block of console about it.

    Everything this step can get wrong arrives as a picture: a model that answered with
    the picture it was given, or with the same picture at a larger size and the Chinese
    still on it, is a 200 with an image in it, and the lightbox cannot tell that from a
    translation. The panel is left to show the picture; this leaves the two pictures side
    by side in `picture_folder()` and the account of the answer in the console, so "it came
    back untranslated" can be looked at and measured rather than argued about -- the
    percentage below a few per cent being an answer that is not an edit of the wording.

    Nothing here may fail the step, which is why the writing is wrapped rather than
    assumed: a host that writes nowhere, or a disk that is full, costs the two files and
    leaves the console block saying so, while the picture itself still goes to the panel.
    """
    folder = picture_folder()
    slug = re.sub(r"[^a-z0-9]+", "-", language.lower()).strip("-") or "target"
    kept, lost = [], "this host has nowhere to write"
    for kind, content in (("source", source), ("translated", picture)):
        if folder is None:
            break
        path = folder / f"{stem}-{slug}-{kind}{PICTURE_SUFFIX[image_mime(content)]}"
        try:
            path.write_bytes(content)
        except OSError as error:
            lost = str(error)
            break
        kept.append(path)
    changed = picture_changed(source, picture)
    changed = (
        "the pictures could not be compared"
        if changed is None
        else f"{changed:.1f}% of the picture differs from the one that went out"
    )
    if len(kept) == 2:
        where = f"kept: {kept[0].name} and {kept[1].name} in {folder}"
    elif kept:
        where = f"kept: only {kept[0].name}, not the other one -- {lost}"
    else:
        where = f"kept: nothing -- {lost}"
    print(
        f"picture: {PICTURE_MODEL} answered in {seconds:.1f}s with {len(picture):,} bytes "
        f"of {image_mime(picture)} at {picture_shape(picture)}, for {len(source):,} bytes "
        f"at {picture_shape(source)} of {stem}\n"
        f"  {changed}\n"
        f"  model said: {said or '(no text came back with the picture)'}\n"
        f"  {where}",
        flush=True,
    )


def picture_model(content, language, stem="picture"):
    """`content`, with the Chinese wording on it translated into `language`.

    One request does the whole job: the picture goes out as inline base64 beside the
    prompt, and the picture with its wording changed comes back in the same shape. That
    is what this step wanted all along -- nothing has to be read off the picture,
    translated, and drawn back on -- and the layout is the model's to keep because
    keeping it is what the prompt asks for.

    The reply is asked for `TEXT` as well as `IMAGE` because an image model answers with
    a picture only when the modalities say so: asked for text alone it would describe the
    edit instead of making it.

    `stem` names the picture in the two files `picture_report` keeps and in the console
    block it prints; it is the picture's own file name, which is what the caller has.
    """
    if not PICTURE_KEY:
        raise RuntimeError("GEMINI_KEY is not set, so there is no picture model to call")
    started = time.time()
    call = requests.post(
        f"{PICTURE_BASE_URL}/models/{PICTURE_MODEL}:generateContent",
        headers={"x-goog-api-key": PICTURE_KEY},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": IMAGE_TRANSLATE_PROMPT.format(language=language)},
                        {
                            "inline_data": {
                                "mime_type": image_mime(content),
                                "data": base64.b64encode(content).decode(),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        },
        timeout=IMAGE_TIMEOUT,
    )
    if call.status_code != 200:
        raise RuntimeError(
            f"{PICTURE_MODEL} answered {call.status_code}: {picture_reason(call)}"
        )
    picture, said = picture_bytes(call.json())
    picture_report(stem, language, content, picture, said, time.time() - started)
    return picture


def parse_wording(reply):
    """The lines and boxes out of the model's reply, as fractions of the picture.

    The reply is JSON as asked for, but it arrives inside a ```json fence as often as
    not and a box can come back as a string or be missing, so what is taken from it is
    only what is usable: a line of text with four numbers that are not obviously
    wrong.
    """
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    lines = []
    for entry in data.get("lines") or []:
        if not isinstance(entry, dict):
            continue
        # One line of wording is one line here: the model sometimes answers with a
        # break inside the text of a single box, and that would turn one numbered line
        # into two when the lines are sent to be translated.
        text = " ".join(str(entry.get("text") or "").split())
        box = []
        for name in ("x", "y", "w", "h"):
            try:
                box.append(float(entry.get(name)))
            except (TypeError, ValueError):
                box.append(0.0)
        if text and 0 <= box[0] < 1 and 0 <= box[1] < 1 and box[2] > 0 and box[3] > 0:
            lines.append({"text": text, **dict(zip("xywh", box))})
    return lines


@lru_cache(maxsize=64)
def picture_font(size):
    """A font at `size`: the system's own where there is one, Pillow's built-in else."""
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def wrapped_lines(draw, text, font, width):
    """`text` cut into lines that fit `width`, a word too wide to fit broken up.

    Words are kept whole where they fit, and a word wider than the line -- a model
    number, a URL -- is cut at the character that fits rather than running off the
    picture.
    """
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while len(word) > 1 and draw.textlength(word, font=font) > width:
            cut = len(word)
            while cut > 1 and draw.textlength(word[:cut], font=font) > width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines or [""]


def line_box(line, width, height):
    """A line's box in pixels, or None when the model's numbers are not usable.

    A wrong box is dropped rather than clamped: what is drawn over an original goes on
    top of it, so a line the model placed taller than a third of the picture, or as a
    sliver too small to letter, or as tall as it is narrow (see
    `IMAGE_MAX_BOX_ASPECT`), is one this leaves alone -- the original stays.
    """
    if not (0 <= line["x"] < 1 and 0 <= line["y"] < 1):
        return None
    if line["w"] > IMAGE_MAX_BOX_WIDTH or line["h"] > IMAGE_MAX_BOX_HEIGHT:
        return None
    if line["h"] > IMAGE_MAX_BOX_ASPECT * line["w"]:
        return None
    box = (
        round(line["x"] * width),
        round(line["y"] * height),
        min(width, round((line["x"] + line["w"]) * width)),
        min(height, round((line["y"] + line["h"]) * height)),
    )
    if box[2] - box[0] < IMAGE_MIN_BOX[0] or box[3] - box[1] < IMAGE_MIN_BOX[1]:
        return None
    return box


def grown_box(box, need_width, need_height, width, height, band=None):
    """`box` grown to hold a `need_width` by `need_height` block, staying on the picture.

    The growth is around the original box's own centre, so a translated line sits
    where the wording it replaces sat, and it is pulled back inside the picture rather
    than clipped at its edge. `band` is the vertical room this block has: the wording
    above and below it belongs to another block, so the growth stops there instead of
    painting a second block's translation over the first.
    """
    need_width, need_height = int(need_width), int(need_height)
    top, bottom = band if band else (0, height)
    x0, y0, x1, y1 = (int(value) for value in box)
    grow_x = max(0, need_width - (x1 - x0))
    grow_y = max(0, need_height - (y1 - y0))
    x0, x1 = x0 - grow_x // 2, x1 + grow_x - grow_x // 2
    y0, y1 = y0 - grow_y // 2, y1 + grow_y - grow_y // 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < top:
        y1 += top - y0
        y0 = top
    if x1 > width:
        x0 = max(0, x0 - (x1 - width))
        x1 = width
    if y1 > bottom:
        y0 = max(top, y0 - (y1 - bottom))
        y1 = bottom
    return (x0, y0, x1, y1)


def ring_colour(image, box):
    """The colour around one box, which is what that box is painted with.

    The middle value per channel rather than the average, so a few pixels of the
    wording that the box still holds do not drag the colour towards the type.
    """
    x0, y0, x1, y1 = box
    step_x = max(1, (x1 - x0) // 24)
    step_y = max(1, (y1 - y0) // 24)
    ring = []
    for x in range(x0, x1, step_x):
        ring.append(image.getpixel((x, y0)))
        ring.append(image.getpixel((x, y1 - 1)))
    for y in range(y0, y1, step_y):
        ring.append(image.getpixel((x0, y)))
        ring.append(image.getpixel((x1 - 1, y)))
    ring = [pixel[:3] for pixel in ring]
    middle = len(ring) // 2
    return tuple(sorted(colour[i] for colour in ring)[middle] for i in range(3))


def touching(box, other):
    """True when two boxes are part of the same block of wording.

    Lines of one paragraph sit in the same columns within a line's height of each
    other; two lines that far apart but side by side are two blocks, and are lettered
    separately so that neither is painted over the other's picture.
    """
    gap = max(6, round(1.2 * min(box[3] - box[1], other[3] - other[1])))
    return (
        box[0] < other[2]
        and other[0] < box[2]
        and max(other[1] - box[3], box[1] - other[3]) < gap
    )


def cluster_boxes(entries):
    """The boxes grouped into the blocks of wording they belong to, in reading order.

    One label per line is what leaves the Chinese that a box did not quite cover
    showing at its edges; one label per block covers the block and letters the whole
    translation into it. Boxes are merged until nothing overlaps anything, and the
    lines inside a block are kept in the order they read.
    """
    clusters = [
        {"box": box, "lines": [(box, text)]}
        for box, text in sorted(entries, key=lambda entry: (entry[0][1], entry[0][0]))
    ]
    for index in range(len(clusters)):
        for other in range(index + 1, len(clusters)):
            if not clusters[index]["lines"] or not clusters[other]["lines"]:
                continue
            if touching(clusters[index]["box"], clusters[other]["box"]):
                first, second = clusters[index]["box"], clusters[other]["box"]
                clusters[index]["box"] = (
                    min(first[0], second[0]),
                    min(first[1], second[1]),
                    max(first[2], second[2]),
                    max(first[3], second[3]),
                )
                clusters[index]["lines"] += clusters[other]["lines"]
                clusters[other]["lines"] = []
    return [cluster for cluster in clusters if cluster["lines"]]


def image_font_floor(height):
    """The smallest type this picture may be lettered in.

    A fixed floor would be too small to read on a large picture, so it is a fraction of
    the picture's own height: on a 664 pixel tall picture it comes out at 15 pixels,
    which is what "readable on the screen" means for a picture that size.
    """
    return max(IMAGE_MIN_FONT, round(height * IMAGE_MIN_FONT_RATIO))


def padded(box, width, height):
    """`box` with a margin around it, so the edges of the lines it covers go too.

    The model places its boxes by eye, so a margin is what stops the top of a tall
    character, or a dot beside a line, from being left behind on the picture when the
    block is painted over.
    """
    pad_x = max(3, round(0.015 * width))
    pad_y = max(2, round(0.08 * (box[3] - box[1])))
    return (
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(width, box[2] + pad_x),
        min(height, box[3] + pad_y),
    )


def draw_cluster(draw, image, box, lines, band):
    """Draw one block of translations where the wording it replaces was.

    The type starts at the height the original lines had and comes down until every
    translated line has room, with the box growing around where that wording sat and
    stopping at `band`, which is the room the neighbouring blocks are not using: a
    translation is longer than the Chinese it replaces, and a block of marketing copy
    that had to fit in the space of the Chinese would otherwise be lettered at a size
    nobody can read. None comes back when there is no room legibly, in which case
    nothing was drawn and the original wording is left alone.
    """
    width, height = image.size
    box = padded(box, width, height)
    x0, y0, x1, y1 = box
    texts = [text for _, text in sorted(lines, key=lambda line: (line[0][1], line[0][0]))]
    floor = image_font_floor(height)
    start = max(floor, round((y1 - y0) / max(1, len(texts)) * 0.9))
    for size in range(start, floor - 1, -1):
        font = picture_font(size)
        wrapped = []
        for text in texts:
            wrapped.extend(wrapped_lines(draw, text, font, x1 - x0 - 8))
        need_width = round(max(draw.textlength(line, font=font) for line in wrapped)) + 12
        need_height = len(wrapped) * round(size * 1.2) + 8
        place = grown_box(
            box,
            min(need_width, width),
            min(need_height, band[1] - band[0]),
            width,
            height,
            band,
        )
        if place[2] - place[0] < need_width or place[3] - place[1] < need_height:
            continue
        colour = ring_colour(image, place)
        draw.rounded_rectangle(place, radius=5, fill=colour)
        draw.multiline_text(
            ((place[0] + place[2]) / 2, (place[1] + place[3]) / 2),
            "\n".join(wrapped),
            font=font,
            fill=(250, 250, 250) if sum(colour) < 380 else (18, 18, 18),
            anchor="mm",
            align="center",
            spacing=3,
        )
        return place
    return None


def reletter_image(content, lines):
    """The same picture with `lines` lettered where the model said the wording was.

    Gemma read the wording and where each line of it sits; this puts the translation
    in that wording's place, painted over with the colour around it, so the picture is
    the same picture and the wording on it is the target language. A line whose box
    was not usable (see `line_box`) is left as it is: what is drawn here goes on top of
    the original, so a guess about where the wording is would be worse than a line left
    in Chinese. How many lines were lettered and how many were left comes back with the
    picture, and the panel says which.
    """
    image = Image.open(BytesIO(content)).convert("RGB")
    if image.width > IMAGE_MAX_WIDTH:
        image = image.resize(
            (IMAGE_MAX_WIDTH, round(image.height * IMAGE_MAX_WIDTH / image.width)),
            Image.LANCZOS,
        )
    draw = ImageDraw.Draw(image)
    width, height = image.size

    entries = []
    for line in lines:
        box = line_box(line, width, height)
        if box is not None:
            entries.append((box, line["text"]))

    clusters = cluster_boxes(entries)
    lettered = 0
    for index, cluster in enumerate(clusters):
        # The room a block has is the space between it and its neighbours: the growth
        # one block needs is what painted over the next one before this was here.
        top = (
            0
            if index == 0
            else (clusters[index - 1]["box"][3] + cluster["box"][1]) // 2
        )
        bottom = (
            height
            if index == len(clusters) - 1
            else (cluster["box"][3] + clusters[index + 1]["box"][1]) // 2
        )
        if draw_cluster(draw, image, cluster["box"], cluster["lines"], (top, bottom)) is not None:
            lettered += len(cluster["lines"])
    out = BytesIO()
    image.save(out, "JPEG", quality=IMAGE_QUALITY, optimize=True)
    return out.getvalue(), lettered, len(lines) - lettered


def image_translated(url, language):
    """The same picture, with the wording that is on it in `language`.

    `PICTURE_MODEL` does the whole job in one request (see `picture_model`), and what
    comes back is that picture: nothing was read off it line by line here, so `read` is
    empty and the panel says the picture was returned rather than listing lines. Both
    pictures -- the one that went out and the one that came back -- are kept in
    `picture_folder()`, and a block of console says what the model returned, what it said
    about its own answer, how long the call took and how much of the picture it changed: the
    wording it was given is `IMAGE_TRANSLATE_PROMPT`, and the answer beside it in the
    console is what that wording produced. See `picture_report`.

    Without a key for that model, `IMAGE_MODEL` on the chat endpoint is what is left --
    Gemma, on the default endpoint -- and that is the smaller half of the job: it reads
    the wording off the picture and where each line of it is, the same model translates
    those lines, and `reletter_image` letters them back into the same picture. It has to
    be a model that accepts a picture, which is not something every chat endpoint
    serves. What comes back either way is everything the panel
    needs to say what happened -- the lines read, their translation, and the picture --
    whether or not anything could be drawn on it, and `model` says which of the two ran.
    """
    content = download_image(url)
    if PICTURE_KEY:
        kept = picture_folder()
        where = (
            f"is kept in {kept} and reported below"
            if kept is not None
            else "is reported below, this host having nowhere to keep it"
        )
        print(
            f"picture: sending {url} to {PICTURE_MODEL} for {language} -- what comes back "
            f"{where}",
            flush=True,
        )
        return {
            "url": url,
            "read": [],
            "wording": [],
            "picture": image_url(
                picture_model(content, language, stem=picture_stem(url))
            ),
            "drawn": 0,
            "skipped": 0,
            "model": PICTURE_MODEL,
        }
    print(
        f"GEMINI_KEY is not set, so the wording is read and re-lettered with {IMAGE_MODEL}",
        flush=True,
    )
    reply = complete_again(
        [
            {"type": "text", "text": IMAGE_READ_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url(content)}},
        ],
        model=IMAGE_MODEL,
        timeout=IMAGE_TIMEOUT,
    )
    if not reply:
        # A gateway answering with an error page reaches here as an empty reply, and
        # "read no wording on this picture" would be the wrong thing to say about it:
        # the picture was never read. The lightbox says this instead.
        raise RuntimeError(f"{IMAGE_MODEL} did not answer for this picture")
    lines = parse_wording(reply)
    read = [line["text"] for line in lines]
    translated = translate_lines(
        [(str(number), line["text"]) for number, line in enumerate(lines)],
        language,
        model=IMAGE_MODEL,
    )
    for number, line in enumerate(lines):
        line["text"] = translated.get(str(number)) or line["text"]

    picture, drawn, skipped = reletter_image(content, lines) if lines else (b"", 0, 0)
    return {
        "url": url,
        "read": read,
        "wording": [line["text"] for line in lines],
        "picture": image_url(picture) if drawn else "",
        "drawn": drawn,
        "skipped": skipped,
        "model": IMAGE_MODEL,
    }


def product_detail(url):
    """One product page, with the content its own payload carries, parsed out.

    Everything here is server-side: the page is fetched whole, its React payload
    is decoded into blocks, its `<meta>` tags give the name, price, brand and
    image, and the legacy product-info endpoint -- the one host that answers from
    here -- fills the specs and the description in for the pages that came back as
    a shell without their payload. This is the same data the browser renders from,
    so nothing has to run the page to read it. The 「商品特色」 iframe is not in the
    page at all, so `fetch_feature` goes after it separately: that section's rich
    content is its own page, and the product code says where it is.
    """
    page = fetch_product_page(url)
    payload = rsc_payload(page)
    code_match = PRODUCT_CODE_PATTERN.search(url)

    detail = {
        "url": url,
        "page": page,
        "payload_chars": len(payload),
        "blocks": payload_blocks(payload),
        "feature": {
            "headline": payload_text(payload, "goodsFeatureString"),
            "iframe": payload_text(payload, "goodsFeatureUrl"),
            "url": "",
            "text": "",
            "images": [],
        },
        "meta": {
            key: meta_content(page, key)
            for key in (
                "og:title",
                "description",
                "product:brand",
                "product:price:amount",
                "product:price:currency",
                "product:availability",
                "product:retailer_item_id",
            )
        },
        "image": meta_content(page, "og:image").split("?", 1)[0],
        "code": code_match.group(1) if code_match else "",
        "info_text": "",
    }
    try:
        detail["feature"].update(fetch_feature(detail["code"]))
    except Exception as error:  # noqa: BLE001 - the panel works without the embed
        print(f"product highlights failed: {error}", flush=True)
    try:
        detail["info_text"] = fragment_text(fetch_goods_info(detail["code"]))
    except Exception as error:  # noqa: BLE001 - the panel works without it
        print(f"goods info failed: {error}", flush=True)
    return detail


def extract_items(page_html):
    """Pull the product dicts out of the JSON-LD ItemList of a search page."""
    items = []
    for block in JSON_LD_SCRIPT.findall(page_html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for node in data.get("@graph", []):
            if isinstance(node, dict) and node.get("@type") == "ItemList":
                items.extend(
                    element
                    for element in node.get("itemListElement", [])
                    if isinstance(element, dict)
                )
    return items


def item_texts(items):
    """The (title, note) text we show for each product."""
    return [
        (str(item.get("name") or "(no name)"), str(item.get("description") or ""))
        for item in items
    ]


def parse_numbered_lines(text):
    """Map the "1. some text" lines of a chat reply to {number: "some text"}."""
    lines = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*[.)、]?\s*(.*)", line)
        if match and match.group(2).strip():
            lines[int(match.group(1))] = match.group(2).strip()
    return lines


def translate_texts(items, language):
    """Translate the title and note of a handful of products in one request."""
    originals = item_texts(items)
    if not originals:
        return []

    listing = "\n".join(
        f"{number}. {title} || {note}"
        for number, (title, note) in enumerate(originals, 1)
    )
    prompt = (
        f"Translate each numbered Momoshop product listing below from "
        f"Traditional Chinese into {language}.\n"
        "Before the '||' is the product title, after it is a short note or "
        "promotion that may be empty.\n"
        "Translate every part of the text, brand and product names included. "
        "A Chinese brand or product name goes in the way it is normally "
        f"written in {language} (for example 義美 is I-Mei and 盛香珍 is Sheng "
        "Xiang Zhen), and no Chinese characters may be left in the reply -- "
        "not even in a name written half in Latin letters, where the Chinese "
        "half is spelled out just the same (CHU SHIN WU ER 無二 is CHU SHIN "
        "WU ER Wu Er).\n"
        "Keep the numbering, keep the '||' separator, and keep model numbers, "
        "sizes, units and digits as they are. Do not skip, merge or add lines.\n"
        "Reply with only the translated numbered lines.\n\n"
        f"{listing}"
    )

    translated = parse_numbered_lines(complete(prompt, RESULT_MODEL))

    rows = []
    for number, (title, note) in enumerate(originals, 1):
        line = translated.get(number, "")
        if line:
            title_part, _, note_part = line.partition("||")
            title = title_part.strip() or title
            note = note_part.strip() or note
        rows.append((title, note))
    return rows


def has_chinese(text):
    """True when any Chinese character is left in the text."""
    return CJK_PATTERN.search(text) is not None


def looks_untranslated(translated):
    """True when Chinese is still left in the reply for one product.

    A request that failed comes back as the original Chinese, and the model
    sometimes translates only part of a line, leaving a brand or a bracketed
    flavour in Chinese. Both are caught here and asked for again.
    """
    title, note = translated
    return has_chinese(title) or has_chinese(note)


def translate_plan(total):
    """How many of `total` products this run actually asks the endpoint about.

    `TRANSLATE_LIMIT` is a testing cap, so the whole page can be walked through
    quickly; the products past it keep the text they were scraped with, which is
    what the page shows for them.
    """
    return min(total, TRANSLATE_LIMIT) if TRANSLATE_LIMIT > 0 else total


def translate_lines(pairs, language, model=DETAIL_MODEL):
    """Translate short (key, line) pairs in one request, keyed the same way back.

    The lines go out numbered and come back numbered, the way the result cards do,
    and a line the reply left out keeps the text it came with -- the panel then shows
    that line as it was scraped rather than dropping it. `model` is what translates
    them: the panel uses the detail model, the wording read off a picture is
    translated by the model that read it.
    """
    listing = "\n".join(
        f"{number}. {text}" for number, (_, text) in enumerate(pairs, 1)
    )
    prompt = (
        f"Translate each numbered line below from Traditional Chinese into {language}. "
        "They are lines read off a Momoshop product page: a heading, a bullet, a "
        "specification or a short description.\n"
        "Translate every part of the text, brand and product names included, and "
        "leave no Chinese characters in the reply: a Chinese brand or product name "
        f"goes in the way it is normally written in {language} (for example 義美 is "
        "I-Mei and 盛香珍 is Sheng Xiang Zhen).\n"
        "Keep the numbering, keep model numbers, sizes, units and digits as they are, "
        "and do not skip, merge or add lines.\n"
        "Reply with only the translated numbered lines.\n\n"
        f"{listing}"
    )
    translated = parse_numbered_lines(complete_again(prompt, model))
    return {
        key: (translated.get(number) or text)
        for number, (key, text) in enumerate(pairs, 1)
    }


def translate_paragraph(text, language):
    """Translate one long block in a request of its own, line breaks kept.

    A long block is a description or a specification table, whose line breaks are
    part of how it reads, so it cannot go out as one numbered line. It keeps its
    original text when the request does not answer.
    """
    prompt = (
        "Translate the following Traditional Chinese text from a Momoshop product "
        f"page into {language}.\n"
        "Translate every part of it, brand and product names included, and leave no "
        f"Chinese characters: a Chinese brand or product name goes in the way it is "
        f"normally written in {language} (for example 義美 is I-Mei).\n"
        "Keep the line breaks and the shape of the original, keep model numbers, "
        "sizes, units and digits as they are, and reply with only the translation."
        f"\n\n{text}"
    )
    return complete_again(prompt, DETAIL_MODEL).strip() or text


def detail_texts(detail):
    """The panel's own strings, as (key, text) in the order the panel shows them.

    The key is what a translation is filed under, so the panel can put a translated
    line where it would have shown the scraped one. It is built from the page's
    shape rather than from a counter, so the same key comes back for the same piece
    of text however many times the panel is rendered: `<meta>` first, then 「商品特色」
    with the page's own bullets, then the content blocks, and the legacy text last --
    which is the order `detail_sections_html` lays them out in.
    """
    metas = detail["meta"]
    pairs = []
    for name in ("og:title", "description", "product:brand"):
        if metas.get(name):
            pairs.append((f"meta:{name}", metas[name]))

    feature = detail["feature"]
    if feature["headline"]:
        pairs.append(("feature:headline", feature["headline"]))
    if feature["text"]:
        pairs.append(("feature:text", feature["text"][:DETAIL_INFO_MAX_CHARS]))

    for index, (kind, value) in enumerate(detail["blocks"]):
        if kind == "format":
            pairs.extend(
                (f"block:{index}:{position}", bullet)
                for position, bullet in enumerate(value)
            )
        else:
            pairs.append((f"block:{index}", value))

    if detail["info_text"] and not any(
        kind == "text" for kind, _ in detail["blocks"]
    ):
        pairs.append(("legacy:text", detail["info_text"][:DETAIL_INFO_MAX_CHARS]))
    return pairs


def detail_chunks(pairs):
    """The panel's lines as requests: short ones numbered and chunked, long ones alone.

    How a click waits between two parts of the panel filling in is decided here:
    `DETAIL_CHUNK_CHARS` of short lines per request, and at most `DETAIL_REQUEST_CAP`
    long blocks -- each a request of its own, because a block with line breaks in it
    cannot go out as one numbered line. A page with more long blocks than that is not
    silently left half translated: the panel counts those lines as still in Chinese and
    its button asks for them.
    """
    short, long = [], []
    for key, text in pairs:
        if len(text) > DETAIL_LONG_CHARS or "\n" in text:
            long.append((key, text))
        else:
            short.append((key, text))

    chunks = []
    current, size = [], 0
    for key, text in short:
        if current and size + len(text) > DETAIL_CHUNK_CHARS:
            chunks.append(("lines", current))
            current, size = [], 0
        current.append((key, text))
        size += len(text)
    if current:
        chunks.append(("lines", current))
    chunks.extend(("block", [(key, text)]) for key, text in long[:DETAIL_REQUEST_CAP])
    return chunks


def translate_detail(detail, language, only=None):
    """Translate the panel's text, yielding the translations as each chunk lands.

    Each frame is the whole map rather than the new keys, because each frame is a
    whole panel: the caller renders it with whatever is in the map, and the lines
    that are not in it yet keep the text they were scraped with. That is what makes
    the panel fill in part by part instead of all at once or not at all.

    `only` restricts the work to those keys, which is how the panel asks again for the
    lines that came back in Chinese without asking again for the ones that arrived.

    `language` arrives as the language the radio is set to ("Indonesia"), and every
    prompt below needs the language's own name ("Indonesian") -- a model asked for
    "Indonesia" answers in English or leaves the Chinese where it was. The cards and
    the picture step already pass the name (see `RESULT_LANGUAGE`), so it is taken
    here as either, and a language this app does not offer still translates nothing.
    """
    language = RESULT_LANGUAGE.get(language, language)
    if language not in RESULT_LANGUAGE.values():
        return
    pairs = [
        (key, text)
        for key, text in detail_texts(detail)
        if only is None or key in only
    ]
    if not pairs:
        return

    translations = {}
    for mode, chunk in detail_chunks(pairs):
        if mode == "lines":
            translations.update(translate_lines(chunk, language))
        else:
            translations.update(
                {key: translate_paragraph(text, language) for key, text in chunk}
            )
        yield translations


def detail_leftovers(detail, translations):
    """The panel's lines that are still Chinese: never asked for, or not answered.

    The panel offers to ask for these again, so a request that timed out on the
    endpoint leaves a click's worth of work rather than a page half in Chinese.
    """
    return [
        (key, text)
        for key, text in detail_texts(detail)
        if has_chinese(translations.get(key, text))
    ]


def translate_items(items, language):
    """Translate every product, yielding the whole page as each answer lands.

    Products are asked for one request at a time (`TRANSLATE_BATCH` of them per
    request), so a stalled or half translated reply can only cost those cards, and
    every retry wave re-asks exactly the products that came back with Chinese
    still in them -- never the ones that already arrived. A wave waits before it goes
    out (`TRANSLATE_RETRY_WAIT`), because the requests it is re-asking for are the ones
    the endpoint may still be generating. Frames are yielded in the
    order the answers land, which is what makes the cards appear one by one.
    """
    rows = list(item_texts(items))
    total = len(items)
    pending = list(range(translate_plan(total)))
    if len(pending) < total:
        print(
            f"translate: {len(pending)} of {total} product(s) only "
            f"(TRANSLATE_LIMIT={TRANSLATE_LIMIT}); the rest stay as scraped",
            flush=True,
        )

    for attempt in range(1, TRANSLATE_ATTEMPTS + 1):
        if not pending:
            return
        if attempt > 1:
            wait = (
                TRANSLATE_RATE_LIMIT_WAIT
                if RATE_LIMITED.is_set()
                else TRANSLATE_RETRY_WAIT * (attempt - 1)
            )
            print(
                f"retry {attempt - 1}: {len(pending)} product(s) are still in Chinese; "
                f"waiting {wait:.0f}s so the endpoint can finish what it was already asked",
                flush=True,
            )
            time.sleep(wait)
        # After a 429 the token is refusing concurrency, so the retries go out one
        # request at a time instead of walking into the same wall again.
        workers = 1 if RATE_LIMITED.is_set() else TRANSLATE_WORKERS
        groups = [
            pending[start : start + TRANSLATE_BATCH]
            for start in range(0, len(pending), TRANSLATE_BATCH)
        ]
        pool = ThreadPoolExecutor(max_workers=min(len(groups), workers))
        try:
            futures = {
                pool.submit(translate_texts, [items[i] for i in group], language): group
                for group in groups
            }
            for future in as_completed(futures):
                try:
                    translated = future.result()
                except Exception as error:  # noqa: BLE001 - one bad reply must not break the page
                    print(f"translation request failed: {error}", flush=True)
                    continue
                for index, row in zip(futures[future], translated):
                    rows[index] = row
                yield rows
        finally:
            # Requests stuck on the endpoint must not keep us waiting for them.
            pool.shutdown(wait=False, cancel_futures=True)
        pending = [index for index in pending if looks_untranslated(rows[index])]

    if pending:
        print(
            f"{len(pending)} product(s) are still in Chinese -- try the search "
            "again in a moment",
            flush=True,
        )


# Cards use inline CSS so they render the same whatever the surrounding theme
# does, while still following the active Gradio theme through its CSS variables.
CARD_STYLE = (
    "display:flex;gap:14px;align-items:flex-start;padding:10px;margin:0;"
    "border:1px solid var(--border-color-primary,#e5e7eb);border-radius:10px;"
    "background:var(--background-fill-secondary,#fff);color:inherit;"
    "cursor:pointer;"
)
THUMB_STYLE = (
    "width:110px;height:110px;flex:0 0 110px;object-fit:contain;"
    "border-radius:8px;background:var(--background-fill-primary,#f3f4f6);"
)
NO_IMAGE_STYLE = (
    THUMB_STYLE
    + "display:flex;align-items:center;justify-content:center;font-size:12px;"
    "opacity:0.5;"
)
TEXT_STYLE = "display:flex;flex-direction:column;gap:6px;min-width:0;"
TITLE_STYLE = "font-size:15px;font-weight:600;line-height:1.35;overflow-wrap:anywhere;"
META_STYLE = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px;"
PRICE_STYLE = "font-weight:700;color:var(--color-accent,#c2410c);"
RATING_STYLE = "opacity:0.75;"
NOTE_STYLE = "font-size:12px;opacity:0.7;overflow-wrap:anywhere;"
HEAD_STYLE = "font-size:15px;font-weight:600;opacity:0.85;margin-bottom:10px;"
LIST_STYLE = "display:flex;flex-direction:column;gap:10px;"
MESSAGE_STYLE = "opacity:0.75;"

# The detail panel shows one product page: the content its own payload carries,
# read out section by section, with the page source behind a fold, and with each line
# replaced by its translation as that answer arrives while the Chinese it replaced
# stays one fold away. Every value is escaped, so nothing of the page reaches the app
# as anything but characters. The source is big, so it scrolls instead of wrapping: a
# wrapped megabyte of HTML is slow to lay out in a browser.
DETAIL_STYLE = (
    "border:1px solid var(--border-color-primary,#e5e7eb);border-radius:10px;"
    "padding:12px;background:var(--background-fill-secondary,#fff);"
)
DETAIL_HEAD_STYLE = (
    "font-size:15px;font-weight:600;line-height:1.35;margin-bottom:6px;"
    "overflow-wrap:anywhere;"
)
DETAIL_URL_STYLE = (
    "font-size:12px;opacity:0.7;margin-bottom:8px;overflow-wrap:anywhere;"
)
DETAIL_SOURCE_STYLE = (
    "margin:0;padding:10px;max-height:420px;overflow:auto;font-size:12px;"
    "line-height:1.45;white-space:pre;font-family:ui-monospace,SFMono-Regular,"
    "Menlo,Consolas,monospace;background:var(--background-fill-primary,#f3f4f6);"
    "border-radius:8px;color:inherit;"
)
# The panel reads the page's own blocks out of the payload it carries, so what it
# shows is the same text the browser renders, section by section.
DETAIL_ROW_STYLE = "display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;"
DETAIL_SECTION_STYLE = "font-size:14px;font-weight:600;margin:14px 0 4px;"
DETAIL_LEAD_STYLE = "font-size:13px;font-weight:600;margin-bottom:4px;"
DETAIL_BODY_STYLE = (
    "font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;"
)
DETAIL_LIST_STYLE = "margin:0;padding-left:20px;font-size:13px;line-height:1.6;"
DETAIL_FACTS_STYLE = (
    "display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:13px;"
)
DETAIL_NOTE_STYLE = "font-size:12px;opacity:0.7;margin-top:6px;overflow-wrap:anywhere;"
DETAIL_RAW_STYLE = "margin-top:14px;font-size:12px;"
DETAIL_EMPTY_STYLE = "font-size:13px;opacity:0.7;"
# While the panel is translating, the page is on the screen as the page wrote it and one
# layer over it says so: the panel is washed out to half in its own background colour,
# and the label rides above that wash so it keeps full strength -- and stays at the top of
# the window however far the page is scrolled under it. The layer takes no clicks, so the
# links and pictures in the page underneath still work while it is up.
DETAIL_VEIL_WRAP_STYLE = "position:relative;"
DETAIL_VEIL_STYLE = (
    "position:absolute;inset:0;border-radius:10px;pointer-events:none;z-index:5;"
    "background:var(--background-fill-secondary,#fff);opacity:0.5;"
)
DETAIL_VEIL_LABEL_STYLE = (
    "position:absolute;inset:0;pointer-events:none;z-index:6;display:flex;"
    "justify-content:center;align-items:flex-start;padding-top:14px;"
)
DETAIL_VEIL_PILL_STYLE = (
    "position:sticky;top:10px;font-size:12px;font-weight:600;line-height:1.4;"
    "padding:5px 12px;border-radius:999px;white-space:nowrap;color:#fff;"
    "background:rgba(17,24,39,0.85);box-shadow:0 3px 12px rgba(0,0,0,0.28);"
)
DETAIL_FEATURE_IMAGE_STYLE = (
    "max-width:min(100%,340px);border-radius:8px;cursor:zoom-in;"
    "background:var(--background-fill-primary,#f3f4f6);"
)
DETAIL_THUMB_STYLE = THUMB_STYLE + "cursor:zoom-in;"
DETAIL_FEATURE_ROW_STYLE = (
    "display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;margin-top:6px;"
)
# The links and buttons in the panel are labels rather than addresses (see
# `link_html`), so they are bubbles like the ones Gradio draws for its own controls.
BUTTON_STYLE = (
    "font:inherit;font-size:12px;line-height:1.4;padding:3px 10px;border-radius:999px;"
    "border:1px solid var(--border-color-primary,#d1d5db);cursor:pointer;"
    "background:var(--background-fill-secondary,#fff);color:inherit;"
)
# A picture opens in an overlay over the whole window: the panel behind keeps its
# place in the results, and the picture is one the browser fetches itself, exactly as
# it does in the strip.
LIGHTBOX_STYLE = (
    "position:fixed;inset:0;z-index:1000;background:rgba(15,15,15,0.72);"
    "display:flex;align-items:center;justify-content:center;padding:16px;"
)
LIGHTBOX_INNER_STYLE = (
    "max-width:min(96vw,1060px);max-height:94vh;overflow:auto;padding:12px;"
    "border-radius:12px;background:var(--background-fill-primary,#fff);"
    "color:var(--body-text-color,#111);box-shadow:0 14px 44px rgba(0,0,0,0.45);"
)
LIGHTBOX_BAR_STYLE = (
    "display:flex;flex-wrap:wrap;gap:10px;align-items:center;"
    "justify-content:space-between;font-size:13px;margin:6px 0;"
)
LIGHTBOX_IMAGE_STYLE = (
    "display:block;max-width:min(90vw,1020px);max-height:62vh;margin:8px auto;"
    "border-radius:8px;background:var(--background-fill-secondary,#f3f4f6);"
)
LIGHTBOX_WORDING_STYLE = (
    "display:grid;grid-template-columns:1fr 1fr;gap:4px 14px;font-size:13px;"
    "margin-top:8px;align-items:start;"
)
# The icon that goes with a link out of the app, so a link reads as one at a glance
# without the address being printed next to it.
EXTERNAL_ICON = (
    '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
    'style="vertical-align:-1px"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 '
    '0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>'
    "</svg>"
)

# A page can come back enormous; this is only a guard against one that would not fit
# through the browser at all, and it is smaller than it was because the panel is now
# rendered again every time a piece of its translation lands -- the fold is paid for
# on every frame. What is cut is said in the fold itself.
DETAIL_MAX_CHARS = 400_000

# The legacy fragment's text is a few kilobytes; this keeps a pathological one from
# filling the panel. It is also what a line of it is capped at before it is sent to be
# translated, so the line the panel shows and the line it asked about are the same one.
DETAIL_INFO_MAX_CHARS = 20_000

# How many product pages are held in memory, so a second click on a card costs nothing
# (see `cached_detail`). Each one carries the raw page, so this is a small number.
DETAIL_CACHE_SIZE = 3
_DETAIL_CACHE = {}

# The panel is held per session, not in a `gr.State`: a state is the copy the browser
# hands back with the next click, and a frame that was already on its way when the panel
# was replaced carried the old panel with it (see `panel_open`). A few sessions are kept
# for as long as they are the ones clicking.
DETAIL_SESSIONS = 6
_PANEL_STATES = {}

# How many panel events may be in flight at once. One is the queue's default and is what
# made a click on a second card wait for the first card's translation to finish; three is
# a translation, the click that replaced it, and a picture in between.
DETAIL_CONCURRENCY = 3

# The cards are not links any more, so they need a hover and focus cue of their
# own to read as clickable. `css_template` styles are scoped to the component.
CARD_CSS = """
.mm-card { transition: border-color 120ms ease, box-shadow 120ms ease; }
.mm-card:hover {
    border-color: var(--color-accent, #c2410c);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}
.mm-card:focus-visible {
    outline: 2px solid var(--color-accent, #c2410c);
    outline-offset: 2px;
}
"""

# Clicking a card asks the server for that product's page instead of opening it
# in a new tab. The results stream in, so the listener sits on the document and
# matches whichever card was clicked; `trigger` is the gr.HTML component's own
# event trigger, and its payload arrives in `open_detail` as a `gr.EventData`.
# This runs again if the component is re-rendered, hence the guard -- the newest
# `trigger` is kept in a global so the listener never holds a stale one.
CARD_CLICK_JS = """
window.momoTrigger = trigger;
if (!window.momoCardClick) {
  window.momoOpenDetail = function (card) {
    var url = card.getAttribute('data-mm-url');
    if (!url || !window.momoTrigger) { return; }
    window.momoTrigger('click', {
      url: url,
      title: card.getAttribute('data-mm-title') || ''
    });
    var panel = document.getElementById('mm-detail');
    if (panel) { panel.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  };
  window.momoCardFrom = function (event) {
    return event.target && event.target.closest
      ? event.target.closest('[data-mm-url]')
      : null;
  };
  window.momoCardClick = function (event) {
    var card = window.momoCardFrom(event);
    if (card) { event.preventDefault(); window.momoOpenDetail(card); }
  };
  window.momoCardKey = function (event) {
    if (event.key !== 'Enter' && event.key !== ' ') { return; }
    var card = window.momoCardFrom(event);
    if (card) { event.preventDefault(); window.momoOpenDetail(card); }
  };
  document.addEventListener('click', window.momoCardClick);
  document.addEventListener('keydown', window.momoCardKey);
}
"""


DETAIL_CSS = """
.mm-zoom { cursor: zoom-in; transition: filter 120ms ease, box-shadow 120ms ease; }
.mm-zoom:hover {
    filter: brightness(1.04);
    box-shadow: 0 0 0 2px var(--color-accent, #c2410c);
}
.mm-zoom:focus-visible {
    outline: 2px solid var(--color-accent, #c2410c);
    outline-offset: 2px;
}
.mm-link {
    color: var(--color-accent, #c2410c);
    text-decoration: none;
    border-bottom: 1px dotted currentColor;
    cursor: pointer;
}
.mm-link:hover { border-bottom-style: solid; }
.mm-btn { font: inherit; }
.mm-btn:hover {
    border-color: var(--color-accent, #c2410c);
    color: var(--color-accent, #c2410c);
}
.mm-detail summary { cursor: pointer; }
"""

# The panel's own clicks -- a picture opened bigger, the wording on one asked for, the
# lines that came back in Chinese asked for again -- are one listener on the document,
# the way the cards are, and they all arrive at the same event: the payload's `action`
# is what tells them apart. `trigger` is the gr.HTML component's own event trigger, and
# it is kept in a global because the panel is re-rendered on every frame, so a listener
# that captured the first one would stop working after the first translation landed.
DETAIL_JS = """
window.momoDetailTrigger = trigger;
if (!window.momoDetailSetup) {
  window.momoDetailSetup = true;
  window.momoDetailPayload = function (event) {
    var node = event.target;
    if (!node || !node.closest) { return null; }
    var found = node.closest('[data-mm-translate-image]');
    if (found) {
      return { action: 'image-translate',
               url: found.getAttribute('data-mm-translate-image') };
    }
    found = node.closest('[data-mm-translate-page]');
    if (found) {
      return { action: 'page-translate',
               url: found.getAttribute('data-mm-translate-page') };
    }
    found = node.closest('[data-mm-image]');
    if (found) {
      return { action: 'image', url: found.getAttribute('data-mm-image') };
    }
    return null;
  };
  window.momoDetailClick = function (event) {
    if (!window.momoDetailTrigger || !event.target) { return; }
    // Only the backdrop itself and the close button close the overlay; a click on
    // anything inside it belongs to whatever was clicked.
    if (event.target.hasAttribute && event.target.hasAttribute('data-mm-close')) {
      event.preventDefault();
      window.momoDetailTrigger('click', { action: 'image-close' });
      return;
    }
    var payload = window.momoDetailPayload(event);
    if (payload) {
      event.preventDefault();
      window.momoDetailTrigger('click', payload);
    }
  };
  window.momoDetailKey = function (event) {
    if (event.key !== 'Enter' && event.key !== ' ') { return; }
    var payload = window.momoDetailPayload(event);
    if (payload) {
      event.preventDefault();
      window.momoDetailTrigger('click', payload);
    }
  };
  document.addEventListener('click', window.momoDetailClick);
  document.addEventListener('keydown', window.momoDetailKey);
}
"""


def message_html(text):
    """A small styled message, used for the empty and error states."""
    return f'<div style="{MESSAGE_STYLE}">{html.escape(str(text))}</div>'


DETAIL_HINT = message_html(
    "Click a product to read its page here. The page is fetched and read on the "
    "server -- 商品特色 (the rich page that section embeds included), 商品規格 and "
    "商品說明 -- section by section, with the raw HTML one click away behind it, and "
    "its text is translated into your language as each answer arrives. A picture in "
    "the panel opens bigger when clicked, and the wording written into it can be "
    "translated into your language there."
)


def card_html(item, title, note):
    """One product as a single row: thumbnail on the left, text on the right."""
    image = item.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    # The "?t=..." cache-buster is not needed to load the image.
    image = (image or "").split("?", 1)[0]

    if image:
        thumb = (
            f'<img src="{html.escape(image, quote=True)}" alt="" '
            f'loading="lazy" style="{THUMB_STYLE}">'
        )
    else:
        thumb = f'<div style="{NO_IMAGE_STYLE}">no image</div>'

    offers = item.get("offers") or {}
    rating = item.get("aggregateRating") or {}

    meta = []
    if offers.get("price") is not None:
        price = f"{offers.get('priceCurrency', '')} {offers['price']}".strip()
        meta.append(f'<span style="{PRICE_STYLE}">{html.escape(price)}</span>')
    if rating.get("ratingValue"):
        stars = f"⭐ {rating['ratingValue']}"
        if rating.get("reviewCount"):
            stars += f" ({rating['reviewCount']})"
        meta.append(f'<span style="{RATING_STYLE}">{html.escape(stars)}</span>')

    note_html = f'<div style="{NOTE_STYLE}">{html.escape(note)}</div>' if note else ""
    url = html.escape(str(item.get("url") or ""), quote=True)

    # A card is not a link any more: clicking it shows the product page in the
    # detail panel above the results, so the source can be inspected -- and
    # later stripped -- without leaving the search.
    return (
        f'<div class="mm-card" data-mm-url="{url}" '
        f'data-mm-title="{html.escape(title, quote=True)}" '
        f'role="button" tabindex="0" style="{CARD_STYLE}">'
        f"{thumb}"
        f'<div style="{TEXT_STYLE}">'
        f'<div style="{TITLE_STYLE}">{html.escape(title)}</div>'
        f'<div style="{META_STYLE}">{"".join(meta)}</div>'
        f"{note_html}"
        "</div></div>"
    )


def render_results(keyword, language, items, rows):
    """Render the heading plus one card per product that is ready to be shown.

    With `REVEAL_TRANSLATED_ONLY` a product whose text has not come back
    translated yet is left out of the page rather than shown in Chinese, so the
    page only ever shows the language the user asked for and fills up product by
    product as the answers land. Until the first one arrives the page says how
    many products are being translated instead of showing nothing at all.
    """
    if not items:
        return message_html(f"No items found for 「{keyword}」")

    visible = list(zip(items, rows))
    if language in RESULT_LANGUAGE and REVEAL_TRANSLATED_ONLY:
        visible = [
            (item, row) for item, row in visible if not looks_untranslated(row)
        ]
    planned = translate_plan(len(items))
    if not visible:
        return message_html(
            f"Translating {planned} product(s) for 「{html.escape(keyword)}」..."
        )

    heading = f"Found {len(items)} items for 「{html.escape(keyword)}」"
    if language in RESULT_LANGUAGE:
        heading += f" — translated into {RESULT_LANGUAGE[language]}"
        if len(visible) < planned:
            heading += f" ({len(visible)} of {planned} translated so far)"

    # With the testing cap only some of the page is translated, so what the rest are
    # showing is said out loud rather than left to look like a page in that language.
    limit_note = ""
    if language in RESULT_LANGUAGE and planned < len(items):
        limit_note = (
            f'<div style="{NOTE_STYLE}">The first {planned} of {len(items)} products '
            f"are translated while the flow is being tested "
            f"(TRANSLATE_LIMIT={TRANSLATE_LIMIT}); the rest are shown as scraped."
            "</div>"
        )

    cards = "".join(
        card_html(item, title, note) for item, (title, note) in visible
    )
    return (
        f'<div style="{HEAD_STYLE}">{heading}</div>'
        f"{limit_note}"
        f'<div style="{LIST_STYLE}">{cards}</div>'
    )


def detail_fact(value):
    """One label of the summary row, escaped."""
    return f'<span style="{NOTE_STYLE}">{html.escape(value)}</span>'


def detail_translation(translations, key, original):
    """What the panel shows for one scraped line: its translation, or the line itself.

    A line that was never asked for -- one past the point the translation stops -- a line
    whose request did not come back, and a line whose translation simply has not landed
    yet all have no entry in `translations`, so this falls back to the text it was scraped
    with. That is what the page shows under the translating layer, and what the head
    counts when the endpoint never answered for some of them.
    """
    return translations.get(key, original)


def detail_line(translations, key, original, style=DETAIL_BODY_STYLE):
    """One scraped line as the panel shows it: its translation, and nothing else.

    The translation is the whole of the line. The Chinese it replaced used to be kept
    in a `原文` fold under it, which is a second copy of every line on the screen to read
    past -- in the language the reader came here not to read -- for a comparison the
    product page itself is already one labelled link away for.

    Until a line's translation lands, the line is the text the page was scraped with:
    the panel opens on the page at once, and the layer over it is what says the lines
    are still being replaced.
    """
    if not original:
        return ""
    translated = detail_translation(translations, key, original)
    return f'<div style="{style}">{html.escape(translated)}</div>'


def link_html(url, label):
    """A link off this page, labelled rather than spelled out.

    The panel printed the address before; it reads off a label now, with the address
    kept in the title for a hover, because a Momoshop URL is long enough to wrap the
    line it is on and to push everything else around it.
    """
    return (
        f'<a class="mm-link" href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener" title="{html.escape(url, quote=True)}">'
        f"{html.escape(label)} {EXTERNAL_ICON}</a>"
    )


def zoom_image_html(url, style):
    """One picture in the panel, made clickable: it opens bigger, with its wording.

    The click is caught in the browser and handed to the server (see `DETAIL_JS`),
    which is the side that can read the wording off the picture and letter it again in
    the target language; the picture itself is still fetched by the browser, exactly
    as it is when nobody clicks it.
    """
    return (
        f'<img class="mm-zoom" src="{html.escape(url, quote=True)}" alt="" '
        f'loading="lazy" data-mm-image="{html.escape(url, quote=True)}" '
        f'role="button" tabindex="0" title="click for a bigger picture, and its '
        f'wording in your language" style="{style}">'
    )


def detail_summary_html(detail, translations):
    """The product at a glance: its image, price, brand and code.

    These come from the page's own `<meta>` tags, which is where the search page's
    cards get the same numbers from, so the panel and the card agree. The title, the
    description and the brand are the page's own text, so they are translated with the
    rest -- a brand line that stayed in Chinese was the last of the page's words on the
    panel; the price, the code and the picture are not words to translate.
    """
    meta = detail["meta"]
    facts = []
    price = " ".join(
        part
        for part in (
            meta.get("product:price:currency"),
            meta.get("product:price:amount"),
        )
        if part
    )
    if price:
        facts.append(f'<span style="{PRICE_STYLE}">{html.escape(price)}</span>')
    for key, label in (
        ("product:brand", "brand"),
        ("product:availability", "availability"),
        ("product:retailer_item_id", "code"),
    ):
        if meta.get(key):
            value = detail_translation(translations, f"meta:{key}", meta[key])
            facts.append(detail_fact(f"{label}: {value}"))

    image = detail["image"]
    thumb = (
        zoom_image_html(image, DETAIL_THUMB_STYLE)
        if image
        else f'<div style="{NO_IMAGE_STYLE}">no image</div>'
    )
    return (
        f'<div style="{DETAIL_ROW_STYLE}">'
        f"{thumb}"
        f'<div style="{TEXT_STYLE}">'
        f'{detail_line(translations, "meta:og:title", meta.get("og:title", ""), DETAIL_LEAD_STYLE)}'
        f'<div style="{DETAIL_FACTS_STYLE}">{"".join(facts)}</div>'
        f'{detail_line(translations, "meta:description", meta.get("description", ""))}'
        "</div></div>"
    )


def detail_bullets_html(bullets, translations, keys):
    """A list of bullet lines, translated where a translation for them is filed.

    A bullet with no translation yet is the line as it was scraped; the layer over the
    panel is what says those lines are on their way.
    """
    items = [
        f"<li>{html.escape(detail_translation(translations, key, bullet))}</li>"
        for key, bullet in zip(keys, bullets)
    ]
    if not items:
        return ""
    return f'<ul style="{DETAIL_LIST_STYLE}">' + "".join(items) + "</ul>"


def detail_feature_html(detail, translations):
    """The 「商品特色」 the product page only embeds: its text, pictures and links.

    The section is an iframe, and what the iframe shows is the supplier's own page,
    which `fetch_feature` read from the product code. That content is usually pictures
    -- the panel shows a strip of them and says that is what the section is -- with
    text when the supplier wrote any. Every picture in the strip opens bigger and can
    have its wording lettered in the target language, and the section's own page is
    one labelled link away rather than an address printed in the panel.
    """
    feature = detail["feature"]
    parts = []
    if feature["text"]:
        parts.append(
            detail_line(
                translations,
                "feature:text",
                feature["text"][:DETAIL_INFO_MAX_CHARS],
            )
        )
    if feature["images"]:
        parts.append(
            f'<div style="{DETAIL_FEATURE_ROW_STYLE}">'
            + "".join(
                zoom_image_html(url, DETAIL_FEATURE_IMAGE_STYLE)
                for url in feature["images"]
            )
            + "</div>"
        )
        if not feature["text"]:
            parts.append(
                f'<div style="{DETAIL_NOTE_STYLE}">The section has no text of its '
                "own: the supplier describes it in the "
                f"{len(feature['images'])} picture(s) above, each of which opens "
                "bigger -- and can have its wording translated into your language -- "
                "when clicked."
                "</div>"
            )
    if feature["url"]:
        parts.append(
            f'<div style="{DETAIL_NOTE_STYLE}">its own page, the one the section\'s '
            f"iframe shows: {link_html(feature['url'], 'open the section page')}</div>"
        )
    elif feature["iframe"]:
        parts.append(
            f'<div style="{DETAIL_NOTE_STYLE}">the section\'s iframe: '
            f"{link_html(feature['iframe'], 'open the iframe')} &middot; nothing was "
            "read out of it for this product</div>"
        )
    return "".join(parts)


def detail_raw_html(page):
    """The page source, escaped and folded away: menus, scripts and styles as sent."""
    total = len(page)
    if total > DETAIL_MAX_CHARS:
        page = page[:DETAIL_MAX_CHARS] + (
            "\n<!-- the rest of this page is not shown: it is longer than "
            f"{DETAIL_MAX_CHARS:,} characters -->"
        )
    return (
        f'<details style="{DETAIL_RAW_STYLE}">'
        f"<summary>raw page source ({total:,} characters)</summary>"
        f'<pre style="{DETAIL_SOURCE_STYLE}">{html.escape(page)}</pre>'
        "</details>"
    )


def detail_sections_html(detail, translations):
    """The page's own content blocks, section by section, translated where it can be.

    「商品特色」 is read first and up out of the list, because it is the one section
    that is more than its block: a headline the page carries beside it, and the rich
    content the page only embeds -- read separately by `fetch_feature`. The legacy
    product-info text follows -- but only for a page whose payload had no paragraphs
    of its own, so the spec table and the description are never printed twice.

    Every line here is filed under the key `detail_texts` gives it. A line whose
    translation has not landed is shown as it was scraped: the panel opens on the whole
    page, and the layer over it says the translation is still running.
    """
    blocks = list(detail["blocks"])
    has_body = any(kind == "text" for kind, _ in blocks)
    parts = []

    bullets_at = next(
        (index for index, (kind, _) in enumerate(blocks) if kind == "format"), None
    )
    if detail["feature"]["headline"] or detail["feature"]["iframe"] or bullets_at is not None:
        # The bullets are lifted into 「商品特色」 from where the page put them, so the
        # rest of the loop skips that index rather than the list dropping it: the keys
        # `detail_texts` files translations under are positions in this same list, and
        # a shorter list would file every block after it under the wrong name.
        bullets = blocks[bullets_at][1] if bullets_at is not None else []
        keys = [f"block:{bullets_at}:{position}" for position in range(len(bullets))]
        parts.append(
            f'<div style="{DETAIL_SECTION_STYLE}">product highlights</div>'
        )
        parts.append(
            detail_line(
                translations,
                "feature:headline",
                detail["feature"]["headline"],
                DETAIL_LEAD_STYLE,
            )
        )
        if bullets:
            parts.append(detail_bullets_html(bullets, translations, keys))
        parts.append(detail_feature_html(detail, translations))

    for index, (kind, value) in enumerate(blocks):
        if index == bullets_at:
            continue
        key = f"block:{index}"
        if kind == "title":
            parts.append(
                f'<div style="{DETAIL_SECTION_STYLE}">'
                f"{html.escape(detail_translation(translations, key, value))}"
                "</div>"
            )
        elif kind == "format":
            keys = [f"{key}:{position}" for position in range(len(value))]
            parts.append(detail_bullets_html(value, translations, keys))
        else:
            parts.append(detail_line(translations, key, value))

    if detail["info_text"] and not has_body:
        parts.append(
            f'<div style="{DETAIL_SECTION_STYLE}">Specs and description from the '
            "legacy product-info page, for a shell that came back without its "
            "own blocks</div>"
        )
        parts.append(
            detail_line(
                translations,
                "legacy:text",
                detail["info_text"][:DETAIL_INFO_MAX_CHARS],
            )
        )

    if not parts:
        parts.append(
            f'<div style="{DETAIL_EMPTY_STYLE}">This page carries no content blocks '
            "at all: it came back as a shell without its data payload.</div>"
        )
    return "".join(parts)


def detail_head_html(detail, title, translations, language, translating, leftovers):
    """The panel's own first lines: the product, the links off it, and its state.

    The address is not spelled out any more: it is a label with the icon, which opens
    the same page in a new tab, and the two counts of what was read stay beside it.
    The state of the translation is said here once it is over -- done, or how many lines
    are still in Chinese -- and that last case is where asking for those lines again is
    offered. While the translation is running the head says nothing about it: the layer
    over the panel says it, with the count of lines translated so far on it.
    """
    head = html.escape(
        detail_translation(translations, "meta:og:title", "")
        or title
        or detail["meta"].get("og:title")
        or "Product page"
    )
    if not language:
        state = "shown as Momoshop wrote it: the language you searched in is Chinese"
    elif translating:
        state = ""
    elif leftovers:
        state = (
            f"translated into {language} &middot; {leftovers} line(s) are still in "
            "Chinese -- the endpoint did not answer for them, or one click asks for "
            "only part of a page this long"
        )
    else:
        state = f"translated into {language}"

    controls = [link_html(detail["url"], "original page")]
    if leftovers and not translating:
        controls.append(
            f'<button class="mm-btn" style="{BUTTON_STYLE}" '
            f'data-mm-translate-page="{html.escape(detail["url"], quote=True)}">'
            f"ask again for those {leftovers} line(s)</button>"
        )
    counts = (
        f"{len(detail['page']):,} characters of raw HTML, "
        f"{detail['payload_chars']:,} of them the payload the page renders from"
    )
    state_html = f'<div style="{DETAIL_URL_STYLE}">{state}</div>' if state else ""
    return (
        f'<div style="{DETAIL_HEAD_STYLE}">{head}</div>'
        f'<div style="{DETAIL_URL_STYLE}">{" ".join(controls)} &middot; {counts}</div>'
        f"{state_html}"
    )


def picture_note(language):
    """What the panel says the button on a picture does, in one place for both uses.

    With a key for the picture model that is the whole job in one request: the picture
    goes out, and the picture with its wording translated comes back. Saying exactly
    that -- an edit of this picture rather than a new picture of the product -- is the
    difference between a working feature and a misleading one. The fallback cannot say
    it, which is why the two are written out separately rather than blended.
    """
    if PICTURE_KEY:
        return (
            f"{PICTURE_MODEL} is asked to translate the Chinese wording on this picture "
            f"into {language} and to keep everything else on it as it is -- the layout, "
            "the product graphics, the arrows, the numbers and the background -- so what "
            "comes back is this picture with its wording changed rather than a new "
            "picture of the product."
        )
    return (
        f"{IMAGE_MODEL} reads the Chinese wording off this picture and the same "
        f"picture is lettered again in {language} where that wording was. No model "
        "here can draw a picture -- they all answer in text -- so this is the picture "
        "Momoshop published with its wording replaced, not a new one."
    )


def picture_wait():
    """What the lightbox says while the picture step is running.

    One line, whichever path is doing the work: until the picture comes back, that
    waiting is the whole of what the panel knows. What the step does inside -- one
    request to an image model, or a reading, a translation and a drawing -- is what the
    note beside the button said before it was pressed (`picture_note`), so it is not
    said again here as a paragraph to read while the picture is being made.
    """
    return "Translating ..."


def lightbox_html(lightbox, language):
    """The picture that was clicked, bigger, with the wording step on it.

    An overlay rather than a page of its own, so the panel behind keeps its place in
    the results, and the picture is still the one the browser fetches, exactly as it is
    in the strip. Clicking the backdrop or the close button puts it away, and the button
    beside the picture asks the server to translate the wording on it -- the picture
    goes out to a model that answers with a picture, so the frame before it is on the
    screen first, and nothing is hidden when it goes wrong: the error is the frame.
    """
    url = lightbox["url"]
    stage = lightbox.get("stage", "open")
    result = lightbox.get("result") or {}
    shown = result.get("picture") or url
    parts = [
        f'<div class="mm-lightbox" style="{LIGHTBOX_STYLE}" data-mm-close="1">',
        f'<div class="mm-lightbox-inner" style="{LIGHTBOX_INNER_STYLE}">',
        f'<div style="{LIGHTBOX_BAR_STYLE}">'
        f'<span style="{DETAIL_LEAD_STYLE}margin:0">the picture</span>'
        f'<button class="mm-btn" style="{BUTTON_STYLE}" data-mm-close="1">'
        "close ✕</button></div>",
        f'<img src="{html.escape(shown, quote=True)}" alt="" '
        f'style="{LIGHTBOX_IMAGE_STYLE}">',
    ]

    if stage == "reading":
        parts.append(
            f'<div style="{DETAIL_NOTE_STYLE}">{html.escape(picture_wait())}</div>'
        )
    elif stage == "error":
        parts.append(
            f'<div style="{DETAIL_NOTE_STYLE}">that did not work: '
            f'{html.escape(str(lightbox.get("error") or "no answer from the endpoint"))}'
            "</div>"
        )
    elif stage == "done":
        parts.append(detail_wording_html(result, url))
    else:
        parts.append(
            f'<div style="{LIGHTBOX_BAR_STYLE}">'
            f'<button class="mm-btn" style="{BUTTON_STYLE}" '
            f'data-mm-translate-image="{html.escape(url, quote=True)}">'
            f"translate the wording into {html.escape(language)}</button></div>"
        )
        parts.append(
            f'<div style="{DETAIL_NOTE_STYLE}">{html.escape(picture_note(language))}'
            "</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def detail_wording_html(result, url):
    """What the picture step did, for the frame under the picture.

    `PICTURE_MODEL` answers with the picture itself rather than with lines and boxes to
    draw in, so that path has no list to show: the picture above the note is the whole
    answer, and the note says where it came from. The other path is read-and-re-letter,
    and there this is the part worth reading when the lettering could not be placed -- a
    line written sideways, or one the model placed too small to draw -- so the
    translation of a line that stayed as it was is in the list either way, and the count
    below says how many of each there are.
    """
    if result.get("model") == PICTURE_MODEL:
        kept = picture_folder()
        where = (
            f"kept in {html.escape(str(kept))}"
            if kept is not None
            else "not kept on disk, this host having nowhere to write"
        )
        return (
            f'<div style="{DETAIL_NOTE_STYLE}">{html.escape(PICTURE_MODEL)} returned this '
            "picture with its Chinese wording translated: the layout, the product "
            "graphics, the arrows, the numbers and the background on it are the model's "
            "own edit of the picture Momoshop published, and nothing was drawn over it "
            f"here. The picture that went out and this one are both {where}, and the "
            "console says how long the call took and how much of the picture it changed."
            "</div>"
            f'<div style="{LIGHTBOX_BAR_STYLE}">'
            f'<button class="mm-btn" style="{BUTTON_STYLE}" '
            f'data-mm-image="{html.escape(url, quote=True)}">show the original picture'
            "</button></div>"
        )
    read = result.get("read") or []
    wording = result.get("wording") or []
    if not read:
        return (
            f'<div style="{DETAIL_NOTE_STYLE}">{html.escape(IMAGE_MODEL)} read no '
            "wording on this picture, so there is nothing to replace.</div>"
            f'<div style="{LIGHTBOX_BAR_STYLE}">'
            f'<button class="mm-btn" style="{BUTTON_STYLE}" '
            f'data-mm-image="{html.escape(url, quote=True)}">show the original picture'
            "</button></div>"
        )

    rows = "".join(
        f'<div style="{DETAIL_NOTE_STYLE}">{html.escape(text)}</div>'
        f'<div style="{DETAIL_BODY_STYLE}">'
        f'{html.escape(wording[number] if number < len(wording) else text)}</div>'
        for number, text in enumerate(read)
    )
    drawn = result.get("drawn", 0)
    left = len(read) - drawn
    note = f"{drawn} of {len(read)} line(s) were lettered into the picture"
    if left:
        note += (
            f"; {left} were left as they are -- written sideways, or placed too small "
            "to letter -- and are in the list above with their translation"
        )
    return (
        f'<div style="{LIGHTBOX_WORDING_STYLE}">{rows}</div>'
        f'<div style="{DETAIL_NOTE_STYLE}">{note}.</div>'
        f'<div style="{LIGHTBOX_BAR_STYLE}">'
        f'<button class="mm-btn" style="{BUTTON_STYLE}" '
        f'data-mm-image="{html.escape(url, quote=True)}">show the original picture'
        "</button></div>"
    )


def detail_veil_html(content, label):
    """`content`, half washed out in the panel's own colour, with `label` riding on top.

    Two layers rather than one: the wash is what dims the page to 0.5, and the label is
    over the wash so the label itself stays at full strength. Neither takes a click, so
    the page underneath can still be scrolled and its links still work while it is being
    translated.
    """
    return (
        f'<div style="{DETAIL_VEIL_WRAP_STYLE}">'
        f"{content}"
        f'<div style="{DETAIL_VEIL_STYLE}"></div>'
        f'<div style="{DETAIL_VEIL_LABEL_STYLE}">'
        f'<div style="{DETAIL_VEIL_PILL_STYLE}">{html.escape(label)}</div>'
        "</div></div>"
    )


def detail_veil_label(language, done=0, total=0):
    """What the layer says: that it is translating, into what, and how far it has got.

    The count is left off a page whose lines have not been read yet -- the first frame of
    a click, before the page arrives -- because "0 of 0" would be a claim about a page
    the panel has not seen.
    """
    label = f"Translating ... into {language}" if language else "Translating ..."
    return f"{label} ({done} of {total} line(s))" if total else label


def detail_waiting_html(title, language):
    """The panel while the page it just opened has not been fetched yet.

    The click is answered before the fetch, so this is the frame that is on the screen
    while the page is a network round trip away: the product that was clicked, dimmed
    under the same layer the translating frames wear. It is the same shape as the panel
    that replaces it, so the page arrives under the layer rather than instead of it.
    """
    return detail_veil_html(
        f'<div style="{DETAIL_STYLE}">'
        f'<div style="{DETAIL_HEAD_STYLE}">{html.escape(title or "This product")}</div>'
        f'<div style="{DETAIL_EMPTY_STYLE}">reading the page ...</div>'
        "</div>",
        detail_veil_label(language),
    )


def detail_html(
    detail,
    title,
    error="",
    translations=None,
    language="",
    translating=False,
    leftovers=0,
    lightbox=None,
):
    """The detail panel: one product page, read out of the payload it carries.

    `product_detail` fetched the page and decoded its React payload and `<meta>` tags;
    this lays that out as sections, with the raw source kept behind a fold for whatever
    the sections do not cover. Every value is escaped, so the page's own scripts and
    styles are characters on the screen rather than anything that can run in the app or
    break out of the panel.

    `translations` is what the panel's text has been translated into so far, keyed the
    way `detail_texts` keys it. While `translating` is set, the whole page is on the
    screen -- as the page wrote it where no translation has landed yet -- under one layer
    washed to half opacity that says it is translating; the layer is what the reader is
    meant to notice, and it goes when the last line does. `lightbox` is a picture that
    was clicked, drawn over the panel; the panel behind it is rendered either way, and the
    lightbox is outside the layer because it is not part of what is being translated.
    """
    translations = translations or {}
    url = detail.get("url", "")
    if error:
        return message_html(f"Could not load {url or 'that page'}: {error}")

    panel = (
        f'<div style="{DETAIL_STYLE}">'
        f"{detail_head_html(detail, title, translations, language, translating, leftovers)}"
        f"{detail_summary_html(detail, translations)}"
        f"{detail_sections_html(detail, translations)}"
        f"{detail_raw_html(detail['page'])}"
        "</div>"
    )
    if translating:
        keys = [key for key, _ in detail_texts(detail)]
        done = sum(1 for key in keys if key in translations)
        panel = detail_veil_html(panel, detail_veil_label(language, done, len(keys)))

    return panel + (lightbox_html(lightbox, language) if lightbox else "")


def panel_key(request):
    """Which session's panel a run is about: the browser session Gradio hands it.

    A caller that is not a browser -- the API, a test -- has no session hash and gets the
    one shared panel, which is what this was before the panel was per session.
    """
    return getattr(request, "session_hash", None) or ""


def panel_held(key):
    """The panel this session is showing, as the server holds it."""
    return _PANEL_STATES.get(key)


def panel_open(key, detail, title, language, target):
    """Start this session's panel on `detail`, as the newest of its own generations.

    The panel is held here rather than in a `gr.State` because a state is the copy the
    browser hands back with the next click: a card click's translation runs for as long
    as the endpoint takes, and any frame of it that was already on its way when another
    card was clicked carried the *old* panel with it -- which is how a click on a second
    card could be answered with the first card's page, and how a picture that was open
    could be taken away by a frame that knew nothing about it. Every frame is rendered
    from what the session is showing now, and the generation is what tells a run that it
    is no longer the one.
    """
    previous = _PANEL_STATES.get(key)
    state = {
        "generation": 1 if previous is None else previous["generation"] + 1,
        "detail": detail,
        "title": title,
        "translations": {},
        "language": language,
        "target": target,
        "lightbox": None,
        "translating": bool(target),
        "leftovers": 0,
    }
    _PANEL_STATES.pop(key, None)
    _PANEL_STATES[key] = state
    while len(_PANEL_STATES) > DETAIL_SESSIONS:
        _PANEL_STATES.pop(next(iter(_PANEL_STATES)))
    return state


def panel_take(key):
    """Take the panel over: this run is the one that writes it from here on.

    Asking for the lines that came back in Chinese is not a new panel -- it is the same
    page and what has already been translated -- so this keeps the state and only
    becomes its newest generation, which is what stops an older run from writing over
    it.
    """
    state = _PANEL_STATES.get(key)
    if state is not None:
        state["generation"] += 1
    return state


def panel_close(key):
    """Put this session's panel away: nothing is open, and every run on it is old news.

    Dropping the state rather than only bumping its generation is what makes the panel
    closed and not merely out of date: a run that is still translating finds no state to
    write to (`panel_update` returns `None`) and goes quiet, and a click that arrives
    from the panel that is no longer on the screen -- a picture, a retry -- finds nothing
    to act on either.
    """
    _PANEL_STATES.pop(key, None)


def panel_update(key, generation, **values):
    """Write to this session's panel, but only while this run still owns it.

    Returns the state it wrote to, or `None` when the panel has been replaced, which is
    a run's signal that it is old news and should end with `panel_quiet()`. That is what
    keeps two clicks that overlap from fighting over the panel: the newest one is the
    panel, and the one before it goes quiet instead of putting its page back.
    """
    state = _PANEL_STATES.get(key)
    if state is None or state["generation"] != generation:
        return None
    state.update(values)
    return state


def panel_quiet():
    """The last frame of a run whose panel a newer click has taken over.

    Ending without one is not enough: the value a generator finishes on is sent again as
    the completion of the run, so a run that simply stopped put the page it was showing
    back on the screen a moment after another card had replaced it -- and left it there.
    `gr.skip()` asks for no change at all, which is what "this run is old news" looks
    like on the wire.
    """
    return gr.skip()


def panel_of(key):
    """The panel as this session has it: the one place the panel is put together.

    Every frame of every panel event is rendered from here -- a frame that adds a
    translation and a frame that opens a picture go through the same code with the same
    state, so they cannot disagree about what the panel looks like, and a frame that
    arrives late shows what is there now rather than what was there when it started.
    """
    state = panel_held(key)
    if state is None:
        return DETAIL_HINT
    return detail_html(
        state["detail"],
        state["title"],
        translations=state["translations"],
        language=state["target"],
        translating=state["translating"],
        leftovers=state["leftovers"],
        lightbox=state["lightbox"],
    )


def cached_detail(url):
    """`product_detail` for a page already read, so a second click costs nothing.

    Three pages are kept: a detail carries up to two megabytes of raw HTML, and the
    panel is one product at a time, so this is a short memory rather than a cache that
    grows with the session -- and clicking the same card twice, or coming back to the
    one just read, is free.
    """
    detail = _DETAIL_CACHE.pop(url, None)
    if detail is not None:
        _DETAIL_CACHE[url] = detail
        return detail
    detail = product_detail(url)
    _DETAIL_CACHE[url] = detail
    while len(_DETAIL_CACHE) > DETAIL_CACHE_SIZE:
        _DETAIL_CACHE.pop(next(iter(_DETAIL_CACHE)))
    return detail


def open_detail(language, evt: gr.EventData, request: gr.Request):
    """Fetch the page of the card that was clicked and show what it carries.

    The URL comes from the card, which got it from the search page, and it is checked
    against the Momoshop host before anything is fetched, so the panel cannot be pointed
    at another site from the browser console.

    The click payload is what the card's `js_on_load` passes to `trigger()`, and Gradio
    hands it over as the event's own data: read off the event with `evt.url`, it lands
    in `_data`. A click that carries nothing -- an old tab whose script predates this,
    say -- is the one case that reaches here without it, so the payload is read from
    `_data` and anything unexpected means "no url", rather than an error in the panel.

    The panel is a stream of frames, the way the results page is. The first one is on the
    screen before the page is even fetched, and every frame until the last translation
    lands wears one layer over the page saying it is translating: the page is under it, as
    the page wrote it where an answer has not come back yet, washed to half so the layer
    is what the eye lands on, and the lines turn over to the language that was asked for
    under the layer rather than the reader watching them arrive. A search in Chinese has
    nothing to translate -- the page is in that language already -- so that is a single
    frame, and it is the page, with no layer on it.

    Clicking a second card while this one is still being translated starts a panel of
    its own (see `panel_open`), and this run then ends on `panel_quiet()` rather than
    putting its page back, so the newest click is the panel whatever the endpoint is
    doing.
    """
    key = panel_key(request)
    payload = getattr(evt, "_data", None)
    payload = payload if isinstance(payload, dict) else {}
    url = str(payload.get("url") or "").strip()
    title = str(payload.get("title") or "").strip()
    if not url.startswith(("http://", "https://")) or "momoshop.com.tw" not in url:
        yield DETAIL_HINT
        return
    # The click is answered before the page is fetched, so the panel says what it is
    # doing while the fetch runs rather than leaving the last product on the screen: the
    # page itself is a network round trip of a few seconds. Taking the panel first is what
    # keeps a run that is still translating the last card from putting a frame of it back
    # over this one while the fetch is in the air.
    target = RESULT_LANGUAGE.get(language, "")
    if target:
        panel_take(key)
        yield detail_waiting_html(title, target)
    try:
        detail = cached_detail(url)
    except Exception as error:  # noqa: BLE001 - the panel reports the failure
        print(f"product page failed: {error}", flush=True)
        yield detail_html({"url": url}, title, f"{type(error).__name__}: {error}")
        return
    print(
        f"detail: {len(detail['page']):,} characters, "
        f"{detail['payload_chars']:,} of payload and "
        f"{len(detail['blocks'])} content block(s) from {url}",
        flush=True,
    )

    state = panel_open(key, detail, title, language, target)
    generation = state["generation"]
    yield panel_of(key)

    for translations in translate_detail(detail, language):
        state = panel_update(key, generation, translations=translations)
        if state is None:
            yield panel_quiet()
            return
        yield panel_of(key)

    state = panel_update(
        key,
        generation,
        translating=False,
        leftovers=len(detail_leftovers(detail, state["translations"])),
    )
    if state is None:
        yield panel_quiet()
        return
    yield panel_of(key)


def panel_action(evt: gr.EventData, request: gr.Request):
    """A click inside the panel: a picture opened, its wording asked for, or the lines
    that came back in Chinese asked for again.

    The panel is rendered from what the session is showing rather than fetched again:
    the page was read when the card was clicked, so opening a picture costs one frame
    and not another product page. The wording step is a download and a picture model, so
    its frame says what it is doing before it does it, and a step that fails says why
    instead of leaving the picture as it was with no explanation.

    The language is the one the panel was opened with, not the one selected now: the
    text in front of the user is in that language, and the wording on a picture is
    meant to match it.

    A click on a card while this is running is a new panel (`panel_open`), and this run
    then ends on `panel_quiet()` rather than writing a picture into the page that
    replaced it.
    """
    key = panel_key(request)
    payload = getattr(evt, "_data", None)
    payload = payload if isinstance(payload, dict) else {}
    action = str(payload.get("action") or "")
    url = str(payload.get("url") or "").strip()
    state = panel_held(key)
    if state is None or not state.get("detail"):
        yield DETAIL_HINT
        return
    generation = state["generation"]

    if action == "image":
        panel_update(
            key, generation, lightbox={"url": url, "stage": "open"} if url else None
        )
    elif action == "image-close":
        panel_update(key, generation, lightbox=None)
    elif action == "image-translate":
        reading = {"url": url, "stage": "reading"}
        if panel_update(key, generation, lightbox=reading) is None:
            yield panel_quiet()
            return
        yield panel_of(key)
        result = picture_result(url, state.get("target") or "English")
        if panel_update(key, generation, lightbox=result) is None:
            yield panel_quiet()
            return
    elif action == "page-translate":
        state = panel_take(key)
        if state is None:
            yield DETAIL_HINT
            return
        generation = state["generation"]
        detail, translations = state["detail"], state["translations"]
        left = {name for name, _ in detail_leftovers(detail, translations)}
        if panel_update(key, generation, translating=True, leftovers=len(left)) is None:
            yield panel_quiet()
            return
        yield panel_of(key)
        for chunk in translate_detail(detail, state["language"], only=left):
            translations.update(chunk)
            if panel_update(key, generation, translations=translations) is None:
                yield panel_quiet()
                return
            yield panel_of(key)
        if panel_update(
            key,
            generation,
            translating=False,
            leftovers=len(detail_leftovers(detail, translations)),
        ) is None:
            yield panel_quiet()
            return

    yield panel_of(key)


def picture_result(url, language):
    """`image_translated`, or the frame that says why it did not happen.

    A picture step is a download and a model answer at the very least, which is more
    places to fail than the rest of the panel has -- a key that is not set, a prompt the
    model refuses, a picture that is too big -- so the lightbox shows the failure rather
    than the picture as it was: "nothing happened" and "that did not work" are different
    things to the person looking at it.
    """
    if not url.startswith(("http://", "https://")) or "momoshop.com.tw" not in url:
        return {"url": url, "stage": "error", "error": "not a Momoshop picture"}
    try:
        return {"url": url, "stage": "done", "result": image_translated(url, language)}
    except Exception as error:  # noqa: BLE001 - the lightbox reports the failure
        print(f"picture failed: {error}", flush=True)
        return {"url": url, "stage": "error", "error": f"{type(error).__name__}: {error}"}


def search(query, language, request: gr.Request, progress=gr.Progress()):
    """Search Momoshop and stream the result cards, translating them as we go.

    A search puts the detail panel away as well: the product that was open belongs to
    the results this run is replacing, and leaving it on the screen would hang another
    search's page over this one's cards. That is the run's first frame, before the query
    is even translated -- `gr.skip()` leaves the cards as they are, so the panel goes at
    the click rather than when Momoshop answers. The hint goes out with every frame after
    it too, because the panel is a component of its own: a translation that is still
    landing for the page being put away must not be the last word on it.
    """
    panel_close(panel_key(request))
    yield gr.skip(), DETAIL_HINT
    if not (query or "").strip():
        yield message_html("Please type a search query."), DETAIL_HINT
        return

    progress(0.05, desc="Translating your query...")
    keyword = (
        translate(query, TARGET_LANGUAGE)
        if language in TRANSLATE_TO_TRADITIONAL_CHINESE
        else query
    )

    progress(0.2, desc=f"Searching Momoshop for 「{keyword}」...")
    items = extract_items(search_momo(keyword))

    if not items:
        yield render_results(keyword, language, items, []), DETAIL_HINT
        return

    # With REVEAL_TRANSLATED_ONLY this frame is the "translating N products"
    # placeholder rather than the page in Chinese: the cards are added one by one as
    # their translations land, and nothing is shown in a language the user did not
    # ask for.
    yield render_results(keyword, language, items, item_texts(items)), DETAIL_HINT

    if language not in RESULT_LANGUAGE:
        return

    # A search starts with a clean slate: the endpoint may well have recovered
    # from whatever rate limited the previous one.
    RATE_LIMITED.clear()

    # The bar counts the products this run actually asks about, which with the
    # testing cap is fewer than the page found.
    planned = translate_plan(len(items))
    for done, rows in enumerate(
        translate_items(items, RESULT_LANGUAGE[language]), start=1
    ):
        # Retries add frames after the first pass, so the step is capped at the
        # number of products the pass walks through.
        step = min(done, planned)
        progress(
            0.2 + 0.7 * step / planned,
            desc=f"Translating results ({step}/{planned})...",
        )
        yield render_results(keyword, language, items, rows), DETAIL_HINT


# Blocks rather than Interface: the detail panel is a second output, written to
# by a card click instead of by `search`, and an Interface has room for only the
# one fn that produces its outputs.
with gr.Blocks(title="Multilanguage Momoshop Search") as demo:
    gr.Markdown(
        "# Multilanguage Momoshop Search\n"
        "Search Momoshop in your own language. Your query is translated to "
        "Traditional Chinese, and every product is translated back for you."
    )
    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="Type your search query here...",
            scale=4,
        )
        language_box = gr.Radio(
            choices=["English", "Indonesia", "Chinese"],
            label="Language",
            value="English",
            scale=1,
        )
    search_button = gr.Button("Search", variant="primary")

    # The content of the last card that was clicked, read out of the page's own
    # payload, translated into the language of the search, and written to again by the
    # clicks inside it -- a picture opened bigger, the wording on it asked for. It sits
    # above the results so a click has somewhere to put it and scrolls to it there; the
    # sections, the picture overlay and the folded source scroll inside their own boxes,
    # so the results stay in reach.
    detail_box = gr.HTML(
        value=DETAIL_HINT,
        elem_id="mm-detail",
        apply_default_css=False,
        css_template=DETAIL_CSS,
        js_on_load=DETAIL_JS,
    )

    # What the panel is showing -- the page that was read, what has been translated so
    # far, which picture is open -- is held by the server per session (`_PANEL_STATES`)
    # rather than in a `gr.State`: a state is the copy the browser hands back with the
    # next click, and a frame that was already on its way when the panel was replaced
    # would carry the old panel with it.

    # One card per product, so every image sits next to its own text.
    results_box = gr.HTML(
        apply_default_css=False,
        elem_id="mm-results",
        css_template=CARD_CSS,
        js_on_load=CARD_CLICK_JS,
    )

    # A search writes both outputs: the cards, and the panel -- which it puts back to
    # its hint, so a product read from the previous search is not left hanging over the
    # results of this one. The progress animation is pinned to the cards, though: left to
    # itself it is drawn on every output of the event, which would put a second
    # `Translating your query...` over the panel.
    search_button.click(
        search,
        [query_box, language_box],
        [results_box, detail_box],
        api_name="predict",
        show_progress_on=results_box,
    )
    query_box.submit(
        search,
        [query_box, language_box],
        [results_box, detail_box],
        show_progress_on=results_box,
    )
    # A card carries its product URL; clicking one asks for that page here, in the
    # language the search is in, and it is translated as it arrives. The clicks inside
    # the panel that comes back are the panel's own event.
    #
    # The two are set up the same way, and it is what makes clicking a second card work
    # while the first one is still being translated:
    #
    #   * `trigger_mode="multiple"` -- under the default ("once") the browser drops a
    #     click that lands while the same event is still running, silently, which is how
    #     a click on another card could do nothing at all;
    #   * a shared `concurrency_id` with more than one slot -- so the newest click runs
    #     next to the translation it replaced instead of queueing behind it:
    #   * and, on the server, `_PANEL_STATES`: whichever click is newest is the panel,
    #     older runs go quiet at their next frame, and no frame can put back a page or
    #     take away a picture that is not what the session is showing.
    results_box.click(
        open_detail,
        [language_box],
        [detail_box],
        concurrency_id="panel",
        concurrency_limit=DETAIL_CONCURRENCY,
        trigger_mode="multiple",
    )
    detail_box.click(
        panel_action,
        None,
        [detail_box],
        concurrency_id="panel",
        concurrency_limit=DETAIL_CONCURRENCY,
        trigger_mode="multiple",
    )

# A hosted runner does not run `python main.py`: it imports this file and serves a
# top-level ASGI app, and without one it refuses the submission with
# `PYTHON_ENTRYPOINT_NOT_FOUND`. `app` is exactly the app `demo.launch()` would build --
# `mount_gradio_app` is Gradio's documented way of serving a Blocks under uvicorn, and
# it wires the same startup (which is what starts the queue) -- and `application` is the
# same object under the other name those runners look for. No server is started here:
# importing the module still does not bind a port.
app = gr.mount_gradio_app(FastAPI(), demo, path="/")
application = app

# Run directly and it is the plain Gradio server, as it always was; `demo.launch()`
# builds its own app from the same `demo`, which is why the `app` above is built at
# import and never served in this case.
if __name__ == "__main__":
    demo.launch()
