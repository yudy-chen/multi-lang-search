# multi-lang-search -- notes

Why the app is built the way it is, and every number it quotes: the endpoint's latency and rate limits, why one product per request, why two chat models, how the detail panel is chunked, and what the picture step does with a Gemini key and without one. None of it is needed to run the app -- [README.md](README.md) is the install-and-test page; this is the evidence behind it.

---

## Running it

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` holds the packages `main.py` imports from outside the standard
library, pinned to the versions this was run on (Python 3.12). Then copy `.env.example`
to `.env` and put the Radeon endpoint's token in `AMD_AI_TOKEN`; `GEMINI_KEY` is
optional, and a process without it reads the wording off a picture and letters the
translation back onto the same picture here instead of asking an image model.
`python main.py` starts the app on the URL `demo.launch()` prints. That is not the only
way in: the file ends with `app = gr.mount_gradio_app(FastAPI(), demo, path="/")` (and
`application = app`), so a hosted runner that imports the module and serves a top-level
ASGI app gets the same Gradio app at the same routes -- `uvicorn main:app` is that
runner. Building it at import binds no port; `python main.py` still goes through
`demo.launch()` and ignores the export. Which model answers
what is a constant at the top of `main.py`, not a package -- or a value in `.env`, when
the endpoint is not the one those constants name (below).

## Changing the endpoint

The Radeon token expires, and the app was written so that this is configuration rather
than a rewrite. `AI_BASE_URL`, `AI_API_KEY` and the four model names are read from
`.env` once at import; unset is the Radeon endpoint and the names it serves, so a `.env`
that only carries the old `AMD_AI_TOKEN` behaves exactly as it did, and a runner that
sets nothing still gets the app every number in this file was measured on.
`AI_API_KEY` is read first and `AMD_AI_TOKEN` second, so the older name keeps working.

What travels: `complete()` is the only place a chat request is built, and it sends
`model` and `messages` and nothing else -- no `response_format`, no `temperature`, no
`extra_body`, no streaming. That is why another provider is a base URL and not a port:
every provider-specific knob the app does not use is a knob it cannot be rejected for.
The rate-limit path is the same shape: a `429` in the error string sets `RATE_LIMITED`,
and the waits around it (`TRANSLATE_RATE_LIMIT_WAIT`, `DETAIL_RETRY_WAIT`) are the
one behaviour every provider does report the same way.

What does not travel: the picture step. `PICTURE_MODEL` is Google's own REST --
`x-goog-api-key`, `generateContent`, `responseModalities` -- on `GEMINI_KEY`, because
the Radeon endpoint serves no model that returns a picture (see below). And the fallback
for a process with no picture key, `IMAGE_MODEL` reading the wording off the picture,
needs a model that accepts a picture, which is the one thing another chat endpoint does
not promise. DeepSeek is the example that matters here: its API is text only now
(`-vision-exp` is retired and its requests are served by a text model), so with
`AI_BASE_URL` pointed there and no `GEMINI_KEY` the lightbox says the picture could not
be read -- `image_translated` raises rather than letting an empty `complete()` pass as
"no wording on this picture". A free path that covers text and vision with one key is
Gemini's OpenAI-compatible layer (`https://generativelanguage.googleapis.com/v1beta/openai/`,
a flash model from `GET /v1beta/openai/models`), which is also the key the picture step
already wants; OpenRouter's `:free` models and a local Ollama cover the text half only,
unless a vision model is named as `IMAGE_MODEL`.

Every number in this file is the Radeon endpoint's, which is why the app is shaped the
way it is: one listing per request, a timeout per request, a query model that answers in
seconds, three translation workers. None of that shape is wrong on a faster endpoint --
`TRANSLATE_WORKERS = 3` and the retry waits become conservative rather than necessary --
but none of the timings transfer either, and the first thing worth re-running after a
switch is `python _endpoint_check.py`, which reads the same four names and asks whichever
endpoint `.env` now names.

`_env_check.py` is the seam without a provider in it: it starts `_mock_provider.py` on a
local port, points `main.py` at it and calls the query, the cards, the panel and the
picture read, then prints the requests the fake endpoint received. What that measures is
that the four names and the key arrive where they are meant to -- `AI_BASE_URL` on every
call, `DETAIL_MODEL` on the panel, `IMAGE_MODEL` on the picture -- including the branch
where a text-only endpoint answers the picture read with a 400 and the lightbox says so
instead of the app raising. It spends no token, which is why it was worth writing.

## How the results stay translated

Momoshop listings are Chinese, so `main.py` translates the product titles and
notes back into the chosen language. Every product is its own request, and a card
is only put on the page once its text is translated: the page never shows the
listings in Chinese and swaps them afterwards, it fills up product by product as
the answers land. Setting `REVEAL_TRANSLATED_ONLY = False` restores the other
behaviour -- the whole page in Chinese straight away, translations filled in as
they arrive -- which is more useful when a page takes minutes and the listings are
recognisable from their images.

The Radeon endpoint is slow rather than broken when it is busy, and its cost is
mostly per request rather than per product. Measured back to back against the live
endpoint while it was busy -- a quiet one answers the same one-listing request in
4.3s to 11.3s:

| request | what happened |
| --- | --- |
| one listing | answered in 31.0s |
| one listing, slightly later | answered in 33.2s |
| three one-listing requests in flight | 31.0s / 33.2s / 59.9s, all three translated |
| three listings in one request | still generating after 150s, nothing translated |
| six listings in one request | did not answer within 150s either |

That is why the page asks for its products one at a time: a loaded endpoint
finishes the small request while the big one is still generating, and a request
that stalls can only leave its own card behind. The price is the request count --
thirty one-listing requests cost about thirty times one request, where five
six-listing batches cost five. `TRANSLATE_BATCH` (1) and `TRANSLATE_WORKERS` (3) are
the dial between the two: raise the batch size when the endpoint has headroom and
the page has to come back in a fifth of the requests.

`REQUEST_TIMEOUT` (60s) is the ceiling for one request, and it sits above the
slow-but-alive case above: the request that landed at 59.9s was a real answer that
a 45s ceiling would have thrown away and asked for again. A ceiling is not free --
a request we gave up on keeps being generated and holds its slot on the endpoint --
which is another reason for one product per request: a small request is done in
seconds, so little of the endpoint is left working on ghost answers.

Retries are per product. After a pass, `translate_items` re-asks for exactly the
products that came back with Chinese still in them -- a request that timed out
returns nothing, and an empty reply reads as Chinese left -- and never for the ones
that already arrived. `TRANSLATE_ATTEMPTS` (3) is how many times a product is asked
for, and anything still in Chinese after the last attempt is counted in the console
instead of passing silently. A retry wave waits before it goes out: `TRANSLATE_RETRY_WAIT`
(5s, twice that for the second wave) and `TRANSLATE_RATE_LIMIT_WAIT` (20s) when the last
wave saw a 429. The products a wave re-asks for are the ones whose requests were
abandoned at the ceiling and are *still generating* on the endpoint, so a wave that
leaves immediately runs beside its own ghosts and asks for slots the endpoint has
already given away -- which is how five products become the fifteen requests a
five-product search can spend (`TRANSLATE_LIMIT` 5 x `TRANSLATE_ATTEMPTS` 3, counted
with the endpoint stubbed out in place of `complete`). The detail half of the flow has
waited between its attempts from the start (`DETAIL_RETRY_WAIT`); this is the same rule
for the requests that cost the most. Cards appear in the order the answers land, so the
page grows while the user watches.

`TRANSLATE_LIMIT` (5) is a testing cap, and the one number here that is not meant to
stay: a page of listings is tens of requests and minutes of waiting, so while the
flow is being tried out only the first five products are sent. The rest are not
dropped -- they keep the text they were scraped with and the page says so above the
cards -- because the count of what the search found has to stay true whatever the
endpoint is asked for. Set it to 0 and the whole page is translated. The query
translation has its own deadline (`QUERY_DEADLINE`, 30s), so a stall there falls back
to the query as typed instead of leaving the page on the progress bar.

`TRANSLATE_WORKERS` is three, which is where the requests stop paying for
themselves. Three one-listing requests were sent together to the loaded endpoint
and all three landed -- 31.0s, 33.2s, 59.9s -- so the page gains a card about every
twenty seconds instead of every thirty. Beyond that the endpoint shares its
capacity between the requests generating on it: three concurrent *six-listing*
batches each ran past 45s while the one-listing ones were coming back, so overlap
what is small and few, not what is big. Requests abandoned past the ceiling are the
worst of it, because they keep generating and hold the slots the retries need.

The endpoint also rate limits the token
(`429 ... token_concurrency_rate_limit_exceeded`) when too many requests share it;
when a request reports that, the flag it raises drops the retries to one request at
a time, because more concurrency is exactly what was refused, and it makes the next
wave wait the longer `TRANSLATE_RATE_LIMIT_WAIT`. A reply clears the flag again, so a
token that has recovered goes back to `TRANSLATE_WORKERS` requests without waiting for
the next search to try it: the flag is a back-off, not a latch. Measured on the live
token while this was being checked: three one-listing requests sent together answered
two of the three inside the same twenty-second window, both translations correct, with
no 429 at all, and four tiny prompts in parallel came back in 1.8-2.7s -- so a 429 is
the token's load at the moment, not a shape these requests have to avoid. That is also
what `_endpoint_check.py` answers when a 429 is what was seen and the question is whether
it is still refusing: it prints the token's model list and asks two of the models for one
word each, so a live token reads as `ok` in seconds and a refused one reads as the `429`
text -- measured after a click that came back untranslated on a 429, the two models
answered `ok` in 2.0s each, which is what says the Chinese in the panel was a moment
rather than a broken setup.

Translation is verified rather than trusted: every reply is checked for leftover
Chinese (`has_chinese`), both because a request that failed comes back as the
original Chinese and because the model sometimes translates only part of a line,
leaving a brand or a bracketed flavour behind. Anything that fails the check is
asked for again -- on its own -- and the prompt itself asks for brand and product
names to be translated as well, including the Chinese half of a name written half
in Latin letters. That last rule was checked against the live endpoint with the
titles that were reported as untranslated:

| listing as scraped | translated into Indonesian |
| --- | --- |
| `【CHU SHIN WU ER 無二】初心真摯20品禮盒-精裝橘235g±3%...` | `【CHU SHIN WU ER Wu Er】Hati Murni Ikhlas 20 Jenis Kotak Hadiah-... 235g±3%...` |
| `HanNaMiChan【花生姑娘】 脫油花生米獨立小包裝30包...` | `HanNaMiChan【Gadis Kacang Tanah】Kacang Tanah Tanpa Minyak Kemasan Kecil Individu 30 Kantong...` |
| `【盛香珍】杏仁小魚乾150g(原味/辣味)` | `【Sheng Xiang Zhen】Ikan Kering Kecil Almond 150g(Rasa Asli/Rasa Pedas)` |

## Why two models

`QUERY_MODEL` (`DeepSeek-V4-Flash`) translates the one short line the user is
waiting on, and `RESULT_MODEL` (`Gemma-4-31B-IT`) translates the page of listings:
different input sizes with different latency budgets, so they are not
interchangeable. `DETAIL_MODEL` is the query model again, for the panel's text -- the
user is sitting on a click there -- and `IMAGE_MODEL` (`Gemma-4-31B-IT`) is the model
that is shown a picture: it takes one as input and answers in text, and it is what the
picture step falls back to when it has no key for an image model. It is not the only
model on the endpoint that takes a picture -- eleven of the sixteen it listed when this
was written name an image among their input modalities, and four of them answered a
64x64 test picture, each with `Red`: `Gemma-4-31B-IT` in 5.5s,
`DeepSeek-V4-Flash-Vision-Exp` in 1.1s, `MiniCPM-V-4.7-1B` in 0.5s and
`MiniCPM-V-4.7-35B-A3B` in 1.5s -- where `DeepSeek-V4-Flash`, the query model, refused
it with `400 ... does not support image input. Remove the image content or use a
vision-capable model`. Gemma is what ships because the step was measured on it, and
on pictures rather than a test square (see "Without a key" below). The key that step
needs, and the image model it names, sit outside the Radeon endpoint: the wording
written into a picture is the model `PICTURE_MODEL` names on `GEMINI_KEY` (see
"A picture, opened bigger" below), so the picture step has a token of its own. The two
chat models are not two rate limits either: both are served from the same token, so a
page of result requests can throttle the query translation that runs
behind it -- measured, a query request came back `429` while result requests were
still generating. That is survivable here because each step has somewhere to fall
back to: a stalled query goes out as the user typed it (`QUERY_DEADLINE`), and a
result request that comes back empty or still Chinese is asked again on its own.

Measured on one listing (盛香珍 peanuts), with the prompt the app sends and then with
the listing line on its own:

| model | one listing | reply |
| --- | --- | --- |
| `DeepSeek-V4-Flash` | 11.3s, 4.3s alone | `1. 【Sheng Xiang Zhen】Snack Foods (Peanut Type / Small Fish Peanuts / Garlic Slices with Small Fish / Garlic Fava Bean Crisps / Salted Egg Yolk Sunflower Seeds) Drinking Snacks \|\| Save up to 100` |
| `Gemma-4-31B-IT` | 6.6s, 38.8s alone | `1. 【Sheng Xiang Zhen】Leisure Snacks (Peanuts / Dried Small Fish with Peanuts / Dried Small Fish with Garlic Slices / Garlic Broad Beans / Salted Egg Yolk Sunflower Seeds) Drinking Snacks \|\| Up to 100 off` |

Both replies are good, and on these runs neither model is reliably the faster of the
two: the spread inside one model (4.3s to 11.3s, 6.6s to 38.8s) is wider than the gap
between them. The number of products is where the page's time actually goes, and there
the shipped pair took **10.1s** for five products with all five translated, against
**6.7s** for the same five on `DeepSeek-V4-Flash` -- close enough that neither of those
is what the split was drawn against. That was `MiMo-V2.6-Flash`: with it as the result
model one listing took **51.1s** -- 85% of `REQUEST_TIMEOUT` (60s), so a slightly slower
answer times out and is paid for again on each of the three attempts -- and the same
five-product search had not finished in **nine minutes**, where the same search with
`DeepSeek-V4-Flash` took **11.1s**. The one line the user is waiting on before anything
is on the page is the worst place for a fifty-second tail, so the query runs on the model
whose worst case measured smallest, and the page of cards runs on `RESULT_MODEL` --
`Gemma-4-31B-IT` today -- where a slow answer costs one card rather than the search. The
one-line change is `RESULT_MODEL`'s value.

## Product pages, read out of the page's own payload

Clicking a card does not open the product page in a new tab any more. The card is
a `div` carrying its own `data-mm-url` (from the search page's JSON-LD), and the
click fetches that page on the server -- a fetch from the browser would be blocked
by CORS -- then shows the product's content in the panel above the results, which
scrolls itself into view. A second click replaces what the panel holds, and the click it
replaced goes quiet: the panel is held on the server, not in a `gr.State` (a state is the
copy the browser hands back with the next click, so a frame already on its way carried
the old panel with it), each click is a generation of that panel, and a run ends on a
frame that asks for no change (`gr.skip()`) once it is no longer the newest generation.

A new search puts the panel away as well. The product the panel was showing belongs to
the results being replaced, so `search` writes the hint back into it -- in its first
frame, before the query is even translated: `gr.skip()` leaves the cards as they are,
so the panel goes with the click on Search rather than with the page that comes back.
Closing drops the panel's held state rather than only making it stale, which is also
what keeps a translation that is still landing for the old page from painting it back
over the new results.

No browser is needed to read the page. A product page is a Next.js app, and the
server already puts the content in the HTML it sends: the React flight payload, one
`self.__next_f.push([1, "..."])` per chunk, holding JSON whose quotes the surrounding
script string has escaped a second time, plus the `<meta>` tags around it. The
browser renders that data; it does not fetch it. So `rsc_payload` decodes the
chunks, `payload_blocks` reads the content blocks out of them in order -- `title`,
`format` (a bullet list), `text` -- and the `<meta>` tags give the name, price,
brand, availability, code and image. That is what the panel lays out as sections:

- **商品特色** -- the site's own heading for this section is a React label, and the
  text under it is the page's `goodsFeatureString` headline plus its bullet list,
  followed by the supplier's own content, which the page only embeds (below).
- **商品規格 / 商品說明** -- the specification table and the description, as the page
  carries them.

The one part that is *not* in the payload is the iframe the 「商品特色」 section shows.
`goodsFeatureUrl` points at `m.momoshop.com.tw/describe.momo`, and that page is itself
a shell: its own script pads the product code to ten digits, splits it 4/3/3 and
points a second iframe at `/expertimg/0014/028/360/mobile/14028360.html` -- the
supplier's rich page, which is where the section's real content is. That path is read
off the shell once and then built here, because the shell's own host refuses as often
as it answers: the same file fetched repeatedly answers 200 with a plain
`User-Agent`, then 403 for a burst of requests, with browser-shaped headers and
cookies included. The content hosts the site's own images come from -- `www.` and
`image.` -- serve the identical file and do not do that, so `feature_urls` builds the
path and `fetch_feature` takes the first host that answers with content.

What comes back is usually pictures rather than a paragraph: 義美's page is 759
characters of HTML and four images, 370 KB of JPEG in all. So the section shows a
strip of them, and says that the section has no text of its own rather than showing
an empty heading. A product with nothing behind it, or a code that is not a plain
number, reads as empty and the panel keeps the headline and the bullets the page
carries.

For the pages that come back as a shell without their payload (measured: 40 KB with
no content blocks in them), the legacy product-info endpoint fills the gap:
`goods/dsgoodsinfo.jsp?i_code=...` serves 商品規格 and 商品說明 as plain HTML on the
host that answers normally, and `fragment_text` turns it into text. It is only used
when the page's own payload had no paragraphs, so nothing is printed twice. Product
images are the same host's `goodsimg` paths, built from the `selectGoodsImagesResult`
data the payload also carries.

Everything the panel prints is escaped, so the page's own scripts and styles are
characters on the screen rather than anything that can run in the app or break out of
its box. The raw source is not thrown away either: it is one click away behind a
closed `<details>`, with `DETAIL_MAX_CHARS` (400 KB) only a guard against a page too
large to send through the browser at all. That number is smaller than it used to be
(2 MB) because the panel is rendered again every time a piece of its translation
lands, so the fold is paid for on every frame of a click.

The click is wired through the component itself rather than through a hidden input
and a button: `js_on_load` on the results component puts one listener on the
document (the cards stream in one at a time, so anything attached per card would
have to be re-attached), that listener matches a card with `closest('[data-mm-url]')`
and calls the component's own `trigger('click', {url, title})`, and the payload
arrives in `open_detail` as a `gr.EventData`. `open_detail` will only fetch a
`momoshop.com.tw` URL, so the panel cannot be pointed at another host from the
browser console, and a page that fails to load is reported in the panel instead of
raising -- `product_detail` is where the two fetches happen (the page and the legacy
fragment), so a failure in either comes back as one message in the panel. The panel
component carries the same kind of listener for its own clicks -- a picture opened,
its wording asked for, the lines still in Chinese asked for again -- and they all
arrive at one event, with an `action` in the payload to tell them apart.

## The panel's text is translated too

The panel used to be the one place that showed Momoshop's Chinese back to an English
search. It is translated now, and the way the results page is: clicking a card streams
frames. The first one is the product that was clicked, before the page is even fetched,
and it already wears the layer every frame wears until the translation is over: the page
as the page wrote it, washed to half opacity in the panel's own colour, with one dark pill
on it -- `Translating ... into English (3 of 19 line(s))`. The lines turn over to the
language that was asked for under the layer, a few at a time, and the layer and the wash
go when the last one does. A search in Chinese has nothing to translate -- the page is in
that language already -- so that is a single frame, and it has no layer on it.

What is translated is decided by the page rather than by a counter. `detail_texts`
walks the same content `detail_sections_html` lays out -- the `<meta>` title,
description and brand, the 「商品特色」 headline, the page's bullets and blocks, the
supplier's text, the legacy fragment -- and gives every line a key built from its place
in that structure (`meta:og:title`, `feature:headline`, `block:1:2`, `legacy:text`). The
panel renders each line through `detail_translation`, so a line with a translation filed
under its key is shown in it, and a line without one is shown as it was scraped -- under
the layer, so what the eye lands on is that the page is being translated rather than the
Chinese behind it. A line the endpoint never answered for is still Chinese when the layer
comes off, which the head counts, with a button beside it that asks for exactly those
lines again. A translated line is one line: the Chinese it replaced is not kept in a fold
under it, because that would be a second copy of every line on the panel to read past in
the language the reader came here not to read, and the page itself is one labelled link
away.

The work goes out in chunks rather than as one page: short lines are numbered, up to
`DETAIL_CHUNK_CHARS` (2,400 characters) per request, and a block with line breaks in
it -- a description, a specification table -- goes out on its own so its shape
survives, up to `DETAIL_REQUEST_CAP` (12) of them. That is what bounds the wait
between two parts of the panel turning over, and the count on the layer is where that
wait is visible. When it is over the head says which state the panel is in:
"translated into English", or "translated into English · 2 line(s) are still in
Chinese", and that second case comes with a button that asks for exactly those lines
again -- a request that timed out leaves one click's worth of work rather than a page
half in Chinese. Measured on the 義美 page: 12 lines, 4 frames, 23.5s, none left in
Chinese.

This is the one place `DETAIL_MODEL` is used, and it is the query model rather than the
result model on purpose: the user is sitting on a click, so the panel wants the model
that answers in seconds -- one listing measured 4.3s and 11.3s on `DeepSeek-V4-Flash`
against 38.8s on `Gemma-4-31B-IT`, and 2.9s against 51.1s when the result model was
`MiMo-V2.6-Flash`, which is the span the split was drawn across. A line the endpoint
refused is asked for again after
`TRANSLATE_RATE_LIMIT_WAIT` rather than after `DETAIL_RETRY_WAIT`, because a 429 is the
token refusing the concurrency the retry is about to ask for again: measured, a click
that landed while the search's own translations were still in flight read
`429 ... token_concurrency_rate_limit_exceeded` and came back with the same Chinese three
seconds later, so a refusal now waits out the same twenty seconds the result retries wait
and only a plain empty answer waits the short one. The page's address is not printed:
it is a labelled link with an icon that opens the same page in a new tab, with the
address itself in the title for a hover. The feature section's own page and its iframe
are labelled links the same way.

`python _panel_check.py` is the offline check for those frames: it renders the panel before
its first translation, half way through and after, from a made-up page with the shape
`product_detail` returns, drives a whole click through `open_detail` and a search through
`search` with the fetch and the endpoint stubbed out, and fails if a translating frame has
anything but exactly one half-opacity layer carrying `Translating ... into <language>`, if
the page is not under it, if the finished frame still wears it or still shows a scraped
line -- the brand line included, which is the one the panel used to leave in Chinese -- if
a search leaves the panel open, or if the picture step's frame is anything but
`Translating ...`.

## A picture, opened bigger, and the wording written into it

Every picture in the panel opens in an overlay when it is clicked: the same picture at
full width, a close button, and one button that asks for the wording written into the
picture, in the language the search is in. That button is one request to an image model,
the one `PICTURE_MODEL` names -- `gemini-2.5-flash-image` is what the table below
measured, and `gemini-3.1-flash-image-preview` is what ships -- on a key of its own
(`GEMINI_KEY` in `.env`, read once into `PICTURE_KEY`): the picture goes out as inline
base64 beside `IMAGE_TRANSLATE_PROMPT` --

> Translate into <language> every piece of Chinese wording in this picture -- the wording
> in the badges, the wording printed on the product packets, and the small print -- and
> put the <language> wording where the Chinese was, in the same place, at the same size,
> in the same colour and slant, so that no Chinese character is left in the picture. The
> products themselves stay as they are: their shapes, colours, photographs, graphics,
> arrows and numbers are not to be changed, only the wording on them replaced.

-- and the picture with its wording translated comes back in the same reply, in the part
that carries `inlineData`. There is no list of lines for that path, because nothing is
read off the picture line by line here: the picture above the note is the whole answer,
and the note says where it came from -- an edit of the picture Momoshop published, not a
new picture of the product.

The wording of that prompt is the difference between a picture that comes back translated
and one that comes back as it was, and the measurement is worth writing down. The first
version of this prompt asked for the picture to be translated and for the product on it
not to be touched (`Retain the exact visual layout, product graphics ... without modifying
any physical object details`), and on a picture of a product the Chinese is printed on
the product: measured on 義美's five-box collage, that call answered `200` with a picture
whose round badge still read 任選4 -- with an `ANY` lettered under it, which is the model
translating without replacing anything -- where the wording above came back with the same
badge as `CHOOSE ANY 4` and the flavour tabs as `MATCHA`, `CHOCOLATE` and `STRAWBERRY`.
Three wordings that ask for the Chinese to be *replaced* all did that; the one that asked
for the picture to be translated and the product left alone did not.

The image model is not on the Radeon endpoint, and that is not a matter of preference.
Measured there:

| what was tried | what came back |
| --- | --- |
| `GET /v1/models` | sixteen models, and no picture among the modalities any of them outputs: `text` for the chat models, `decision` for `Clef`/`d3`/`d3-flash`, `ocr` for `MinerU2.5-Pro`, `audio` for `OmniVoice` |
| `POST /v1/images/generations` with `Gemma-4-31B-IT` | `400 ... is a chat model and cannot be used with /v1/images/generations` |
| `POST /v1/images/edits` with the same model | it wants `images[0].image_url` as a string, then refuses the chat model the same way |
| asking `Gemma-4-31B-IT` for a picture in a chat | `I cannot generate images. I am a text-based AI model.` |

So the step needs a key of its own, and a key that can pay for an image model. Measured
on the key this was built with:

| call | what came back |
| --- | --- |
| `GET /v1beta/models` | `200`, 50 models, `gemini-2.5-flash-image` among them, `generateContent` supported |
| the picture call the app makes | `402 Your prepayment credits are depleted. Please go to AI Studio ... to manage your project and billing.` |
| the same body with a field that does not exist (`unknown_field`) | `400 Invalid JSON payload received. Unknown name "unknown_field" ... Cannot find field.` |
| the same call naming a model that does not exist | `404 models/not-a-model is not found for API version v1beta ...` |
| `gemini-2.5-flash:generateContent` asking for the word ok | `402` as well |
| the same with `gemini-3.8-flash` and `gemini-3.5-flash-lite`, text only, no picture | `402` the same -- so the balance, not the image model, is what is missing |
| the same call after the project's balance was topped up | `200` in 11.8s, `finishReason: STOP`, a picture in `inlineData` (`1024x1024` PNG, 1,783,221 bytes) and 1,290 image tokens in `candidatesTokensDetails` |
| `gemini-2.0-flash` / `gemini-2.5-flash-lite` | `404 ... is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite` |
| a picture + the prompt, sent to `gemini-flash-lite-latest` | `200` with no image part at all: one paragraph (`100%防伪验证 -> 100% Anti-counterfeiting verification`) and `finishReason: STOP` |
| the shipped body with `gemini-2.5-flash-image`, on the 320x320 `_OL.jpg` a card carries | `200` in 11.4s, `1024x1024` PNG, 1,782,863 bytes, 14.6% of the pixels differing from the picture that went out |
| the same on the 640x640 `og:image` of the same product | `200` in 12.0s, `1024x1024` PNG, 1,695,510 bytes, 9.2% of the pixels differing |
| the same on a 421 KB `expertimg` composite, which is what the panel's clicks carry | `200` in 11.4s, `1024x1024` PNG, 2,375,377 bytes, 19.1% of the pixels differing, and every piece of Chinese still legible on it -- a re-render, not an edit |
| the same picture with the wording sharpened (`the reply ... must contain no Chinese, Japanese or Korean character anywhere in it`, `cover it with the colour around it`) | `200` in the same 11.4s and one line of it translated (`2020選手指定品牌` -> `2020 ATHLETE PICK`) where the shipped wording left all of it -- so the wording was left alone rather than churned on one picture |
| `gemini-3.1-flash-image-preview`, the app's own card picture and prompt | `200` with a candidate that has no `content` at all -- no picture, no text, nothing in the body to say why -- where the same model edits a 213 KB ad and a 421 KB composite happily, so `-image` in the name is not enough on its own |
| `GET /v1beta/models`, the names carrying `-image` beside the names carrying `-lite` | six image models, each `generateContent`: `gemini-2.5-flash-image`, `gemini-3.1-flash-image`, `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`, `gemini-3-pro-image-preview` -- and on the same listing `gemini-flash-lite-latest`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, every one of them `generateContent` too |

The 400 and the 404 are what make the 402 readable: this endpoint validates the body
before it looks at the balance, so a body it does not like is refused for that reason
first -- which means the 402 on *this* body says the request was understood, the field
names and the model are real, and the project the key belongs to has nothing left to
spend. The `200` on the model list is the other half of that: the key itself is
accepted, so what is missing is the project's prepay balance rather than the key, the
request or the model -- every model on the list answers `402` the same way, image and
text alike. If AI Studio shows a balance, the project to check is the one this key
belongs to: AI Studio lists which project each key is in, and the balance is per
project, so a key made in another project answers exactly this, and so does a project
whose credit is Cloud credit rather than the prepay plan this message names. Nothing
about the request changes when that is fixed; there is simply no picture
until it is, and the panel is where that shows, because the failure is the frame: the
overlay reads `that did not work: RuntimeError: gemini-2.5-flash-image answered 402: Your
prepayment credits are depleted...` rather than sitting there as if the click had not
registered.

`PICTURE_MODEL` is the one line to edit when a better drawing is wanted or this model is
retired, and it is worth reading twice: a name from that listing is not a guess this step
can afford, because a wrong one is a `200` with prose in it and no picture, which reaches
the overlay as `that did not work: RuntimeError: ... returned no picture`. `_model_check.py`
is the hand's-breadth check to run before a change to `PICTURE_MODEL` is believed: it reads
the name the app actually holds out of `main.py` -- and the prompt and the body with it --
and sends it one of the pictures the app kept, and it prints which names on the listing
return a picture at all. `python _model_check.py <picture>` sends that picture instead,
which is the check that matters: the pictures this app is handed are not interchangeable,
and a name can return a picture for one of them and nothing at all for another.

The `200` on `generateContent` is what makes the name look right when it is not. On this
key's listing the text models and the image models are one list, and
`supportedGenerationMethods` reads `generateContent` for `gemini-flash-lite-latest` exactly
as it does for `gemini-2.5-flash-image` -- so the listing cannot be read for it. What can
be read is the reply: an image model answers with an `inlineData` part, and everything else
answers with a paragraph about the picture it was handed. Every name that returns one
carries `-image` (`gemini-2.5-flash-image`, `gemini-3.1-flash-image`,
`gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`,
`gemini-3-pro-image-preview` on the listing measured above), and that is the whole of the
signal: `gemini-flash-lite-latest` and `gemini-3.1-flash-lite-image` are two lines apart on
it and only one of them returns the picture. What the `-image` is *not* is a promise for a
particular request: `gemini-3.1-flash-image-preview` returns pictures for the ad and the
composite measured above and nothing at all -- a candidate with no `content` in it -- for
the 320x320 picture a card carries, so the name and the picture are checked together rather
than the name alone. If the model this step uses is ever retired, or
a better drawing is wanted, the one line to edit is `PICTURE_MODEL`.

### What the picture step leaves behind: the folder and the console

An answer that is the picture it was given, or the same picture at a larger size with the
Chinese still on it, is a `200` carrying an image, and in the overlay that looks exactly
like a translation -- the panel shows one picture and gives no way to tell which one it
is. So the step keeps both pictures and says what happened, every time it runs:

| kept in `translated_pictures/` | what it is |
| --- | --- |
| `<picture>-<language>-source.<ext>` | the picture that went out, as Momoshop served it |
| `<picture>-<language>-translated.<ext>` | what the model returned, in its own format |

and one block of console per call, printed as the overlay's frame is built:

```
picture: sending https://img2.momoshop.com.tw/ecm/img/online/10/014/900/000/14473107_OL.jpg to gemini-2.5-flash-image for English -- what comes back is kept in translated_pictures and reported below
picture: gemini-2.5-flash-image answered in 11.8s with 1,783,221 bytes of image/png at 1024x1024, for 20,545 bytes at 320x320 of 14473107_OL
  15.7% of the picture differs from the one that went out
  model said: Here's the image with the Chinese text translated into English, maintaining the original layout and styling:
  kept: 14473107_OL-english-source.jpg and 14473107_OL-english-translated.png in ...\translated_pictures
```

Two numbers are there to be read. The sizes and the format are what the model does with
the picture it is given: this one took a 320x320 JPEG and answered with a 1024x1024 PNG,
so the overlay is showing a picture the step generated rather than one it looked up. The
percentage is the comparison between the two files -- the returned one scaled to the size
of the one that went out, differences past 8 of 255 counted, because re-encoding moves
every pixel a little without changing anything on the picture. It is the one thing about
the answer that can be checked without looking at it: a few per cent is the wording
replaced, and nothing at all is an answer that is not an edit.

There is a better picture to ask about, and it is not the one on a card. Every picture the
search payload carries is a 320x320 `_OL.jpg` -- the collage above included -- and
stripping `_OL` off that URL is a `404`, so the card's picture has no bigger sibling. The
product page the detail panel already fetches does have one: the same collage as its
640x640 `og:image`. Sent that one, the model translated the flavour tabs too (`MATCHA`,
`CHOCOLATE`, `STRAWBERRY`), where on the 320x320 one it left them alone and lettered a
legend of its own into the empty space at the bottom instead; the print on the packets
themselves is still Chinese either way, because that line is a dozen pixels tall on the
picture this app is handed. Pointing the button at `og:image` would mean the overlay
translating a different URL from the one it opened, so it is written down here rather than
done -- and the step is per-picture in `PICTURE_MODEL` if it is ever wanted.

### Without a key: the same picture read and re-lettered, with Gemma

`Gemma-4-31B-IT` does take a picture as *input*, which is the half of the job the Radeon
endpoint can do, and it is what `image_translated` falls back to when there is no
`GEMINI_KEY` -- which the panel says in as many words rather than leaving it to be
guessed. The frame with the button carries the note: `Gemma-4-31B-IT reads the Chinese
wording off this picture and the same picture is lettered again in English where that
wording was. No model here can draw a picture -- they all answer in text ...`, and while
the step runs the frame under the picture says one line, `Translating ...`, on both
paths: what the step does inside is the note beside the button, said before it is
pressed rather than as a paragraph to read while the picture is being made.

`parse_wording` reads the JSON out of the reply (it arrives inside a ``` fence as often
as not); the same model translates those lines; and `reletter_image` paints over each
block of wording with the colour sampled from the ring around it and letters the
translation in its place.

The drawing has to be defensive, because a box read off a picture is a guess and what
is drawn goes *on top of* the original:

- A box taller than a third of the picture, or taller than it is wide, is not a line of
  Latin text -- these pictures carry vertical Chinese as often as not -- and it is left
  alone rather than lettered sideways into the product.
- A box too small to letter (`IMAGE_MIN_BOX`, 40x12) is left alone too.
- Blocks rather than lines: boxes that overlap sideways and sit within a line's height
  of each other are merged into one block, which is painted once and lettered once. One
  label per line left the Chinese that a box did not quite cover showing at its edges,
  and on one picture the fill of the second block painted over the text of the first.
- A block grows to fit its translation -- a translation is longer than the Chinese more
  often than not -- but only inside the room between it and its neighbours, and only
  down to a type size of `IMAGE_MIN_FONT_RATIO` (2.2% of the picture's height, 12px
  floor), because a size nobody can read is not a translation.

The type comes from the system's own font (Arial, Segoe UI, DejaVu) or from Pillow's
built-in one where there is none, and the re-lettering is Pillow, which is already a
dependency of Gradio. Measured on 義美's four pictures (370 KB of JPEG in all): reading
the wording took 9-76s per picture, translating it 4-11s, and 5 to 11 lines were read
per picture. On the pictures that are horizontal text, every line was lettered (5 of
5, 5 of 5); the fourth is mostly vertical text and lettered 3 of 11. Nothing is dropped
silently: the overlay lists every line that was read, what it was translated to, and
how many of them went into the picture -- so a line left as it was is still readable in
the target language, in the list.

This is also why the app is a `gr.Blocks` app rather than a `gr.Interface` one: the
panel is a second `gr.HTML` output, written by a card click as well as by `search`,
and an Interface has room for only the one fn that produces its outputs. The search
event's progress animation is pinned to the cards with `show_progress_on=results_box`:
Gradio draws it on every output of an event unless it is told which output it belongs
on, so a search that also writes the panel would otherwise put a second
`Translating your query...` over the panel. Both HTML components keep
`apply_default_css=False` and their own inline styles, so the cards look the same as
before; the hover and focus cue that the old `<a href>` gave for free now comes from
the results component's `css_template`. `PAGE_TIMEOUT` (30s) covers every page this
app fetches.
