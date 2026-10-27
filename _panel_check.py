"""Scratch: what the panel shows before, during and after its translation.

Everything the detail panel puts on the screen for one click is rendered here from the
real `detail_html` / `open_detail` / `panel_action` / `search`, and checked for the shape
of what the reader sees: the page at once, with exactly one layer over it washed to half
opacity saying it is translating, nothing but the translation left when that is over --
brand line included -- and no panel left open by a search. The page is a made-up one
with the same shape `product_detail` returns and the endpoint is stubbed out, so this
needs no network and no key -- `python _panel_check.py` and the last line says whether
the frames are right.
"""

import types

import main

DETAIL = {
    "url": "https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=8426761",
    "page": "<html><body>商品特色 原料 營養標示</body></html>",
    "payload_chars": 1234,
    "blocks": [
        ("title", "商品特色"),
        ("format", ["100% 防偽驗證", "蛋白質 20g", "支鏈氨基酸"]),
        ("text", "飲用方式：每日一包"),
    ],
    "feature": {
        "headline": "日本原裝進口",
        "iframe": "",
        "url": "",
        "text": "",
        "images": [],
    },
    "meta": {
        "og:title": "三多 維他命C 發泡錠",
        "description": "維他命C 1000mg 發泡錠",
        "product:brand": "三多",
        "product:price:amount": "599",
        "product:price:currency": "TWD",
        "product:availability": "in stock",
        "product:retailer_item_id": "8426761",
    },
    "image": "",
    "code": "8426761",
    "info_text": "",
}
TITLE = "Suntory Vitamin C Effervescent Tablets"
PAIRS = main.detail_texts(DETAIL)
ALL = {key: f"translated line {number}" for number, (key, _) in enumerate(PAIRS)}
HALF = {key: ALL[key] for key, _ in PAIRS[: len(PAIRS) // 2]}

# The two halves of the layer: the wash that dims the page, and the label on top of it.
# Both are checked by their own style, so no frame can pass on a lookalike.
VEIL = main.DETAIL_VEIL_STYLE
PILL = main.DETAIL_VEIL_PILL_STYLE
LABEL = "Translating ... into English"

failures = []
# The scraped lines as they are on the page. The section header is not one of them --
# it is the panel's own label for the section, and it glosses itself in English.
SOURCES = [
    text for _, text in PAIRS if main.has_chinese(text) and text != "商品特色"
]
# The brand is a `<meta>` value, but it is the page's words all the same: it used to be
# the one Chinese line left on a finished panel.
BRAND = DETAIL["meta"]["product:brand"]
BRAND_KEY = "meta:product:brand"


def check(name, frame, expected, forbidden=()):
    for text in forbidden:
        if text in frame:
            failures.append(f"{name}: still shows {text!r}")
    for text in expected:
        if text not in frame:
            failures.append(f"{name}: missing {text!r}")
    if "原文" in frame:
        failures.append(f"{name}: still has an 原文 fold")
    print(f"{name}: {len(frame):,} characters", flush=True)


def veils(name, frame):
    """A translating frame wears the layer once, over the page, and says what it is."""
    if frame.count(PILL) != 1:
        failures.append(f"{name}: {frame.count(PILL)} translating layer(s), want 1")
    if VEIL not in frame:
        failures.append(f"{name}: the page is not washed out while it translates")
    if LABEL not in frame:
        failures.append(f"{name}: the layer does not say it is translating")


before = main.detail_html(
    DETAIL, TITLE, translations={}, language="English", translating=True
)
check(
    "the click, before any translation",
    before,
    [TITLE, f"brand: {BRAND}"] + SOURCES,
)
veils("the click, before any translation", before)

half = main.detail_html(
    DETAIL, TITLE, translations=HALF, language="English", translating=True
)
check(
    "half the lines translated",
    half,
    [f"({len(HALF)} of {len(PAIRS)} line(s))", ALL[PAIRS[0][0]]]
    + [text for _, text in PAIRS[len(PAIRS) // 2 :] if main.has_chinese(text)],
)
veils("half the lines translated", half)

done = main.detail_html(
    DETAIL, TITLE, translations=ALL, language="English", translating=False
)
check(
    "all of it translated",
    done,
    [ALL[key] for key, _ in PAIRS]
    + ["translated into English", f"brand: {ALL[BRAND_KEY]}"],
    forbidden=SOURCES,
)
if f"brand: {ALL[BRAND_KEY]}" not in done or BRAND in done:
    failures.append("the brand line is still in Chinese on the finished panel")
if VEIL in done or PILL in done:
    failures.append("the finished panel still has the translating layer on it")

scraped = main.detail_html(DETAIL, TITLE, translations={}, language="", translating=False)
check(
    "a Chinese search, shown as scraped",
    scraped,
    ["防偽", "shown as Momoshop wrote it"],
)
if VEIL in scraped:
    failures.append("a Chinese search is washed out for a translation that never runs")

# The click itself: the frames `open_detail` streams, with the page and the endpoint
# stubbed out so the run is about the frames rather than about the network.
fetched = []


def fake_detail(url):
    fetched.append(url)
    return dict(DETAIL)


def fake_translate(detail, language):
    yield dict(HALF)
    yield dict(ALL)


main.cached_detail = fake_detail
main.translate_detail = fake_translate
request = types.SimpleNamespace(session_hash="panel-check")
event = types.SimpleNamespace(_data={"url": DETAIL["url"], "title": TITLE})
run = main.open_detail("English", event, request)
first = next(run)
if fetched:
    failures.append("the first frame of a click came after the page was fetched")
frames = [first] + list(run)
print(f"a click: {len(frames)} frames", flush=True)

if LABEL not in first or TITLE not in first:
    failures.append("the first frame of a click is not the waiting one")
for number, frame in enumerate(frames):
    if "原文" in frame:
        failures.append(f"frame {number} of a click still has an 原文 fold")
    if number < len(frames) - 1:
        veils(f"frame {number} of a click", frame)
    elif VEIL in frame or PILL in frame:
        failures.append("the last frame of a click still wears the layer")
for text in SOURCES:
    if text in frames[-1]:
        failures.append(f"the last frame of a click still shows {text!r}")
if ALL[PAIRS[-1][0]] not in frames[-1]:
    failures.append("the last frame does not carry the whole translation")

# The picture step: one line while it runs, the same whichever path does the work.
reading = main.lightbox_html({"url": DETAIL["url"], "stage": "reading"}, "English")
check(
    "a picture being translated",
    reading,
    ["Translating ..."],
    forbidden=["sending the picture to", "takes a moment", "reading the wording off"],
)

main.picture_result = lambda url, language: {
    "url": url,
    "stage": "done",
    "result": {"model": main.PICTURE_MODEL, "picture": url},
}
picture_event = types.SimpleNamespace(
    _data={"action": "image-translate", "url": DETAIL["url"]}
)
picture_frames = list(main.panel_action(picture_event, request))
check(
    "the frame the picture step starts on",
    picture_frames[0],
    ["Translating ..."],
)

# A search puts the panel away. The product that was open belongs to the results being
# replaced, so the panel goes back to its hint with every frame, and the click that was
# translating it can no longer write to it.
generation = main.panel_held("panel-check")["generation"]
main.search_momo = lambda keyword: "<html></html>"
main.extract_items = lambda page: [
    {"url": DETAIL["url"], "title": TITLE, "image": None}
]
main.render_results = lambda keyword, language, items, rows: "<div>cards</div>"
searched = list(main.search("維他命", "Chinese", request))
if not searched:
    failures.append("a search streamed no frame")
if searched[0][0] != main.gr.skip():
    failures.append("the frame that puts the panel away replaced the cards as well")
for number, (cards, panel) in enumerate(searched):
    if panel != main.DETAIL_HINT:
        failures.append(f"frame {number} of a search left the panel open")
if main.panel_held("panel-check") is not None:
    failures.append("a search left the panel held on the server")
if main.panel_update("panel-check", generation, translating=True) is not None:
    failures.append("a click from before the search can still write to the panel")
check(
    "the panel a search put away",
    searched[0][1],
    ["Click a product to read its page here."],
)
print(f"a search while a panel was open: {len(searched)} frames", flush=True)

# The progress animation belongs on the cards. Gradio draws it over every output of an
# event unless it is told which one it belongs on, so a search that writes the panel too
# would otherwise put a second `Translating your query...` over the panel.
for fn in main.demo.fns.values():
    written = list(fn.outputs)
    if main.detail_box not in written or len(written) == 1:
        continue
    if list(fn.show_progress_on or []) != [main.results_box]:
        failures.append(
            f"the {fn.api_name} event draws its progress over more than the cards"
        )
main._PANEL_STATES.clear()

if failures:
    print("\nFAILED:", flush=True)
    for failure in failures:
        print(f"  {failure}", flush=True)
    raise SystemExit(1)
print(
    "\nthe panel, the search that puts it away and the picture step are as they should be",
    flush=True,
)

