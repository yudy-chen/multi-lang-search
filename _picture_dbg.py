"""Scratch: what the picture step actually gets back from the image model.

Standalone on purpose: `import main` pulls in gradio, which costs about nineteen seconds
here, and adding that to a search and an image call overruns the shell's window. Nothing
is imported from the app -- the model, the base URL and the prompt are copied from
`main.py` verbatim -- so a run is one page, one picture and one call.

Every step lands in `_picture_dbg/log.txt` as well as the console, because an image call
can outlive the window and the log is what survives.

Kept rather than deleted: `gen <variant> [source.bin]` is the only way to put a wording in
front of the image model without paying the app's 19s gradio import, which is what the
prompt in `main.py` was chosen with -- `plain` is the wording the app shipped first, and
`shipped` is the one it ships now, so the two can be run over the same picture any time the
translation is doubted. `fetch`, `items` and `og` are how the pictures themselves were
found: `fetch` for the pictures on a search page, `og` for the bigger one the product page
carries.
"""

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
OUT = HERE / "_picture_dbg"
OUT.mkdir(exist_ok=True)
LOG = OUT / "log.txt"

MODEL = "gemini-2.5-flash-image"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SEARCH_URL = "https://www.momoshop.com.tw/search/" + requests.utils.quote("義美")
PROMPT = (
    "Translate all Chinese text in the provided image into clear {language}. Retain the "
    "exact visual layout, product graphics, diagram elements, arrows, numbers, and "
    "background styling without modifying any physical object details."
)
# The wording the app shipped first, kept verbatim as the control: on a picture of a
# product it came back with the wording unchanged. The others are what the picture step
# could ask for instead, and `shipped` is what `main.py`'s `IMAGE_TRANSLATE_PROMPT` holds
# now -- kept here in the same words so a doubt about the translation can be settled
# against the same picture.
VARIANTS = {
    "plain": PROMPT,
    "shipped": (
        "Translate into {language} every piece of Chinese wording in this picture -- the "
        "wording in the badges, the wording printed on the product packets, and the small "
        "print -- and put the {language} wording where the Chinese was, in the same place, "
        "at the same size, in the same colour and slant, so that no Chinese character is "
        "left in the picture. The products themselves stay as they are: their shapes, "
        "colours, photographs, graphics, arrows and numbers are not to be changed, only "
        "the wording on them replaced."
    ),
    "replace": (
        "Replace every Chinese character written in this image with its {language} "
        "translation, drawn in the same place, at the same size, in the same colour, slant "
        "and style as the character it replaces. Keep the layout, the product photographs, "
        "the graphics and the numbers exactly as they are. Do not leave a single Chinese "
        "character in the picture, including the wording inside the round badge."
    ),
    "label": (
        "This is a Taiwanese product picture. Every piece of Chinese wording on it -- the "
        "round badge, the print on each box, the small print and the shelf labels -- is to "
        "be lettered again in {language}. Cover the Chinese with the {language} wording in "
        "the same spot, matching its size and colour, and change nothing else about the "
        "picture."
    ),
    "final": (
        "Every Chinese character written into this picture is to be lettered again in "
        "{language}: put the {language} wording where the Chinese was, in the same place, "
        "at the same size, in the same colour and the same style, and leave no Chinese "
        "character in the picture. Badges and small print count, not just the largest "
        "line. Keep the layout, the product photographs, the graphics, the arrows and the "
        "numbers as they are, and change nothing else on it."
    ),
    "packaging": (
        "Translate into {language} every piece of Chinese wording in this picture, "
        "including the wording printed on the product packets and the small print under "
        "the logos: letter the {language} wording over that Chinese, in the same place and "
        "at the same size, colour and slant, so that no Chinese character is left in the "
        "picture. The packets stay where they are with their shapes, colours, photographs "
        "and numbers -- only the wording on them is replaced."
    ),
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}
PICTURE_PATTERN = re.compile(
    r"https://img\d?\.momoshop\.com\.tw/[^\"'\\ ]+?\.(?:jpg|jpeg|png|webp)"
)


def say(*parts):
    line = " ".join(str(part) for part in parts)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def mime(content):
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"GIF8"):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def fetch():
    page = requests.get(
        "https://www.momoshop.com.tw/search/" + requests.utils.quote("義美"),
        headers=HEADERS,
        timeout=30.0,
    )
    say("search page:", page.status_code, f"{len(page.text):,} chars")
    urls = list(dict.fromkeys(PICTURE_PATTERN.findall(page.text)))
    (OUT / "urls.txt").write_text("\n".join(urls), encoding="utf-8")
    say("pictures on the page:", len(urls))
    for url in urls[:5]:
        say("  ", url)
    content = requests.get(urls[0], headers=HEADERS, timeout=30.0).content
    (OUT / "input.bin").write_bytes(content)
    say("input:", urls[0])
    say("input:", f"{len(content):,} bytes |", mime(content), "|", Image.open(BytesIO(content)).size)


def compare(content, output):
    before = Image.open(BytesIO(content)).convert("RGB")
    after = Image.open(output).convert("RGB")
    shape = ""
    if before.size != after.size:
        shape = f" (input {before.size} -> output {after.size}, output scaled to compare)"
        after = after.resize(before.size)
    difference = ImageChops.difference(before, after).convert("L")
    changed = sum(difference.histogram()[8:])
    total = before.size[0] * before.size[1]
    return f"{100 * changed / total:.1f}% of pixels changed{shape}"


def items():
    """Which picture each search card actually carries, and whether a bigger sibling exists."""
    page = requests.get(SEARCH_URL, headers=HEADERS, timeout=30.0)
    blocks = re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', page.text, re.DOTALL
    )
    say("ld+json blocks:", len(blocks))
    urls = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in (data.get("@graph") or []) if isinstance(data, dict) else []:
            if isinstance(node, dict) and node.get("@type") == "ItemList":
                for element in node.get("itemListElement") or []:
                    if isinstance(element, dict) and element.get("image"):
                        urls.append(str(element["image"]).split("?")[0])
    say("items carrying a picture:", len(urls))
    for url in urls[:3]:
        reply = requests.get(url, headers=HEADERS, timeout=30.0)
        say(f"   {len(reply.content):,} bytes |", Image.open(BytesIO(reply.content)).size, "|", url)
    bigger = re.sub(r"_OL(?=\.)", "", urls[0])
    check = requests.get(bigger, headers=HEADERS, timeout=30.0)
    say("same picture without _OL:", check.status_code, f"{len(check.content):,} bytes |", bigger)
    if check.status_code == 200 and check.content[:1] != b"<":
        (OUT / "bigger.bin").write_bytes(check.content)
        say("   shape:", Image.open(BytesIO(check.content)).size, "| saved bigger.bin")


def search_elements():
    """The ItemList entries of a search page: each product's picture, title and page URL."""
    page = requests.get(SEARCH_URL, headers=HEADERS, timeout=30.0)
    elements = []
    blocks = re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', page.text, re.DOTALL
    )
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in (data.get("@graph") or []) if isinstance(data, dict) else []:
            if isinstance(node, dict) and node.get("@type") == "ItemList":
                elements.extend(
                    element
                    for element in node.get("itemListElement") or []
                    if isinstance(element, dict)
                )
    return elements


def og():
    """The picture the product page itself carries: much bigger than a card's thumbnail."""
    element = search_elements()[0]
    page = requests.get(str(element["url"]), headers=HEADERS, timeout=30.0)
    say("product page:", page.status_code, f"{len(page.text):,} chars |", element["url"])
    match = re.search(r'property="og:image"[^>]*content="([^"]+)"', page.text) or re.search(
        r'content="([^"]+)"[^>]*property="og:image"', page.text
    )
    if not match:
        say("no og:image on the page")
        return
    url = match.group(1)
    say("og:image:", url.split("?")[0])
    picture = requests.get(url, headers=HEADERS, timeout=30.0).content
    (OUT / "og.bin").write_bytes(picture)
    say(
        "og:image:",
        f"{len(picture):,} bytes as {mime(picture)} |",
        Image.open(BytesIO(picture)).size,
        "| saved og.bin",
    )


def gen():
    load_dotenv(HERE / ".env")
    key = (os.getenv("GEMINI_KEY") or "").strip()
    variant = sys.argv[2] if len(sys.argv) > 2 else "plain"
    source = OUT / (sys.argv[3] if len(sys.argv) > 3 else "input.bin")
    if not source.exists():
        url = (OUT / "urls.txt").read_text(encoding="utf-8").splitlines()[0]
        source.write_bytes(requests.get(url, headers=HEADERS, timeout=30.0).content)
    content = source.read_bytes()
    prompt = VARIANTS[variant].format(language="English")
    say("-" * 72)
    say("variant:", variant, "| source:", source.name, f"{len(content):,} bytes as {mime(content)}")
    say("model:", MODEL, "| key chars:", len(key))
    say("prompt:", prompt)
    started = time.time()
    call = requests.post(
        f"{BASE_URL}/models/{MODEL}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime(content),
                                "data": base64.b64encode(content).decode(),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        },
        timeout=240.0,
    )
    say("status:", call.status_code, "after", f"{time.time() - started:.1f}s")
    if call.status_code != 200:
        say("body:", call.text[:600])
        return

    reply = call.json()
    candidate = (reply.get("candidates") or [{}])[0]
    (OUT / f"{variant}.json").write_text(
        json.dumps(reply, ensure_ascii=False)[:40000], encoding="utf-8"
    )
    say("finishReason:", candidate.get("finishReason"))
    say("usage:", json.dumps(reply.get("usageMetadata") or {}))
    for number, part in enumerate((candidate.get("content") or {}).get("parts") or [], 1):
        inline = part.get("inlineData") or part.get("inline_data") or {}
        if inline.get("data"):
            picture = base64.b64decode(inline["data"])
            kind = str(inline.get("mimeType") or inline.get("mime_type"))
            name = f"{variant}.png" if "png" in kind else f"{variant}.jpg"
            (OUT / name).write_bytes(picture)
            shape = Image.open(BytesIO(picture)).size
            say(f"part {number}: PICTURE {len(picture):,} bytes | {kind} | {shape} | saved {name}")
            say("input vs output:", compare(content, OUT / name))
        else:
            say(f"part {number}: text:", repr(str(part.get("text"))[:400]))


if __name__ == "__main__":
    {"fetch": fetch, "items": items, "og": og}.get(sys.argv[1], gen)()
