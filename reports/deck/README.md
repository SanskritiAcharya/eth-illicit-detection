# The deck

`eth-deck.html` is the source of record — a self-contained 17-slide deck.
Arrow keys or the on-screen arrows move between slides.

| file | what it is |
|---|---|
| `eth-deck.html` | the deck itself; edit this |
| `eth-deck.pdf` | export for submitting and presenting |
| `eth-deck.pptx` | for editing in Canva or PowerPoint |
| `build_pptx.py` | regenerates `eth-deck.pptx` |

## Editing it in Canva

Use **`eth-deck.pptx`**. Every headline, label, bar and caption is a real
PowerPoint text box or shape — nothing is a flattened image — so it stays
editable after import. The two fonts, **Noto Serif** and **IBM Plex Mono**,
are both in Canva's free font library, so they map by name instead of being
silently substituted.

Do **not** import `eth-deck.pdf` into Canva. Headless Chrome writes web fonts
into a PDF as Type 3 fonts — glyph drawing programs with no font name to match
— so Canva substitutes a default face and the spacing shifts. That is a
property of the PDF export, not of Canva.

Canva also imports HTML directly, and documents that route as producing
editable text boxes, so `eth-deck.html` is a reasonable second option.

## Regenerating

    # PDF, from the HTML
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
      --no-pdf-header-footer --virtual-time-budget=8000 \
      --print-to-pdf=reports/deck/eth-deck.pdf file://$PWD/reports/deck/eth-deck.html

    # PPTX
    pip install python-pptx && python reports/deck/build_pptx.py reports/deck/eth-deck.pptx

`build_pptx.py` holds the slide content directly; the two font names are
constants at the top, so swapping faces is a one-line change.
