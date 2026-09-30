"""Generate a vector, paper-style diagram of the current v4 JEPA-WAM-TTT routes.

No third-party Python packages are needed. Run from any directory:
    python3 figures/generate_ttt_architecture.py
"""

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("jepa_wam_ttt_architecture.svg")
W, H = 2400, 1500

NAVY = "#19283D"
SLATE = "#56677C"
BORDER = "#C8D4DF"
TEAL = "#087F83"
TEAL_PALE = "#EAF7F5"
INDIGO = "#5456A9"
INDIGO_PALE = "#F0F1FC"
GOLD = "#B97924"
GOLD_PALE = "#FFF5E7"
GRAY_PALE = "#F7F9FB"

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
    '<title id="title">LeVJEPA-WAM-TTT model architecture</title>',
    '<desc id="desc">End-to-end JEPA-WAM policy, selected DiT block TTT memory, four routes, and temporal segment training.</desc>',
    '<defs>',
    '<marker id="arrow-navy" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1 1 L 11 6 L 1 11" fill="none" stroke="#19283D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>',
    '<marker id="arrow-teal" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1 1 L 11 6 L 1 11" fill="none" stroke="#087F83" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>',
    '<marker id="arrow-indigo" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1 1 L 11 6 L 1 11" fill="none" stroke="#5456A9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>',
    '<marker id="arrow-gold" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" orient="auto"><path d="M 1 1 L 11 6 L 1 11" fill="none" stroke="#B97924" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>',
    '</defs>',
    '<rect width="2400" height="1500" fill="#FFFFFF"/>',
]


def rect(x, y, w, h, fill="#FFFFFF", stroke=BORDER, radius=18, width=2.2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def text(x, y, value, size=28, color=NAVY, weight=500, anchor="start", letter=None):
    letter_attr = f' letter-spacing="{letter}"' if letter is not None else ""
    parts.append(
        f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{letter_attr}>'
        f'{escape(value)}</text>'
    )


def lines(x, y, values, size=26, color=NAVY, weight=500, leading=33, anchor="middle"):
    for index, value in enumerate(values):
        text(x, y + index * leading, value, size, color, weight, anchor)


def arrow(points, color=NAVY, width=3.3, dash=None, marker=True):
    path = " ".join(("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(points))
    marker_attr = f' marker-end="url(#arrow-{color})"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<path d="{path}" fill="none" stroke="{dict(navy=NAVY, teal=TEAL, indigo=INDIGO, gold=GOLD)[color]}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker_attr}/>'
    )


def card(x, y, w, h, title, subtitle=(), kind="neutral", title_size=28, body_size=23):
    style = {
        "neutral": ("#FFFFFF", BORDER, NAVY),
        "teal": (TEAL_PALE, TEAL, TEAL),
        "indigo": (INDIGO_PALE, INDIGO, INDIGO),
        "gold": (GOLD_PALE, GOLD, GOLD),
        "gray": (GRAY_PALE, BORDER, NAVY),
    }[kind]
    rect(x, y, w, h, style[0], style[1], radius=17, width=2.4)
    text(x + w / 2, y + (h / 2 - 3 if not subtitle else 43), title, title_size, style[2], 700, "middle")
    if subtitle:
        lines(x + w / 2, y + 75, subtitle, body_size, SLATE, 500, 29)


def panel(x, y, w, h, letter, title):
    rect(x, y, w, h, "#FFFFFF", BORDER, radius=23, width=2.4)
    rect(x + 27, y + 25, 54, 54, NAVY, NAVY, radius=13, width=0)
    text(x + 54, y + 62, letter, 30, "#FFFFFF", 700, "middle")
    text(x + 101, y + 62, title, 35, NAVY, 700)


text(60, 61, "LeVJEPA-WAM-TTT", 49, NAVY, 750)
text(60, 105, "JEPA-conditioned fast-weight memory inside a GR00T-style flow policy", 28, SLATE, 500)
rect(1880, 48, 460, 49, TEAL_PALE, TEAL, radius=23, width=2)
text(2110, 81, "Current v4 / four routes", 25, TEAL, 700, "middle")

# A — End-to-end policy and the visually supervised base model.
panel(60, 135, 2280, 645, "A", "Policy and representation pathways")
card(100, 254, 250, 108, "Current views", ("primary + wrist",), "teal")
card(430, 254, 270, 108, "V-JEPA 2.1", ("frozen ViT-L",), "teal")
card(780, 254, 245, 108, "Visual bridge", ("projector",), "teal")
card(1105, 245, 325, 250, "Qvv2.5-0.5B", ("causal multimodal", "transformer + LoRA", "visual / action states"), "neutral", 30, 23)
card(1525, 245, 310, 108, "WAM head", ("visual states → JEPA",), "teal")
card(1925, 245, 320, 108, "Predicted JEPA", ("Y-hat; available at test",), "teal")
card(100, 425, 250, 83, "Instruction", (), "gray", 27)
card(1525, 397, 310, 105, "Action states", ("placeholder positions",), "indigo")
card(1925, 418, 320, 200, "Flow-GR00T DiT", ("16 transformer blocks", "cross-attend to action states", "TTT at selected blocks"), "indigo", 30, 23)
card(100, 594, 360, 98, "Proprio + noisy actions", ("state, flow time, action tokens",), "indigo", 27, 21)
card(1937, 655, 300, 91, "20-step action chunk", ("4 flow evaluations",), "indigo", 24, 19)

arrow([(350, 308), (430, 308)], "teal")
arrow([(700, 308), (780, 308)], "teal")
arrow([(1025, 308), (1105, 308)], "teal")
arrow([(1430, 303), (1525, 303)], "teal")
arrow([(1835, 303), (1925, 303)], "teal")
arrow([(350, 466), (725, 466), (725, 421), (1105, 421)], "navy")
arrow([(1430, 445), (1525, 445)], "indigo")
arrow([(1835, 450), (1925, 450)], "indigo")
arrow([(2085, 353), (2085, 418)], "teal")
text(2104, 387, "K/V (explicit)", 20, TEAL, 600)
arrow([(460, 644), (505, 644), (505, 711), (1870, 711), (1870, 565), (1925, 565)], "indigo")
arrow([(2085, 618), (2085, 655)], "indigo")

# A distinct dotted visual supervision branch. It is not used in TTT-only training.
rect(550, 531, 1180, 124, GOLD_PALE, GOLD, radius=18, width=2, dash="10 8")
text(579, 563, "Base JEPA-WAM pretraining only", 24, GOLD, 700)
text(583, 605, "paired views (t, t+31)  →  frozen V-JEPA target  →  cosine alignment", 25, NAVY, 550)
text(583, 635, "Detached target; TTT-only training uses the current-view prediction above.", 21, SLATE, 500)
arrow([(1848, 303), (1873, 303), (1873, 516), (1720, 516), (1720, 531)], "gold", width=2.5, dash="8 7")

# B — Selected DiT block, fast-weight write/read and token selection.
panel(60, 816, 1410, 414, "B", "TTT inside a selected DiT block")
card(97, 906, 245, 96, "Post-attention", ("hidden states",), "gray", 26, 20)
card(433, 906, 240, 96, "Select Q", ("S + R + A",), "indigo", 27, 22)
card(767, 895, 295, 119, "Fast-weight TTT", ("MLP memory write/read",), "teal", 29, 22)
card(1151, 906, 248, 96, "FFN", ("residual output",), "gray", 29, 21)
arrow([(342, 954), (433, 954)], "navy")
arrow([(673, 954), (767, 954)], "indigo")
arrow([(1062, 954), (1151, 954)], "teal")
rect(96, 1028, 1289, 176, GRAY_PALE, BORDER, radius=15, width=1.8)
text(126, 1064, "DiT stream", 23, SLATE, 700)
text(330, 1064, "[ S  |  R × 16  |  F × 32  |  A × 20 ]", 27, NAVY, 700)
text(126, 1102, "Q + direct residual:", 24, INDIGO, 700)
text(394, 1102, "S = state, R = registers, A = action tokens", 23, NAVY, 500)
text(126, 1140, "K/V:", 24, TEAL, 700)
text(218, 1140, "predicted JEPA Y-hat  /  full post-attention DiT stream", 23, NAVY, 500)
text(1073, 1140, "F: no direct residual", 20, SLATE, 600)
text(126, 1178, "Fast update:", 23, TEAL, 700)
text(286, 1178, "delta ← delta − eta · forget · grad MSE(f(K), V);  read f(Q)", 22, NAVY, 500)
arrow([(915, 1028), (915, 1014)], "teal", width=2.5)

# C — Four experimental routes, density by source.
panel(1510, 816, 830, 414, "C", "Four TTT routes")
text(1758, 901, "Explicit JEPA K/V", 24, TEAL, 700, "middle")
text(2113, 901, "Implicit DiT K/V", 24, INDIGO, 700, "middle")
text(1543, 995, "Wrapper", 23, NAVY, 700)
text(1543, 1136, "Inline", 23, NAVY, 700)
card(1657, 919, 258, 124, "4 blocks", ("indices 3, 7, 11, 15", "WAM-predicted K/V"), "teal", 25, 19)
card(1982, 919, 302, 124, "4 blocks", ("indices 3, 7, 11, 15", "full DiT K/V"), "indigo", 25, 19)
card(1657, 1061, 258, 124, "16 blocks", ("all layers", "WAM-predicted K/V"), "teal", 25, 19)
card(1982, 1061, 302, 124, "16 blocks", ("all layers", "full DiT K/V"), "indigo", 25, 19)

# D — Actual TTT-only temporal training schedule.
panel(60, 1261, 2280, 192, "D", "TTT-only training over one 32-observation sequence")
text(97, 1370, "Frozen base policy", 26, SLATE, 700)
text(97, 1407, "Train TTT slow weights + registers", 22, SLATE, 500)
for x, label in ((520, "1–8"), (870, "9–16"), (1220, "17–24"), (1570, "25–32")):
    card(x, 1341, 259, 71, label, (), "teal", 27)
for x in (779, 1129, 1479):
    arrow([(x + 8, 1376), (x + 78, 1376)], "teal", width=2.7)
text(819, 1440, "carry + detach", 19, TEAL, 600, "middle")
text(1169, 1440, "carry + detach", 19, TEAL, 600, "middle")
text(1519, 1440, "carry + detach", 19, TEAL, 600, "middle")
arrow([(1829, 1376), (1919, 1376)], "navy", width=2.7)
card(1920, 1334, 350, 85, "1 optimizer step", ("4 gradients accumulated",), "indigo", 25, 19)

parts.append('</svg>')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(OUT)
