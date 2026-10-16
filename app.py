"""DigitVision - The Machine Perception Lab.

A Streamlit app: draw a digit, watch a trained CNN interpret it.
The machine-learning pipeline (preprocessing + model) is unchanged from the
original app. Only the interface and state handling were redesigned.
"""

import base64
import hashlib
import html
import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# 1. Constants and paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "digitvision_cnn.keras"

CANVAS_SIZE = 280        # drawing surface, pixels (same as the original)
STROKE_WIDTH = 15        # same as the original
INK_THRESHOLD = 30       # pixel > 30 counts as "digit" (same as the original)
FIT_SIZE = 20            # digit is fitted inside 20 x 20 (same as the original)
TARGET_SIZE = 28         # CNN input size
ACCENT_RGB = (255, 79, 31)
ASCII_RAMP = " .:-=+*#%@"


# --------------------------------------------------
# 2. Model loading
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the CNN...")
def load_model():
    """Load the trained CNN once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def get_model():
    """Return (model, error_message). Never raises."""
    try:
        return load_model(), None
    except Exception as exc:  # noqa: BLE001 - show any load problem to the user
        return None, str(exc)


# --------------------------------------------------
# 3. Image pipeline (same steps and numbers as the original)
# --------------------------------------------------

def preprocess(rgba):
    """Turn the canvas RGBA array into the CNN input.

    Returns None when no digit is found. Otherwise returns a dict with the
    model input and every intermediate image, so the UI can show them.
    """
    image = Image.fromarray(rgba.astype("uint8")).convert("L")

    mask = image.point(lambda pixel: 255 if pixel > INK_THRESHOLD else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None

    cropped = image.crop(bbox)

    fitted = cropped.copy()
    fitted.thumbnail((FIT_SIZE, FIT_SIZE), Image.Resampling.LANCZOS)

    processed = Image.new("L", (TARGET_SIZE, TARGET_SIZE), color=0)
    processed.paste(
        fitted,
        ((TARGET_SIZE - fitted.width) // 2, (TARGET_SIZE - fitted.height) // 2),
    )

    image_array = np.asarray(processed, dtype=np.float32) / 255.0
    model_input = image_array[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)

    overlay = image.convert("RGB")
    ImageDraw.Draw(overlay).rectangle(bbox, outline=ACCENT_RGB, width=3)

    return {
        "model_input": model_input,
        "image_array": image_array,
        "captured": overlay,
        "cropped": cropped,
        "fitted": fitted,
        "processed": processed,
    }


def run_inference(model, model_input):
    """Run the CNN once and return the 10 output values."""
    expected = (TARGET_SIZE, TARGET_SIZE, 1)
    actual = tuple(model.input_shape[1:])
    if actual != expected:
        raise ValueError(
            f"The model expects input {actual}, but DigitVision sends {expected}."
        )
    scores = np.asarray(model.predict(model_input, verbose=0)[0], dtype=np.float64)
    if scores.shape != (10,):
        raise ValueError(f"Expected 10 output values, got shape {scores.shape}.")
    return scores


def data_uri(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def ascii_map(image_array):
    """Show the real 28 x 28 values as a character map (darker = brighter pixel)."""
    levels = np.clip(np.rint(image_array * 9), 0, 9).astype(int)
    return "\n".join("".join(ASCII_RAMP[v] * 2 for v in row) for row in levels)


def fmt_score(value):
    if 0 < value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def drawing_hash(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return None
    digest = hashlib.blake2b(
        np.ascontiguousarray(canvas_result.image_data).tobytes(), digest_size=8
    )
    return digest.hexdigest()


def analyse(canvas_result, model, current_hash):
    """Run the full pipeline once and return a result dict for session_state."""
    empty_msg = "NO SIGNAL. The canvas is empty. Draw one digit, then press Recognize."

    if model is None:
        return {"status": "error", "message": "ERROR. The model is not loaded."}
    if canvas_result is None or canvas_result.image_data is None:
        return {"status": "empty", "message": empty_msg}

    try:
        prepared = preprocess(canvas_result.image_data)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"ERROR. Could not read the drawing: {exc}"}
    if prepared is None:
        return {"status": "empty", "message": empty_msg}

    try:
        with st.spinner("Running inference..."):
            scores = run_inference(model, prepared["model_input"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"ERROR. Prediction failed: {exc}"}

    top = int(np.argmax(scores))
    runner = [int(i) for i in np.argsort(scores)[::-1] if int(i) != top][0]

    # Only call the outputs "probabilities" if they behave like probabilities.
    prob_like = bool(
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and abs(float(scores.sum()) - 1.0) < 1e-3
    )

    cropped, fitted = prepared["cropped"], prepared["fitted"]
    return {
        "status": "ok",
        "hash": current_hash,
        "scores": [float(s) for s in scores],
        "top": top,
        "runner": runner,
        "prob_like": prob_like,
        "total": float(scores.sum()),
        "stages": [
            ("Captured", data_uri(prepared["captured"]),
             f"{CANVAS_SIZE} x {CANVAS_SIZE} px, digit box in orange"),
            ("Cropped", data_uri(cropped),
             f"{cropped.width} x {cropped.height} px, empty space removed"),
            ("Fitted", data_uri(fitted),
             f"{fitted.width} x {fitted.height} px, proportions kept"),
            ("CNN input", data_uri(prepared["processed"]),
             f"{TARGET_SIZE} x {TARGET_SIZE} px, centered, 0-1 values"),
        ],
        "ascii": ascii_map(prepared["image_array"]),
    }


# --------------------------------------------------
# 4. HTML builders (no indentation: Markdown would treat it as code)
# --------------------------------------------------

def status_html(message):
    return f'<div class="dv-status" role="status">{html.escape(message)}</div>'


def readout_html(reading, stale):
    scores = reading["scores"]
    top, runner = reading["top"], reading["runner"]
    kind = "Top output score" if not reading["prob_like"] else "Top softmax output"
    gap = scores[top] - scores[runner]
    classes = "dv-read dv-dim" if stale else "dv-read"
    return (
        f'<div class="{classes}">'
        f'<div class="dv-num" role="img" aria-label="Predicted digit {top}">{top}</div>'
        '<div class="dv-facts">'
        f'<div class="dv-fact"><span>{kind}</span><b>{fmt_score(scores[top])}</b></div>'
        f'<div class="dv-fact"><span>Runner-up</span><b>{runner} &middot; {fmt_score(scores[runner])}</b></div>'
        f'<div class="dv-fact"><span>Gap to runner-up</span><b>{fmt_score(gap)}</b></div>'
        '</div></div>'
    )


def bars_html(reading, by_score):
    scores = np.array(reading["scores"])
    top = reading["top"]
    lo, hi = float(scores.min()), float(scores.max())
    order = [int(i) for i in np.argsort(scores)[::-1]] if by_score else list(range(10))

    rows = []
    for digit in order:
        value = float(scores[digit])
        if reading["prob_like"]:
            width = value * 100
        else:
            width = 0.0 if hi == lo else (value - lo) / (hi - lo) * 100
        width = max(width, 0.6)
        win = digit == top
        row_class = "dv-row dv-win" if win else "dv-row"
        tag = '<span class="dv-tag">&#9664; predicted</span>' if win else "<span></span>"
        rows.append(
            f'<div class="{row_class}">'
            f'<span class="dv-d">{digit}</span>'
            f'<div class="dv-track"><div class="dv-fill" style="--w:{width:.2f}%"></div></div>'
            f'<span class="dv-v">{fmt_score(value)}</span>{tag}</div>'
        )
    return '<div class="dv-bars">' + "".join(rows) + "</div>"


def flow_html(reading):
    cells = []
    for index, (name, uri, note) in enumerate(reading["stages"]):
        cells.append(
            f'<div class="dv-stage" style="--i:{index}">'
            f'<div class="dv-frame"><img src="{uri}" alt="{name}: {html.escape(note)}"></div>'
            f'<div class="dv-cap"><b>0{index + 1} {name}</b><br>{html.escape(note)}</div>'
            '</div>'
        )
    return '<div class="dv-flow">' + "".join(cells) + "</div>"


# --------------------------------------------------
# 5. Styling
# --------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;1,9..144,500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
  --paper: #F2EEE6; --ink: #16140F; --gray: #5E5A51; --line: #CFC8B8;
  --accent: #FF4F1F; --bar: #8C877B; --panel: #E8E3D8;
  --mono: 'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
  --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
}
.stApp { background: var(--paper); color: var(--ink);
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1160px; padding: 2rem 1.25rem 3rem; }
.stApp p, .stApp label { color: var(--ink); }

/* Identity */
.dv-kicker { font: 500 .78rem var(--mono); color: var(--gray); letter-spacing: .06em; }
.dv-mark { font-family: var(--serif); font-weight: 600; letter-spacing: -.02em;
  font-size: clamp(2.2rem, 6vw, 3.4rem); line-height: 1; margin: .25rem 0 .35rem; }
.dv-mark em { font-style: italic; font-weight: 500; color: var(--accent); }
.dv-sub { color: var(--gray); margin: 0 0 1.2rem; max-width: 34rem; }
.dv-rule { border: 0; border-top: 2px solid var(--ink); margin: 0 0 1.4rem; }
.dv-label { font: 500 .74rem var(--mono); color: var(--gray); letter-spacing: .08em;
  text-transform: uppercase; margin: 0 0 .5rem; }

/* Canvas workspace */
.dv-ruler { display: flex; align-items: center; gap: .5rem; max-width: 280px;
  font: .72rem var(--mono); color: var(--gray); margin-bottom: .4rem; }
.dv-ruler i { flex: 1; height: 9px; border-left: 1px solid var(--ink);
  border-right: 1px solid var(--ink);
  background: linear-gradient(var(--ink), var(--ink)) center / 100% 1px no-repeat; }
iframe[title*="drawable_canvas"] { display: block; background: #000;
  border: 2px solid var(--ink); box-shadow: 8px 8px 0 var(--accent); }
.dv-hint { font-size: .9rem; color: var(--gray); margin: 1.1rem 0 .6rem; max-width: 30rem; }

/* Buttons */
.stButton > button { width: 100%; border-radius: 0; border: 2px solid var(--ink);
  font: 500 .92rem var(--mono); min-height: 2.9rem; background: transparent; }
.stButton > button p { color: inherit !important; margin: 0; }
.stButton > button[kind="primary"] { background: var(--accent); color: var(--ink); }
.stButton > button:hover { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.stButton > button:disabled { opacity: .45; }
.stButton > button:focus-visible { outline: 3px solid var(--ink); outline-offset: 3px; }

/* Interpretation */
.dv-read { display: flex; gap: 1.4rem; align-items: flex-end; border-top: 2px solid var(--ink);
  padding-top: .9rem; margin-bottom: 1.2rem; }
.dv-dim { opacity: .45; }
.dv-num { font-family: var(--serif); font-weight: 600; font-size: clamp(6rem, 16vw, 9rem);
  line-height: .85; padding: 0 .12em .06em; border-bottom: 10px solid var(--accent); }
.dv-facts { flex: 1; font: .85rem var(--mono); min-width: 0; }
.dv-fact { display: flex; justify-content: space-between; gap: 1rem;
  border-bottom: 1px solid var(--line); padding: .3rem 0; }
.dv-fact b { font-weight: 500; }
.dv-note { font-size: .8rem; color: var(--gray); margin: .6rem 0 0; max-width: 34rem; }
.dv-stale { font: 500 .82rem var(--mono); border: 2px solid var(--ink); padding: .4rem .6rem;
  margin-bottom: .8rem; }
.dv-status { border-left: 6px solid var(--accent); background: var(--panel);
  padding: .75rem .9rem; font: .9rem var(--mono); }
.dv-empty { display: flex; gap: 1.4rem; align-items: flex-end; border-top: 2px solid var(--ink);
  padding-top: .9rem; color: var(--gray); }
.dv-empty .dv-num { border-bottom-color: var(--line); color: var(--line); }

.dv-bars { margin-top: .4rem; }
.dv-row { display: grid; grid-template-columns: 1.6rem 1fr 4.4rem 6.6rem;
  align-items: center; gap: .5rem; padding: .2rem 0; font: .85rem var(--mono); }
.dv-d { font-weight: 500; }
.dv-v { text-align: right; }
.dv-track { height: .8rem; background: #DDD6C7; }
.dv-fill { height: 100%; width: var(--w); background: var(--bar);
  animation: dv-grow .5s ease-out both; }
.dv-win .dv-fill { background: var(--accent); box-shadow: 0 0 0 2px var(--ink); }
.dv-win .dv-d, .dv-win .dv-v { font-weight: 600; }
.dv-tag { font-size: .72rem; background: var(--ink); color: var(--paper); padding: .05rem .35rem;
  justify-self: start; }

/* Signal path */
.dv-flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.dv-stage { animation: dv-in .35s ease-out both; animation-delay: calc(var(--i) * 90ms); }
.dv-frame { aspect-ratio: 1 / 1; background: #000; border: 2px solid var(--ink);
  display: flex; align-items: center; justify-content: center; overflow: hidden; }
.dv-frame img { max-width: 100%; max-height: 100%; image-rendering: pixelated; }
.dv-cap { font: .75rem var(--mono); color: var(--gray); margin-top: .4rem; line-height: 1.4; }
.dv-cap b { font-weight: 500; color: var(--ink); }

.dv-foot { border-top: 1px solid var(--line); margin-top: 2rem; padding-top: .8rem;
  font: .76rem var(--mono); color: var(--gray); line-height: 1.6; }

@keyframes dv-grow { from { width: 0; } }
@keyframes dv-in { from { opacity: 0; transform: translateY(6px); } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
@media (max-width: 640px) {
  .dv-flow { grid-template-columns: repeat(2, 1fr); }
  .dv-row { grid-template-columns: 1.4rem 1fr 3.8rem 5.4rem; gap: .35rem; }
  .dv-read, .dv-empty { gap: .9rem; }
}
</style>
"""


# --------------------------------------------------
# 6. App
# --------------------------------------------------
# (Everything above is plain functions. Everything below draws the page.)

st.set_page_config(page_title="DigitVision", page_icon="◉", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.session_state.setdefault("nonce", 0)      # changing it gives the canvas a fresh key
st.session_state.setdefault("reading", None)  # last result (or notice)


def clear_canvas():
    """Callback: runs before the rerun, so the canvas is rebuilt with a new key."""
    st.session_state.nonce += 1
    st.session_state.reading = None


st.markdown(
    '<div class="dv-kicker">MACHINE PERCEPTION LAB</div>'
    '<div class="dv-mark" role="heading" aria-level="1">Digit<em>Vision</em></div>'
    '<p class="dv-sub">Draw a signal. See what the network receives, '
    'and how it scores each digit from 0 to 9.</p>'
    '<hr class="dv-rule">',
    unsafe_allow_html=True,
)

model, model_error = get_model()
if model_error:
    st.error(f"The CNN could not be loaded. {model_error}")

left, right = st.columns([5, 6], gap="large")

# ---- Left: the drawing workspace ----
with left:
    st.markdown('<div class="dv-label">01 &middot; Signal input</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dv-ruler"><span>0</span><i></i><span>{CANVAS_SIZE} px</span></div>',
        unsafe_allow_html=True,
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key=f"digit_canvas_{st.session_state.nonce}",
    )

    st.markdown(
        '<p class="dv-hint">Draw one digit, large, with a thick stroke. '
        'The app crops and centers it for you.</p>',
        unsafe_allow_html=True,
    )

    btn_recognize, btn_clear = st.columns(2)
    with btn_recognize:
        recognize = st.button("Recognize digit", type="primary", disabled=model is None)
    with btn_clear:
        st.button("Clear canvas", on_click=clear_canvas)

current_hash = drawing_hash(canvas_result)

# One place that runs inference: only when the button is clicked.
if recognize:
    st.session_state.reading = analyse(canvas_result, model, current_hash)

reading = st.session_state.reading

# ---- Right: the machine's interpretation ----
with right:
    st.markdown('<div class="dv-label">02 &middot; Interpretation</div>', unsafe_allow_html=True)

    if reading is None:
        st.markdown(
            '<div class="dv-empty"><div class="dv-num" aria-hidden="true">?</div>'
            '<p>Awaiting signal. Draw a digit, then press <b>Recognize digit</b>.</p></div>',
            unsafe_allow_html=True,
        )
    elif reading["status"] != "ok":
        st.markdown(status_html(reading["message"]), unsafe_allow_html=True)
    else:
        stale = reading["hash"] != current_hash
        if stale:
            st.markdown(
                '<div class="dv-stale" role="status">The drawing changed after this reading. '
                'Press Recognize digit for a new one.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(readout_html(reading, stale), unsafe_allow_html=True)

        order_choice = st.radio(
            "Order the ten classes by",
            ["Digit", "Score"],
            horizontal=True,
            key="order_choice",
        )
        if reading["prob_like"]:
            scale = f"Softmax-style outputs (they sum to {reading['total']:.3f})"
        else:
            scale = "Raw output scores (not probabilities). Bar length is relative."
        st.markdown(f'<div class="dv-label">{html.escape(scale)}</div>', unsafe_allow_html=True)
        st.markdown(bars_html(reading, order_choice == "Score"), unsafe_allow_html=True)
        st.markdown(
            '<p class="dv-note">The highest score is the model\'s choice. '
            'It is not a calibrated measure of certainty: odd drawings can still '
            'get a high score.</p>',
            unsafe_allow_html=True,
        )

# ---- Full width: what the CNN actually received ----
if reading is not None and reading["status"] == "ok":
    st.markdown(
        '<div class="dv-label" style="margin-top:1.8rem">03 &middot; Signal path: '
        'from your drawing to the CNN input</div>',
        unsafe_allow_html=True,
    )
    st.markdown(flow_html(reading), unsafe_allow_html=True)
    with st.expander("Read the raw 28 x 28 input values"):
        st.caption("Each pair of characters is one pixel. Darker characters mean brighter pixels.")
        st.code(reading["ascii"], language=None)

# ---- Footer ----
if model is not None:
    try:
        info = (
            f"input {tuple(model.input_shape)} &middot; output {tuple(model.output_shape)} "
            f"&middot; {model.count_params():,} parameters"
        )
    except Exception:  # noqa: BLE001
        info = "model loaded"
else:
    info = "model not loaded"

st.markdown(
    f'<div class="dv-foot">models/digitvision_cnn.keras &middot; {info}<br>'
    'Trained on MNIST. Predictions may be less reliable for drawings that differ '
    'from the training images.</div>',
    unsafe_allow_html=True,
)