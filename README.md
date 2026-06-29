# DACCAS

**D**ark-web **A**daptive **C**APTCHA **C**lassification **A**nd **S**olving.

A two-step pipeline: a YOLOv8 **classifier** acts as a **dispatcher** that routes
an input CAPTCHA image to the matching module in a **solver library**. Each
solver is an independently loadable module behind a common interface, so the
library is additive — a new CAPTCHA type is supported by registering one new
solver, with no change to the dispatcher or the existing solvers and no
retraining of anything else.

```python
from daccas import DACCAS

# Build once at startup. preload=True loads every model now (see §7) so the
# first real Solve() is instant instead of paying first-use loading cost.
daccas = DACCAS(models_dir="models", charsets_dir="charsets", device="cpu",
                preload=True)

cls = daccas.Classify("captcha.png")          # -> {'class': 'Open Circle', 'confidence': 0.97, ...}
out = daccas.Solve("captcha.png", cls["class"])   # -> structured solver result
# or do both at once:
out = daccas.run("captcha.png")               # -> {'classification': ..., 'result': ...}
```

---

## 1. Directory layout

Place the files exactly like this. The weight/charset **filenames matter** — the
defaults below are what `DACCAS` looks for.

```
DACCAS/
├── daccas/                          # the package (provided)
│   ├── __init__.py
│   ├── daccas.py                    # DACCAS: Classify() + Solve()
│   ├── classifier.py                # DACCASClassifier (the dispatcher)
│   ├── orientation_lr.py            # ConvNeXt-L extractor + geometry (rotation-default)
│   ├── solvers/
│   │   ├── base.py                  # BaseSolver interface
│   │   ├── open_circle.py
│   │   ├── image_rotation_special.py
│   │   ├── image_rotation_default.py
│   │   ├── moving_window.py
│   │   └── text_solver.py
│   └── textmodel/                   # vendored TDA text-model code
│       ├── model.py                 # CaptchaTDAModel
│       ├── resnet_tda.py
│       ├── tda.py
│       └── dataset.py               # CharsetMapper
│
├── models/                          # <-- YOU put the weights here
│   ├── CLASSIFICATION_MODEL.pt              # YOLOv8-cls dispatcher
│   ├── OPEN_CIRCLE_BEST_MODEL.pt            # YOLOv8 detector
│   ├── IMAGE_ROTATION_DEFAULT_MODEL.joblib  # sklearn LR pipeline
│   ├── IMAGE_ROTATION_DEFAULT_META.json     # crop_mode, best_C, classes, ...
│   ├── MOVING_WINDOW_MODEL.pth              # ResNet-34, 32 classes
│   ├── GREGWAR_BEST_MODEL.pth              # TDA, charset_36
│   ├── MOBICMS_BEST_MODEL.pth              # TDA, charset_36
│   ├── KING_FINETUNE_BEST_MODEL.pth        # TDA, charset_36
│   └── GENERAL_BEST_MODEL.pth              # TDA, charset_62
│
├── charsets/
│   ├── charset_36.txt                       # a-z 0-9  (text: Gregwar/Mobicms/King)
│   ├── charset_62.txt                       # a-z A-Z 0-9  (text: Other/General)
│   └── moving_window_charset.txt            # <-- generate this (see §4)
│
├── build_moving_window_charset.py   # helper to make moving_window_charset.txt
├── example.py
└── requirements.txt
```

The package finds weights by name under `models_dir` and charsets under
`charsets_dir`. To use different paths/names, pass them to the solver classes
directly (they all take explicit `*_path` arguments).

---

## 2. Install

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If `Activate.ps1` is blocked ("running scripts is disabled"), either run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once and
retry, or use `.venv\Scripts\activate.bat` instead. You should see `(.venv)` in
your prompt once active. PowerShell note: it does not use `&&` to chain commands
or `\` to continue lines — run each line separately, or use a backtick `` ` ``
for line continuation.

**Windows (PyCharm, no terminal needed):**
File → Settings → Project → Python Interpreter → Add Interpreter →
Add Local Interpreter → Virtualenv (New). PyCharm then offers to install the
packages from `requirements.txt` via the editor banner.

**macOS / Linux:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is a large download (1 GB+), so the install step will sit for a while —
it is not frozen. GPU is optional: pass `device="cuda"` to `DACCAS(...)` if
available; everything defaults to CPU.

---

## 3. The pipeline

### `DACCAS.Classify(image)`
Runs `CLASSIFICATION_MODEL.pt` (YOLOv8-cls, `imgsz=224`). Returns
`{'class', 'confidence', 'raw_name'}`. The model emits filesystem-safe names
(e.g. `text_gregwar`); these are mapped to canonical labels (e.g.
`Text (Gregwar)`).

The nine classes:
`Text (Gregwar)`, `Text (Mobicms)`, `Text (King)`, `Text (Other)`,
`Moving Window`, `Open Circle`, `Image Rotation (Default)`,
`Image Rotation (Special)`, `No Solver`.

### `DACCAS.Solve(image, captcha_class)`
Routes the image to the solver for that class:

| Class | Solver | Returns |
|---|---|---|
| **No Solver** | — | `{solved: False, message: ...}` (deliberate decline) |
| **Open Circle** | YOLOv8 detect (`imgsz=480`, `conf=0.25`) | `x, y` = centre of the most-confident box (`all_boxes` also returned) |
| **Image Rotation (Special)** | seam stitch (close black gap) → Sobel seam score (`SEAM_R=40`, `delta=5`) | `sobel_score` (`objective: "minimize"`) — no weights needed |
| **Image Rotation (Default)** | ConvNeXt-L feature → LR (`crop_mode=circular`, `C=10`) | `p_upright` (`objective: "maximize"`) |
| **Moving Window** | ResNet-34 (contrast 1.25 → 224² → ImageNet norm) | `character`, `class_index`, `confidence` |
| **Text (Gregwar/Mobicms/King)** | TDA transformer + `charset_36` | `text` |
| **Text (Other)** | TDA transformer + `charset_62` | `text` |

The two rotation solvers return a per-image score plus an `objective` field
(`minimize` for Sobel, `maximize` for P(upright)). This is intentional: the
planned multi-variant protocol runs `Solve` once per rotated variant and then
picks the lowest Sobel score / highest P(upright). The current single-image
implementation already exposes exactly what that comparison needs.

---

## 4. Moving Window charset (required for character output)

The ResNet-34 outputs 32 class indices. In training the mapping was
`sorted(df["label"].unique())`, so index *i* = the *i*-th sorted unique label.
That ordering lives in the label CSVs, **not** in the weights, so you must
regenerate it once.

**Windows (PowerShell)** — one line; quote any path containing spaces:
```powershell
python build_moving_window_charset.py --csv "data/darkmarketreloaded_labels.csv" "data/addyrus_labels.csv" --label_col label --out charsets/moving_window_charset.txt
```
(PowerShell continues lines with a backtick `` ` ``, not `\`. Keeping it on one
line avoids that entirely.)

**macOS / Linux:**
```bash
python build_moving_window_charset.py \
    --csv data/darkmarketreloaded_labels.csv data/addyrus_labels.csv \
    --label_col label \
    --out charsets/moving_window_charset.txt
```

This writes one character per line in class-index order. If the file is absent,
the solver still runs but returns the **class index** as the character and sets
`charset_provided: False`.

---

## 5. Image Rotation (Default) — ConvNeXt-L weights

This solver extracts features with a frozen **ConvNeXt-L (ImageNet)** backbone.
By default torchvision downloads those weights from `download.pytorch.org`. If
your environment is offline, download `convnext_large` weights once and pass:

```python
DACCAS(..., convnext_weights_path="models/convnext_large.pth")
```

The `.joblib` itself (StandardScaler + one-vs-rest LogisticRegression, `C=10`)
was trained with scikit-learn 1.7.2. It loads under newer 1.x with a harmless
`InconsistentVersionWarning`, which the solver now silences; pin
`scikit-learn==1.7.2` for exact, warning-free reproduction.

---

## 6. Image Rotation (Special) — seam preprocessing

The raw 100×100 captcha has a black gap ring between the inner disc (radius 40)
and the outer ring. Before scoring, the solver closes that gap (`stitch_seam`):
the outer ring is shrunk by `scale=0.93` so its inner edge moves in, and the
*circular* part of the inner disc (black corners excluded) is stitched back onto
the centre. The Sobel seam band then sits exactly at radius 40. Preprocessing is
on by default; disable with `ImageRotationSpecialSolver(preprocess=False)` or
tune `inner_r` / `scale`. The result dict reports `preprocessed: true`.

---

## 7. Eager loading for low latency

Solvers lazy-load their weights on first use, so a cold first `Solve()` pays the
load cost (and, for Image Rotation Default, the one-time ConvNeXt-L download).
For a real-world / server setting where each prediction must be fast, load
everything up front — either at construction or explicitly:

```python
daccas = DACCAS(..., preload=True)   # loads classifier + all solvers now
# or:
daccas = DACCAS(...)
daccas.warmup(verbose=True)          # same thing, on demand
```

After warmup, every `Solve()` only pays forward-pass compute — no weight
reloading. Notes:

* Startup loads four ~400 MB text models plus the rest, so warmup takes a while;
  it is a **one-time boot cost**. Construct DACCAS once and reuse it for all
  requests (don't rebuild per image).
* The ConvNeXt-L ImageNet weights download once and are cached by torch
  (`~/.cache/torch/hub/checkpoints` or `%USERPROFILE%\.cache\torch\...`), so only
  the very first run on a machine downloads them. Pass `convnext_weights_path`
  to avoid the network entirely.
* Use `device="cuda"` for materially faster forward passes on the neural solvers.

---

## 8. Capturing CAPTCHAs (getting an image in)

DACCAS does not drive a browser, on purpose. Capturing a live element always
needs an automation tool, and which one is your choice. The thing they all share
is that they can return a screenshot as PNG **bytes**, and DACCAS accepts bytes
directly — so any tool integrates the same way:

```
screenshot bytes (or PIL image / path / ndarray)  ->  Classify / Solve
```

`daccas.Classify(...)` and `daccas.Solve(...)` accept a file path, a PIL image,
raw bytes, or a numpy array interchangeably.

Optional convenience adapters live in `daccas.capture` (they don't import
Playwright/Selenium at import time, so neither is a required dependency):

**Playwright (sync):**
```python
from playwright.sync_api import sync_playwright
from daccas import DACCAS, capture

d = DACCAS(preload=True)
with sync_playwright() as p:
    page = p.chromium.launch().new_page()      # device_scale_factor=1 if HiDPI (see below)
    page.goto(url)
    img = capture.from_playwright(page.locator("#captcha-img"))
    print(d.run(img))

# equivalently, with no helper at all — just pass the bytes:
# png = page.locator("#captcha-img").screenshot()
# d.run(png)
```

**Playwright (async):** `img = await capture.from_playwright_async(page.locator("#captcha-img"))`

**Selenium:**
```python
from selenium.webdriver.common.by import By
from daccas import capture
img = capture.from_selenium(driver.find_element(By.ID, "captcha-img"))
# or: capture.from_selenium_driver(driver, By.ID, "captcha-img")
# raw-bytes equivalent: driver.find_element(...).screenshot_as_png
```

**Pixel-geometry caveat.** The Image Rotation (Special) solver assumes a 100×100
captcha with inner-disc radius 40. A screenshot can come back larger — especially
on a HiDPI/Retina display, where the device pixel ratio doubles the pixels. Fix
it by capturing at 1:1 (Playwright: `browser.new_context(device_scale_factor=1)`;
browser at 100% zoom) or by normalising on capture:
`capture.from_playwright(loc, target_size=(100, 100))`. The neural solvers resize
internally and are unaffected.

---

## 9. Extending the library (the design point)

Add a CAPTCHA type in three steps, touching nothing else:

1. Subclass `BaseSolver` (implement `_load` and `solve`); set `HANDLES`.
2. Add the class name to the classifier's `SAFE_TO_CLASS` map and retrain only
   the lightweight YOLOv8-cls dispatcher to recognise the new type.
3. Register the solver in `DACCAS.registry`.

Existing solvers and their weights are untouched.
