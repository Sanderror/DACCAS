# DACCAS

**D**ark web **A**daptive **C**APTCHA **C**lassification **A**nd **S**olving.

DACCAS is a system created for automated CAPTCHA classification and solving on the dark web. It is capable of classifying and solving multiple dark web CAPTCHAs. The system embeds the framework below, which is modular in design, allowing any contributor to add new solutions to the system:

![Framework of DACCAS](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

A two-step pipeline: a YOLOv8 **classifier** acts as a **dispatcher** that routes an input CAPTCHA image to the matching module in the **solver library**. All solvers are integrated as modules. This allows for easy integration of new solutions. Currently the following dark web CAPTCHAs are supported:

* Text CAPTCHA
* Moving Window CAPTCHA
* Open Circle CAPTCHA
* Image Rotation CAPTCHA (two different variants: default and special)

```python
from daccas import DACCAS

# Build once at startup. preload=True loads every model now so it's not done during inference
# which means the solutions are instanteneous.
# device can also be switched to cuda, if you have a gpu
daccas = DACCAS(models_dir="models", charsets_dir="charsets", device="cpu",
                preload=True)

# For only classification
cls = daccas.Classify("captcha.png")
# For only solving
out = daccas.Solve("captcha.png", cls["class"])
# For doing both in one step:
out = daccas.run("captcha.png")
```

---

## 1. Directory layout

The directory looks as follows. All these files are already included by default, except for the models in the models/ directory. Such files can be obtained from https://drive.google.com/drive/folders/1zNzwtzaaarIKGj-JxQ_1HRhsxAcmazfL?usp=drive_link. You should then place them manually in the models/ directory.

```
DACCAS/
├── daccas/                      # the package (provided)
│   ├── __init__.py                  # Initializes the full package
│   ├── daccas.py                    # DACCAS: Classify() + Solve()
│   ├── capture.py                   # Captures an image of the HTML element (supports Playwright and Selenium)
│   ├── classifier.py                # DACCAS Classifier (the dispatcher)
│   ├── orientation_lr.py            # ConvNeXt-L extractor for image rotation default + preprocess the image
│   ├── solvers/                     # All Solvers
│   │   ├── base.py                     # BaseSolver (template class for integrating new solvers)
│   │   ├── open_circle.py              # Open circle solver
│   │   ├── image_rotation_default.py   # Image rotation default solver (uses orientation_lr.py)
│   │   ├── image_rotation_special.py   # Image rotation special solver (preprocess stitch image parts, then Sobel filter)
│   │   ├── moving_window.py            # Moving window solver
│   │   └── text_solver.py              # Text solver (uses textmodel/ directory .py files)
│   └── textmodel/                   # Transformer text model code
│       ├── model.py                   # CaptchaTDAModel
│       ├── resnet_tda.py              # Supporting underlying ResNet backbone
│       ├── tda.py                     # TDA module
│       └── dataset.py                 # CharsetMapper
│
├── models/                          # You put the model weights here
│   ├── CLASSIFICATION_MODEL.pt              # YOLOv8 (classification only)
│   ├── OPEN_CIRCLE_BEST_MODEL.pt            # YOLOv8
│   ├── IMAGE_ROTATION_DEFAULT_MODEL.joblib  # ConvNexT-Large + OvR-Logistic Regression
│   ├── IMAGE_ROTATION_DEFAULT_META.json     # Defines the posisble preprocessing and hyperparameters for the best model of above
│   ├── MOVING_WINDOW_MODEL.pth              # ResNet-34
│   ├── GREGWAR_BEST_MODEL.pth              # Transformer
│   ├── MOBICMS_BEST_MODEL.pth              # Transformer
│   ├── KING_FINETUNE_BEST_MODEL.pth        # Transformer
│   └── GENERAL_BEST_MODEL.pth              # Transformer
│
├── charsets/                            # The character set mappings for text and moving window captchas
│   ├── charset_36.txt                       # a-z 0-9  (text: Gregwar/Mobicms/King models)
│   ├── charset_62.txt                       # a-z A-Z 0-9  (text: General model)
│   └── moving_window_charset.txt            # a-z 0-9 exluding b and 8, and o and 0
│
├── EXAMPLE_USAGE.ipynb                  # Contains examples of the classification and solving models being applied on challenges
├── example.py                           # Contains a full inference example on a dark web site
└── requirements.txt                     # Contains the packages that need to be installed
```

The package finds weights by name under `models_dir` and charsets under
`charsets_dir`. To use different paths/names, pass them to the solver classes
directly (they all take explicit `*_path` arguments).

---

## 2. Install

First, create a virtual environment for your project, e.g. with the following:
```bash
python -m venv .venv && source .venv/bin/activate
```

Then, install the required packages with:

```bash
pip install -r requirements.txt
```

---

## 3. The pipeline

### `DACCAS.Classify(image)`
Runs `CLASSIFICATION_MODEL.pt` (YOLOv8-cls, `imgsz=224`). Returns
`{'class', 'confidence', 'raw_name'}`. The model emits filesystem-safe names
(e.g. `text_gregwar`), which are mapped to the explanatory labels (e.g. `Text (Gregwar)`).

The nine classes included are:
`Text (Gregwar)`, `Text (Mobicms)`, `Text (King)`, `Text (Other)`,
`Moving Window`, `Open Circle`, `Image Rotation (Default)`,
`Image Rotation (Special)`, `No Solver`.

These are thus all linked to one of the solvers in the solvers/ directory.

### `DACCAS.Solve(image, captcha_class)`
This routes the image to the solver belonging to the class, and applies the solving model on the captcha.

It provides a machine-readable output that can be integrated into web automation tools to solve the CAPTCHAs. The following outputs are given:

| Class | Solver | Returns |
|---|---|---|
| **No Solver** | — | `{solved: False, message: ...}`  |
| **Open Circle** | YOLOv8 (`imgsz=480`, `conf=0.25`) | `x, y` = centre of the most-confident box (`all_boxes` also returned) |
| **Image Rotation (Special)** | seam stitch (close black gap) -> Sobel seam score (`SEAM_R=40`, `delta=5`) | `sobel_score` (`objective: "minimize"`) |
| **Image Rotation (Default)** | ConvNeXt-L feature -> Logistic Regression (`crop_mode=circular`, `C=10`) | `p_upright` (`objective: "maximize"`) |
| **Moving Window** | ResNet-34 (contrast 1.25, image size 224x224) | `character`, `class_index`, `confidence` |
| **Text (Gregwar/Mobicms/King)** | TDA Transformer + `charset_36` | `text` |
| **Text (Other)** | TDA Transformer + `charset_62` | `text` |

The two rotation solvers return a per-image score plus an `objective` field
(`minimize` for Sobel, `maximize` for P(upright)). This is because the model has to be run on all rotated variants of a challenge image.
How to run such a model on all your variants can be found in `EXAMPLE_USAGE.ipynb`.

---

## 4. Loading the models at the start

During application in a real-world scenario, it is adviced to load daccas at the start with the argument `preload=True`.
This is because otherwise when a solver is first called, the weights still have to be loaded. This will take crucial time of the inference speed.
When it is loaded upfront before a crawl, this won't be an isssue.
This can be done in one of two ways:

```python
daccas = DACCAS(..., preload=True) 
```
or:
```python
daccas = DACCAS(...)
daccas.warmup(verbose=True)         
```

By default `device=cpu` is used, thus running the models on cpu. If you have access to a gpu, it is advice to use `device=cuda`. This will improve model inference speed, and the loading of the initial models.

---

## 5. Capturing images of CAPTCHAs

As discussed in the paper, DACCAS does not perform CAPTCHA detection. What it does support is capturing images of challenges. It supports this using the two main providers for web automation tools (in Python): Selenium and Playwright. The user has to obtain a UNIQUE ID (!) to the HTML ELEMENT containing the CAPTCHA. Then, DACCAS handles the rest, as is demonstrated in the examples below:

**Playwright (sync):**

For playwright, we have constructed the `capture.from_playwright` function.
```python
from playwright.sync_api import sync_playwright
from daccas import DACCAS, capture

daccas = DACCAS(models_dir="models", charsets_dir="charsets", device="cpu",
                preload=True)
with sync_playwright() as p:
    page = p.chromium.launch().new_page()      # device_scale_factor=1 if HiDPI (see below)
    page.goto(url)

    # THIS IS WHERE DACCAS CAPTURER IS USED
    # NOTE THAT THE ELEMENT TO THE CAPTCHA IMAGE HAS TO BE PROVIDED BY YOU! IT MUST BE A UNIQUE ELEMENT!
    img = capture.from_playwright(page.locator("#captcha-img"))

    # THEN YOU CAN RUN THE INFERENCE (CLASSIFICATION + SOLVING) ON THE CAPTURED IMAGE
    daccas.run(img)

# This can also be done without the DACCAS capturer system, if you want to
# png = page.locator("#captcha-img").screenshot()
# daccas.run(png)
```

**Playwright (async):**

For async it is the same as above, but then use `img = await capture.from_playwright_async(page.locator("#captcha-img"))` at line 11.

**Selenium:**

For selenium it works similar but then use the `capture.from_selenium` function instead
```python
from selenium.webdriver.common.by import By
from daccas import capture
daccas = DACCAS(models_dir="models", charsets_dir="charsets", device="cpu",
                preload=True)
img = capture.from_selenium(driver.find_element(By.ID, "captcha-img"))
# or: capture.from_selenium_driver(driver, By.ID, "captcha-img")
daccas.run(img)
```

Note that it is not obligatory to use the DACCAS Capturer functions. The examples show that one can also just obtain the image in bytes or as an actual image using Playwright and Selenium themselves, and then pass the image to DACCAS run() or classify() function.

---

## 6. Extending the library (Community Contributed Solvers)

As mentioned, anyone can contribute new solutions to DACCAS.

To do so, you should follow the following three steps without touching anything else:

1. Create a Subclass of `BaseSolver` in the solvers directory (implement `_load` and `solve`) and set `HANDLES`.
2. Add the class name to the `classifier.py` classifier's `SAFE_TO_CLASS` map and retrain only
   the lightweight YOLOv8-cls dispatcher to recognise the new type.
3. Register the solver in `DACCAS.registry` by adding the model weights to the DEFAULT_FILES, and adding it to `self.registry: dict[str, object] = {` in line 91.

Existing solvers and their weights should not be touched.
