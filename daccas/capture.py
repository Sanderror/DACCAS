"""CAPTCHA capture helpers (optional, tool-agnostic).

DACCAS deliberately does **not** own a browser. Capturing a live DOM element
always needs some automation driver, and which one (Playwright, Selenium,
Puppeteer, raw CDP, ...) is the user's choice. What every such tool has in
common is that it can hand you the element screenshot as PNG **bytes** -- and
`daccas` accepts bytes directly. So the integration contract is simply:

    bytes (or PIL image, path, ndarray)  ->  DACCAS.Classify / DACCAS.Solve

This module just provides convenience adapters so you don't have to remember the
exact screenshot call for each tool. None of them import playwright/selenium at
module import time, so neither is a hard dependency of DACCAS.

Important -- pixel geometry
---------------------------
The Image Rotation (Special) solver assumes the raw captcha is 100x100 with an
inner-disc radius of 40. A browser screenshot can come back at a different size,
especially under a HiDPI / Retina display (device pixel ratio > 1 doubles the
pixels). Either:
  * capture at device_scale_factor=1 (Playwright) / 100% zoom, or
  * pass `target_size=(100, 100)` here to normalise before solving.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

from PIL import Image

from .solvers.base import load_pil

Size = Optional[Tuple[int, int]]


def _finish(img: Image.Image, target_size: Size) -> Image.Image:
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    return img


def to_image(source, target_size: Size = None) -> Image.Image:
    """Normalise any supported source (path / bytes / PIL / ndarray) to a PIL
    image, optionally resized to `target_size`."""
    return _finish(load_pil(source), target_size)


# --------------------------------------------------------------------------- #
# Playwright (sync API)
# --------------------------------------------------------------------------- #
def from_playwright(locator, target_size: Size = None, **screenshot_kwargs) -> Image.Image:
    """Capture an element with Playwright's *sync* API.

        from playwright.sync_api import sync_playwright
        loc = page.locator("#captcha-img")
        img = from_playwright(loc)

    `locator` is anything with a sync `.screenshot()` returning PNG bytes
    (a Locator, or an ElementHandle)."""
    png = locator.screenshot(**screenshot_kwargs)
    return _finish(Image.open(io.BytesIO(png)), target_size)


# --------------------------------------------------------------------------- #
# Playwright (async API)
# --------------------------------------------------------------------------- #
async def from_playwright_async(locator, target_size: Size = None,
                                **screenshot_kwargs) -> Image.Image:
    """Capture an element with Playwright's *async* API.

        loc = page.locator("#captcha-img")
        img = await from_playwright_async(loc)
    """
    png = await locator.screenshot(**screenshot_kwargs)
    return _finish(Image.open(io.BytesIO(png)), target_size)


# --------------------------------------------------------------------------- #
# Selenium
# --------------------------------------------------------------------------- #
def from_selenium(element, target_size: Size = None) -> Image.Image:
    """Capture an element with Selenium.

        el = driver.find_element(By.ID, "captcha-img")
        img = from_selenium(el)

    `element` is a Selenium WebElement (uses `.screenshot_as_png`)."""
    png = element.screenshot_as_png
    return _finish(Image.open(io.BytesIO(png)), target_size)


def from_selenium_driver(driver, by, value, target_size: Size = None) -> Image.Image:
    """Convenience: locate + capture in one call.

        from selenium.webdriver.common.by import By
        img = from_selenium_driver(driver, By.ID, "captcha-img")
    """
    element = driver.find_element(by, value)
    return from_selenium(element, target_size=target_size)
