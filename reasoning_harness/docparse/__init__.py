"""docparse: extract text, tables and structure from PDF, images, Word and Excel files.

Born-digital formats are parsed exactly from their own structure. Images and scanned
PDF pages go through a small OCR network trained from scratch (docparse.ocr).
"""


def extract(path, **kw):
    from .extract import extract as _extract
    return _extract(path, **kw)
