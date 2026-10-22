"""ParseBench (LlamaIndex) integration: registers a `docparse` pipeline with parse-bench.

Usage (parse-bench needs Python 3.12+; install it next to docparse):
    pip install "parse-bench[fast]"
    python scripts/parsebench_docparse.py download
    python scripts/parsebench_docparse.py run docparse            # or: run docparse --test

Each page becomes Markdown (headings, lists, HTML tables) plus layout boxes labelled with
ParseBench's canonical classes, so all five dimensions can be scored.
"""
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.evaluation.layout_adapters.adapters import PyMuPDF4LLMLayoutAdapter
from parse_bench.extensions import register_layout_adapter, register_pipeline, register_provider
from parse_bench.inference.providers.base import Provider, ProviderPermanentError
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

from .extract import extract, plain, to_markdown

LABELS = {"paragraph": "Text", "list_item": "List-item", "table": "Table", "figure": "Picture",
          "header": "Page-header", "footer": "Page-footer"}


def _label(block):
    if block["type"] == "heading":
        return "Title" if block.get("level", 1) == 1 else "Section-header"
    return LABELS.get(block["type"], "Text")


def page_records(doc):
    """docparse document -> one record per page: body markdown, running header/footer, normalised boxes."""
    sizes = doc.get("page_sizes") or []
    pages = []
    for n in range(1, max(1, doc["pages"]) + 1):
        blocks = [b for b in doc["blocks"] if b.get("page", 1) == n]
        boxes, parts, furniture = [], [], {"header": [], "footer": []}
        number = ""
        for b in blocks:
            if b["type"] in furniture:  # page furniture: its own page fields
                furniture[b["type"]].append(b["text"])
                if b.get("page_number") and not number:
                    number = plain(b["text"]).strip()
                # Also kept in the body inside <page_header>/<page_footer> tags (the ParseBench convention).
                md = f"<page_{b['type']}>{b['text']}</page_{b['type']}>"
            else:
                md = to_markdown({**doc, "blocks": [b], "pages": 1}, tables="html", page_marks=False).strip()
            start = sum(len(p) + 2 for p in parts)
            parts.append(md)
            pos = [start, start + len(md)]
            if b.get("bbox") and n <= len(sizes):
                w, h = sizes[n - 1]
                boxes.append({"class": _label(b), "bbox": list(b["bbox"]), "pos": pos, "text": md,
                              "width": w, "height": h})
        w, h = sizes[n - 1] if n <= len(sizes) else (0.0, 0.0)
        pages.append({"page_index": n - 1, "page_number": n, "text": "\n\n".join(parts), "width": w, "height": h,
                      "page_boxes": boxes, "header": "\n".join(furniture["header"]),
                      "footer": "\n".join(furniture["footer"]), "printed_page_number": number})
    return pages


@register_provider("docparse")
class DocparseProvider(Provider):
    """Local, CPU-only: exact parsing of the PDF text layer; OCR for images and scanned pages."""

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"docparse only supports PARSE, got {request.product_type}")
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")
        started = datetime.now()
        try:
            doc = extract(path, ocr=self.base_config.get("ocr", "auto"))
        except Exception as e:  # report and keep the run going
            raise ProviderPermanentError(f"docparse error: {type(e).__name__}: {e}") from e
        done = datetime.now()
        raw = {"pages": page_records(doc), "num_pages": doc["pages"], "warnings": doc["warnings"]}
        return RawInferenceResult(request=request, pipeline=pipeline, pipeline_name=pipeline.pipeline_name,
                                  product_type=request.product_type, raw_output=raw, started_at=started,
                                  completed_at=done, latency_in_ms=int((done - started).total_seconds() * 1000))

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages, layout = [], []
        for p in raw_result.raw_output.get("pages", []):
            md = p["text"]
            pages.append(PageIR(page_index=p["page_index"], markdown=md))
            if not (p["width"] and p["height"]):
                continue
            items = []
            for box in p["page_boxes"]:
                x0, y0, x1, y1 = box["bbox"]
                W, H = box["width"], box["height"]
                x0, x1 = max(0.0, min(x0, W)), max(0.0, min(x1, W))
                y0, y1 = max(0.0, min(y0, H)), max(0.0, min(y1, H))
                if x1 <= x0 or y1 <= y0:
                    continue
                a, b = box["pos"] or (None, None)
                content = md[a:b] if a is not None else box["text"]
                seg = LayoutSegmentIR(x=x0 / W, y=y0 / H, w=(x1 - x0) / W, h=(y1 - y0) / H, label=box["class"],
                                      start_index=a, end_index=b)
                kind = {"Table": "table", "Picture": "image"}.get(box["class"], "text")
                items.append(LayoutItemIR(type=kind, md=content, html=content if kind == "table" else "",
                                          value=content, bbox=seg, layout_segments=[seg]))
            layout.append(ParseLayoutPageIR(page_number=p["page_number"], width=p["width"], height=p["height"],
                                            md=md, text=md, items=items, page_header_markdown=p.get("header", ""),
                                            page_footer_markdown=p.get("footer", ""),
                                            printed_page_number=p.get("printed_page_number", "")))
        output = ParseOutput(task_type="parse", example_id=raw_result.request.example_id,
                             pipeline_name=raw_result.pipeline_name, pages=pages, layout_pages=layout,
                             markdown="\n\n".join(p.markdown for p in pages))
        return InferenceResult(request=raw_result.request, pipeline_name=raw_result.pipeline_name,
                               product_type=raw_result.product_type, raw_output=raw_result.raw_output, output=output,
                               started_at=raw_result.started_at, completed_at=raw_result.completed_at,
                               latency_in_ms=raw_result.latency_in_ms)


@register_layout_adapter("docparse", priority=90)
class DocparseLayoutAdapter(PyMuPDF4LLMLayoutAdapter):
    """Same page-box layout as PyMuPDF4LLM; docparse already emits canonical labels."""


register_pipeline(PipelineSpec(pipeline_name="docparse", provider_name="docparse",
                               product_type=ProductType.PARSE, config={"ocr": "auto"}))
