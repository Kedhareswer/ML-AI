"""ExtractBench (LlamaIndex) integration: a two-stage parse -> local LLM extraction pipeline.

ExtractBench scores schema-guided extraction (document + JSON Schema -> JSON). A parser alone
cannot fill a schema, so docparse takes part the way ExtractBench's own two-stage baselines do:
it is the *parse stage*, and a language model reads its Markdown and returns the JSON.

To isolate what the parser contributes, the same model, prompt, context budget and scorer are
used with two parse stages:
    docparse_extract_<model>  docparse Markdown (headings, HTML tables, OCR for scanned pages)
    pypdf_extract_<model>     plain pypdf text, the simplest local baseline

The model is served by any OpenAI-compatible endpoint (e.g. llama.cpp's server on a CPU):
    python -m llama_cpp.server --model qwen2.5-3b-instruct-q4_k_m.gguf --n_ctx 16384 --port 8088
    DOCPARSE_LLM_URL=http://127.0.0.1:8088 python scripts/extractbench_docparse.py run docparse_extract_local --test
"""
import json
from pathlib import Path
from typing import Any

from extract_bench.extensions import register_pipeline, register_provider
from extract_bench.inference.providers.extract.vllm_extract import VLLMExtractProvider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType

from .extract import extract, to_markdown

CHARS_PER_TOKEN = 2.5  # conservative: JSON schemas and numbers tokenise densely


def docparse_pages(path):
    doc = extract(path)
    pages = []
    for n in range(1, max(1, doc["pages"]) + 1):
        blocks = [b for b in doc["blocks"] if b.get("page", 1) == n]
        pages.append(to_markdown({**doc, "blocks": blocks, "pages": 1}, tables="html", page_marks=False).strip())
    return pages


def pypdf_pages(path):
    from pypdf import PdfReader
    return [(p.extract_text() or "").strip() for p in PdfReader(str(path)).pages]


PARSERS = {"docparse": docparse_pages, "pypdf": pypdf_pages}


@register_provider("parsed_text_local_llm")
class ParsedTextLocalLLM(VLLMExtractProvider):
    """ExtractBench's self-hosted provider, fed parsed page text instead of page images.

    Extra config: parser ("docparse" | "pypdf"), n_ctx (served context length, tokens).
    Everything else (JSON salvage, schema pruning, normalisation, scoring) is the harness's own.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._parser = self.base_config.get("parser", "docparse")
        self._n_ctx = int(self.base_config.get("n_ctx", 16384))

    def _render_page_images(self, source_path: Path) -> list[str]:
        return [f"<!-- Page {i} -->\n{t}" for i, t in enumerate(PARSERS[self._parser](source_path), 1)]

    def _build_user_content(self, pages: list[str], schema: dict[str, Any], compact_schema: bool = False):
        schema_text = json.dumps(schema, separators=(",", ":")) if compact_schema else json.dumps(schema, indent=2)
        return ("Extract every field from the document text below according to the JSON schema.\n\n"
                f"JSON schema:\n{schema_text}\n\nDocument text:\n" + "\n\n".join(pages) + "\n\n"
                "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
                f"{self._user_instruction}")

    @staticmethod
    def _fit_pages(pages, budget):
        kept, used = [], 0
        for p in pages:
            if used + len(p) > budget:
                if budget - used > 200:
                    kept.append(p[: budget - used])
                    used = budget
                return kept, used, len(pages) - len(kept)
            kept.append(p)
            used += len(p) + 2
        return kept, used, 0

    def _call_api_fitting_context(self, schema: dict[str, Any], pages: list[str]) -> dict[str, Any]:
        """Fit schema + text + output into the served context: compact the schema, prune its trailing
        fields if it alone is too large, then cut the document text to the remaining budget.
        Both parse stages get exactly the same budget."""
        cur, pruned = schema, 0
        out_tokens = min(self._max_tokens, self._n_ctx // 4)
        overhead = len(self._system_prompt) + len(self._user_instruction) + 400
        while True:
            schema_chars = len(json.dumps(cur, separators=(",", ":")))
            budget = int((self._n_ctx - out_tokens) * CHARS_PER_TOKEN) - schema_chars - overhead
            if budget > 2000 or (nxt := self._prune_schema(cur)) is None:
                break
            cur, pruned = nxt, pruned + 1
        for _ in range(4):
            kept, used, cut = self._fit_pages(pages, budget)
            try:
                out = self._call_api(cur, kept, max_tokens=out_tokens, compact_schema=True)
                break
            except Exception as e:  # the server rejects requests over its context window: halve the text
                if not any(k in str(e).lower() for k in ("context", "exceed", "400")):
                    raise
                budget //= 2
        else:
            raise RuntimeError("request does not fit the model context even with a small part of the text")
        out["_config"].update(parser=self._parser, n_ctx=self._n_ctx, text_chars_sent=used)
        if cut:
            out["pages_truncated"], out["truncated"] = cut, True
        if pruned:
            out["schema_prune_rounds"], out["truncated"] = pruned, True
        return out


for parser in PARSERS:
    register_pipeline(PipelineSpec(
        pipeline_name=f"{parser}_extract_local", provider_name="parsed_text_local_llm",
        product_type=ProductType.EXTRACT,
        config={"parser": parser, "endpoint_env_var": "DOCPARSE_LLM_URL", "model": "qwen2.5-3b", "n_ctx": 16384,
                "max_tokens": 4096, "structured_output": False, "temperature": 0.0, "timeout_s": 3600}))
