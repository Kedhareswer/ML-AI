"""Command line: docparse extract | train-ocr | bench | ui."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docparse")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract text, tables and structure from a document")
    e.add_argument("files", nargs="+", help="PDF, image, DOCX/DOC, XLSX/XLS, CSV, PPTX ...")
    e.add_argument("--format", default="md", choices=["md", "json", "text"])
    e.add_argument("--ocr", default="auto", choices=["auto", "always", "never"],
                   help="auto: OCR only PDF pages that have no text layer")
    e.add_argument("--ocr-images", action="store_true", help="also read text inside pictures embedded in documents")
    e.add_argument("--model", help="OCR weights (default artifacts/docparse/ocr.pt)")
    e.add_argument("--dpi", type=int, default=200, help="render resolution for scanned PDF pages")
    e.add_argument("--password", help="password for encrypted PDFs")
    e.add_argument("-o", "--output", help="output file (one input) or directory (several inputs)")

    t = sub.add_parser("train-ocr", help="train the line recognizer from scratch on synthetic lines")
    t.add_argument("--run", required=True)
    t.add_argument("--rounds", type=int, default=4)
    t.add_argument("--lines-per-round", type=int, default=120_000)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--resume", action="store_true")

    b = sub.add_parser("bench", help="benchmark OCR on held-out lines, synthetic pages and real PDF pages")
    b.add_argument("--model", default="artifacts/docparse/ocr.pt")
    b.add_argument("--output", required=True)
    b.add_argument("--lines", type=int, default=2000)
    b.add_argument("--pages", type=int, default=40)
    b.add_argument("--real-pages", type=int, default=40)

    u = sub.add_parser("ui", help="local browser UI: drop a file, get Markdown and JSON")
    u.add_argument("--port", type=int, default=7862)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--model")

    args = ap.parse_args(argv)
    if args.cmd == "extract":
        from .extract import extract, to_markdown, to_text
        for f in args.files:
            doc = extract(f, ocr=args.ocr, ocr_images=args.ocr_images, model=args.model, dpi=args.dpi,
                          password=args.password)
            body = (json.dumps(doc, indent=2, ensure_ascii=False, default=str) if args.format == "json"
                    else to_markdown(doc) if args.format == "md" else to_text(doc) + "\n")
            if args.output and len(args.files) > 1:
                out = Path(args.output)
                out.mkdir(parents=True, exist_ok=True)
                (out / (Path(f).name + "." + args.format)).write_text(body, encoding="utf-8")
            elif args.output:
                Path(args.output).write_text(body, encoding="utf-8")
            else:
                sys.stdout.write(body)
            for w in doc["warnings"]:
                print(f"warning: {Path(f).name}: {w}", file=sys.stderr)
    elif args.cmd == "train-ocr":
        from .ocr.train import train
        _, info = train(args.run, args.rounds, args.lines_per_round, args.batch_size, args.lr, args.seed,
                        args.threads, args.resume, log=lambda m: print(m, flush=True))
        print(json.dumps(info["history"][-1], indent=2))
    elif args.cmd == "bench":
        from .bench import run
        run(args.model, args.output, args.lines, args.pages, args.real_pages)
    elif args.cmd == "ui":
        from .ui import serve
        serve(args.port, args.host, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
