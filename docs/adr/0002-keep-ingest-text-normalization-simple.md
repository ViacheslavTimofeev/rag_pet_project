# Keep ingest text normalization simple

The `_normalize_text` step in `src/ingest/raw_creation.py` is intended to produce readable raw text for downstream chunking, not to fully parse or semantically rewrite source documents. We will keep this normalization limited to HTML entity unescaping, removing script/style blocks and tags, and collapsing whitespace/newlines, because heavier cleanup can hide source-data problems and make ingestion behavior harder to inspect.
