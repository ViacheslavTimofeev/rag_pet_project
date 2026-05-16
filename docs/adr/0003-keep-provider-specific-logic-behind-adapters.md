# Keep provider-specific logic behind adapters

Model, embedding, and vector database providers differ in client APIs, request shapes, response formats, and operational assumptions. We will keep provider-specific logic behind small adapters in the relevant backend modules instead of letting high-level ingestion, retrieval, evaluation, or generation orchestration call providers directly, because this project values backend replaceability and testability more than the absolute minimum amount of glue code.
