# TODO - BM25/Embedding/VectorStore dependency singletons refactor

- [ ] Inspect wiring in BootstrapService, IndexingPipeline, QueryPipeline, and retrieval services
- [ ] Update composition_root.py: add temporary id logging for BM25Index, EmbeddingService, VectorStoreService and ensure it remains the only instantiation site
- [ ] Update IndexingPipeline to accept injected dependencies via constructor injection
- [ ] Update IndexingService to accept injected BM25Index and EmbeddingService (and optionally vector store wrapper) via constructor injection; remove internal instantiation
- [ ] Update BootstrapService to construct IndexingPipeline with injected dependencies
- [ ] Update StartupValidator to log and validate identity ids (temporary)
- [ ] Update retrieval services / QueryPipeline wiring to reuse the same injected instances (no instantiation)
- [ ] Run `python -m compileall src`
- [ ] Run application startup (or provided bootstrap/test script)
- [ ] Verify identical `id()` values across startup/indexing/validation/retrieval and that counts match
- [ ] Remove temporary id logging
- [ ] Final verification and report

