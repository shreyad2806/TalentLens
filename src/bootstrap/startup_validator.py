"""
Startup Validator Module for Bootstrap System.

This module provides validation logic to verify the production readiness of
every required service at startup. Each check is wrapped so that the validator
never crashes because of a missing optional service; it reports PASS, WARNING
or FAIL for every component and lets the caller decide whether to continue.
"""

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class StartupValidator:
    """
    Production startup validator.

    Validates every required service and reports a status per component:
        - PASS  : the service is ready
        - WARNING : the service is missing, disabled, or empty but the system
                    can continue (e.g. no cached chunks, model not loaded yet)
        - FAIL  : the service is misconfigured or unavailable and is required
    """

    OPTIONAL_ENV = {
        "QDRANT_API_KEY", "BGE_MODEL_PATH", "OFFLINE_MODE",
        "MODEL_DOWNLOAD_RETRIES", "USE_GPU",
    }

    def __init__(self, indexing_pipeline):
        """
        Initialize the startup validator.

        Args:
            indexing_pipeline: IndexingPipeline instance to validate.
        """
        self.indexing_pipeline = indexing_pipeline
        logger.info("StartupValidator initialized")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """
        Run the full startup validation suite.

        Returns:
            Dictionary with per-check status, overall validity, and timing.
        """
        start_time = time.perf_counter()

        validation_result = {
            "is_valid": False,
            "deployment_ready": False,
            "checks": {},
            "statistics": {},
            "errors": [],
            "warnings": [],
            "startup_duration_ms": 0.0,
        }

        # Baseline indexing statistics
        try:
            stats = self.indexing_pipeline.get_statistics()
            # Ensure the statistics payload is JSON-serializable for reports/telemetry
            if "bm25_stats" in stats and not isinstance(stats["bm25_stats"], dict):
                try:
                    if dataclasses.is_dataclass(stats["bm25_stats"]):
                        stats["bm25_stats"] = dataclasses.asdict(stats["bm25_stats"])
                    else:
                        stats["bm25_stats"] = dict(stats["bm25_stats"])
                except Exception:
                    stats["bm25_stats"] = str(stats["bm25_stats"])
            validation_result["statistics"] = stats
        except Exception as exc:
            stats = {}
            validation_result["statistics"] = stats
            validation_result["errors"].append(
                f"Failed to read indexing pipeline statistics: {exc}"
            )

        # Service checks
        checks = [
            ("environment_variables", self._check_environment_variables),
            ("configuration", self._check_configuration),
            ("dataset", self._check_dataset),
            ("resume_document_model", self._check_resume_document_model),
            ("chunk_cache", self._check_chunk_cache),
            ("embedding_model", self._check_embedding_model),
            ("vector_store", self._check_vector_store),
            ("documents_indexed", lambda: self._check_documents_indexed(stats)),
            ("vectors_indexed", lambda: self._check_vectors_indexed(stats)),
            ("bm25_indexed", lambda: self._check_bm25_indexed(stats)),
            ("services_healthy", self._check_services_healthy),
            ("search_service", self._check_search_service),
            ("hybrid_retriever", self._check_hybrid_retriever),
            ("consistency", lambda: self._check_consistency(stats)),
        ]

        for name, check_fn in checks:
            try:
                result = check_fn()
            except Exception as exc:
                result = self._make_result(
                    "FAIL",
                    f"Check {name} raised an unexpected exception: {exc}",
                )
                logger.exception(f"Unexpected error in startup check {name}")

            validation_result["checks"][name] = result

            if result["status"] == "FAIL":
                validation_result["errors"].append(result["message"])
            elif result["status"] == "WARNING":
                validation_result["warnings"].append(result["message"])

        # Overall result
        has_failures = any(
            c["status"] == "FAIL" for c in validation_result["checks"].values()
        )
        has_warnings = any(
            c["status"] == "WARNING" for c in validation_result["checks"].values()
        )

        validation_result["is_valid"] = not has_failures
        validation_result["deployment_ready"] = not has_failures and not has_warnings
        validation_result["startup_duration_ms"] = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        if validation_result["is_valid"]:
            logger.info("System validation passed - no failing checks")
        else:
            logger.warning(
                f"System validation failed with {len(validation_result['errors'])} errors"
            )

        return validation_result

    def is_healthy(self) -> bool:
        """Quick health check based on the full validation."""
        return self.validate()["is_valid"]

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(status: str, message: str, **extra) -> Dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "passed": status in ("PASS", "WARNING"),
            **extra,
        }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_environment_variables(self) -> Dict[str, Any]:
        """Validate the presence of environment variables."""
        expected = [
            "VECTOR_STORE_PROVIDER",
            "EMBEDDING_MODEL",
            "EMBEDDING_DEVICE",
            "TOP_K",
            "SCORE_THRESHOLD",
            "BM25_K1",
            "BM25_B",
            "RERANKER_MODEL",
            "RERANKER_TOP_K",
            "LOG_LEVEL",
            "STREAMLIT_SERVER_PORT",
            "STREAMLIT_SERVER_ADDRESS",
        ]

        # Provider-specific settings
        provider = os.getenv("VECTOR_STORE_PROVIDER", "qdrant").lower()
        if provider == "qdrant":
            expected.extend(["QDRANT_URL", "QDRANT_COLLECTION", "QDRANT_API_KEY"])
        elif provider == "pinecone":
            expected.extend([
                "PINECONE_API_KEY",
                "PINECONE_INDEX",
                "PINECONE_CLOUD",
                "PINECONE_REGION",
            ])

        missing = [key for key in expected if not os.getenv(key)]

        if missing:
            return self._make_result(
                "WARNING",
                f"Missing environment variables (defaults will be used): {', '.join(missing)}",
                missing=missing,
                provider=provider,
            )

        return self._make_result(
            "PASS",
            f"All expected environment variables are present (provider={provider})",
            provider=provider,
        )

    def _check_configuration(self) -> Dict[str, Any]:
        """Validate the main configuration module loads and has sensible values."""
        try:
            from ..config import EMBEDDING_MODEL, EMBEDDING_DIM, CATEGORIES

            issues = []
            if not EMBEDDING_MODEL:
                issues.append("EMBEDDING_MODEL is empty")
            if not EMBEDDING_DIM or EMBEDDING_DIM <= 0:
                issues.append("EMBEDDING_DIM is not a positive integer")
            if not CATEGORIES:
                issues.append("CATEGORIES list is empty")

            if issues:
                return self._make_result(
                    "FAIL",
                    f"Configuration issues: {'; '.join(issues)}",
                    embedding_model=EMBEDDING_MODEL,
                    embedding_dim=EMBEDDING_DIM,
                    categories_count=len(CATEGORIES or []),
                )
            return self._make_result(
                "PASS",
                f"Configuration loaded: model={EMBEDDING_MODEL}, dim={EMBEDDING_DIM}",
                embedding_model=EMBEDDING_MODEL,
                embedding_dim=EMBEDDING_DIM,
                categories_count=len(CATEGORIES or []),
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Failed to load application configuration: {exc}",
            )

    def _check_dataset(self) -> Dict[str, Any]:
        """Validate that at least one resume data source is available."""
        from .resume_loader import ResumeLoader

        try:
            loader = ResumeLoader()
            load_result = loader.load_resumes()
            combined_dataset = Path("combined/production_dataset.json")

            has_source = (
                load_result.valid_files > 0
                or load_result.csv_detected
                or combined_dataset.exists()
            )

            if not has_source:
                return self._make_result(
                    "FAIL",
                    "No resume data source found (no valid files, no CSV, no combined dataset)",
                    valid_files=load_result.valid_files,
                    csv_detected=load_result.csv_detected,
                    combined_exists=combined_dataset.exists(),
                )

            return self._make_result(
                "PASS",
                f"Dataset available: valid_files={load_result.valid_files}, "
                f"csv_detected={load_result.csv_detected}, "
                f"combined_exists={combined_dataset.exists()}",
                valid_files=load_result.valid_files,
                csv_detected=load_result.csv_detected,
                combined_exists=combined_dataset.exists(),
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Dataset check failed: {exc}",
            )

    def _check_resume_document_model(self) -> Dict[str, Any]:
        """Validate that the ResumeDocument model is importable and can validate a sample."""
        try:
            from ..models import ResumeDocument

            # Confirm the schema is importable and valid
            schema = ResumeDocument.model_json_schema()
            fields = list(schema.get("properties", {}).keys())

            # Try to validate a sample record if one exists
            combined_dataset = Path("combined/production_dataset.json")
            sample_validated = False
            if combined_dataset.exists():
                try:
                    with open(combined_dataset, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    if records:
                        ResumeDocument.model_validate(records[0])
                        sample_validated = True
                except Exception as val_exc:
                    return self._make_result(
                        "FAIL",
                        f"ResumeDocument failed to validate a sample record: {val_exc}",
                        fields=fields,
                    )

            return self._make_result(
                "PASS",
                f"ResumeDocument model is valid (sample_validated={sample_validated})",
                fields=fields,
                sample_validated=sample_validated,
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"ResumeDocument model could not be loaded: {exc}",
            )

    def _check_chunk_cache(self) -> Dict[str, Any]:
        """Validate chunk cache presence (optional for cold starts)."""
        cache_dir = Path("data/cache")
        indexed_docs_cache = cache_dir / "indexed_documents.json"

        if not cache_dir.exists():
            return self._make_result(
                "WARNING",
                "Chunk cache directory does not exist (data/cache)",
                exists=False,
            )

        cache_files = list(cache_dir.glob("*.json"))
        if not indexed_docs_cache.exists() and not cache_files:
            return self._make_result(
                "WARNING",
                "No chunk cache files found in data/cache",
                cache_dir=str(cache_dir),
                cache_file_count=0,
            )

        return self._make_result(
            "PASS",
            f"Chunk cache directory present with {len(cache_files)} JSON file(s)",
            cache_dir=str(cache_dir),
            cache_file_count=len(cache_files),
        )

    def _check_embedding_model(self) -> Dict[str, Any]:
        """Validate that the embedding model loader is reachable."""
        try:
            from ..embeddings.model_loader import get_model_loader
            loader = get_model_loader()
            diagnostics = loader.get_diagnostics()

            if diagnostics.get("is_loaded"):
                return self._make_result(
                    "PASS",
                    f"Embedding model loaded: {diagnostics['model_name']} "
                    f"({diagnostics['device']}, {diagnostics['memory_usage_mb']:.1f} MB)",
                    diagnostics=diagnostics,
                )

            return self._make_result(
                "WARNING",
                f"Embedding model not yet loaded; it will be lazy-loaded on first use: "
                f"{diagnostics['model_name']} on {diagnostics['device']}",
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Embedding model check failed: {exc}",
            )

    def _check_vector_store(self) -> Dict[str, Any]:
        """Validate the configured vector store service."""
        try:
            from ..vector_store import VectorStoreService
            service = VectorStoreService()

            count = -1
            healthy = None
            try:
                count = service.count()
                if hasattr(service.vector_store, "is_healthy"):
                    healthy = service.vector_store.is_healthy()
            except Exception as exc:
                return self._make_result(
                    "FAIL",
                    f"Vector store service created but query failed: {exc}",
                    provider=service.config.provider.value,
                )

            if healthy is not None and not healthy:
                return self._make_result(
                    "FAIL",
                    f"Vector store health check failed (count={count})",
                    provider=service.config.provider.value,
                    count=count,
                )

            if count == 0:
                return self._make_result(
                    "WARNING",
                    "Vector store is reachable but contains no vectors",
                    provider=service.config.provider.value,
                    count=count,
                )

            return self._make_result(
                "PASS",
                f"Vector store healthy: {count} vector(s) on {service.config.provider.value}",
                provider=service.config.provider.value,
                count=count,
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Vector store service could not be initialized: {exc}",
            )

    def _check_documents_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check that documents are indexed."""
        doc_count = stats.get("indexed_documents", 0)
        if doc_count == 0:
            return self._make_result(
                "FAIL",
                "No documents indexed - bootstrap may have failed",
                count=doc_count,
            )
        return self._make_result(
            "PASS",
            f"Documents indexed: {doc_count}",
            count=doc_count,
        )

    def _check_vectors_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check that vectors are indexed."""
        vector_count = stats.get("vector_count", 0)
        if vector_count == 0:
            return self._make_result(
                "FAIL",
                "No vectors indexed - embedding generation may have failed",
                count=vector_count,
            )
        return self._make_result(
            "PASS",
            f"Vectors indexed: {vector_count}",
            count=vector_count,
        )

    def _check_bm25_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check that BM25 documents are indexed."""
        bm25_count = stats.get("bm25_count", 0)
        if bm25_count == 0:
            return self._make_result(
                "FAIL",
                "No BM25 documents indexed - sparse indexing may have failed",
                count=bm25_count,
            )
        return self._make_result(
            "PASS",
            f"BM25 documents indexed: {bm25_count}",
            count=bm25_count,
        )

    def _check_services_healthy(self) -> Dict[str, Any]:
        """Check that the indexing service is available."""
        try:
            if self.indexing_pipeline.indexing_service is None:
                return self._make_result(
                    "FAIL",
                    "Indexing service not available",
                )
            return self._make_result(
                "PASS",
                "Indexing service available",
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Indexing service check failed: {exc}",
            )

    def _check_search_service(self) -> Dict[str, Any]:
        """Validate SearchService can respond without crashing."""
        try:
            from ..search import SearchService
            search = SearchService(hybrid_service=None)
            results = search.search("python", top_k=1)
            return self._make_result(
                "PASS",
                f"SearchService responded with {len(results)} metadata-scored result(s)",
                result_count=len(results),
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"SearchService check failed: {exc}",
            )

    def _check_hybrid_retriever(self) -> Dict[str, Any]:
        """Validate the hybrid retriever can respond without crashing."""
        try:
            from ..bootstrap.composition_root import create_retrieval_bundle
            bundle = create_retrieval_bundle()
            results = bundle.hybrid_service.search("python", top_k=1)

            if not results:
                return self._make_result(
                    "WARNING",
                    "Hybrid retriever is operational but returned 0 results",
                    result_count=0,
                )

            return self._make_result(
                "PASS",
                f"Hybrid retriever responded with {len(results)} result(s)",
                result_count=len(results),
            )
        except Exception as exc:
            return self._make_result(
                "FAIL",
                f"Hybrid retriever check failed: {exc}",
            )

    def _check_consistency(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency between document, vector, and BM25 counts."""
        doc_count = stats.get("indexed_documents", 0)
        vector_count = stats.get("vector_count", 0)
        bm25_count = stats.get("bm25_count", 0)

        all_zero = (doc_count == 0 and vector_count == 0 and bm25_count == 0)
        all_non_zero = (doc_count > 0 and vector_count > 0 and bm25_count > 0)

        if all_zero or all_non_zero:
            return self._make_result(
                "PASS",
                "Index counts are consistent",
                doc_count=doc_count,
                vector_count=vector_count,
                bm25_count=bm25_count,
            )

        return self._make_result(
            "WARNING",
            f"Inconsistent index counts: docs={doc_count}, vectors={vector_count}, bm25={bm25_count}",
            doc_count=doc_count,
            vector_count=vector_count,
            bm25_count=bm25_count,
        )
