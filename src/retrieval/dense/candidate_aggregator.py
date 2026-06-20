"""
Candidate Aggregator for Dense Retrieval Service.

This module provides functionality to aggregate multiple chunks from the same
resume into a single candidate score using configurable aggregation strategies.

Architecture Notes:
- Aggregates chunks by resume_id
- Configurable aggregation strategies (max, average, weighted)
- Preserves evidence chunks and matched sections
- Calculates final candidate scores
- Returns ranked candidate list

SOLID Principles Applied:
- Single Responsibility: Handles only candidate aggregation
- Open/Closed: Open for new aggregation strategies
- Dependency Inversion: Depends on aggregation interface
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from enum import Enum
from .schema import DenseSearchResult, AggregatedCandidateResult

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """
    Enumeration of available aggregation strategies.
    
    Strategies:
        MAX: Use the maximum score across all chunks
        AVERAGE: Use the average score across all chunks
        WEIGHTED: Use weighted average based on section weights
    """
    MAX = "max"
    AVERAGE = "average"
    WEIGHTED = "weighted"


class CandidateAggregator:
    """
    Aggregates multiple chunks from the same resume into a single candidate score.
    
    This class takes search results (which may contain multiple chunks from the
    same resume) and aggregates them into a single candidate-level score using
    configurable weighted aggregation based on sections.
    
    Architecture Pattern: Aggregator Pattern
    - Groups chunks by resume_id
    - Applies section-specific weights
    - Calculates weighted average scores
    - Preserves evidence chunks for transparency
    
    Example:
        Resume A:
        - Skills score = 0.95 (weight: 0.4)
        - Experience score = 0.91 (weight: 0.3)
        - Projects score = 0.82 (weight: 0.3)
        
        Final Score = (0.95 * 0.4) + (0.91 * 0.3) + (0.82 * 0.3) = 0.90
    """
    
    def __init__(
        self,
        section_weights: Optional[Dict[str, float]] = None,
        strategy: AggregationStrategy = AggregationStrategy.WEIGHTED
    ):
        """
        Initialize the candidate aggregator.
        
        Args:
            section_weights: Dictionary mapping section names to weights.
                          Default weights: skills=0.4, experience=0.3, projects=0.3
                          Weights should sum to 1.0 (used only for WEIGHTED strategy)
            strategy: Aggregation strategy to use (default: WEIGHTED)
                     Options: MAX, AVERAGE, WEIGHTED
        """
        self.strategy = strategy
        
        if section_weights is None:
            # Default weights (used only for WEIGHTED strategy)
            self.section_weights = {
                'skills': 0.4,
                'experience': 0.3,
                'projects': 0.3
            }
        else:
            self.section_weights = section_weights
        
        # Validate weights sum to 1.0 (only for WEIGHTED strategy)
        if self.strategy == AggregationStrategy.WEIGHTED:
            total_weight = sum(self.section_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(
                    f"Section weights sum to {total_weight:.2f}, should sum to 1.0. "
                    "Weights will be normalized."
                )
                # Normalize weights
                self.section_weights = {
                    section: weight / total_weight
                    for section, weight in self.section_weights.items()
                }
        
        logger.info(
            f"CandidateAggregator initialized with strategy={strategy.value}, "
            f"section_weights={self.section_weights}"
        )
    
    def aggregate(self, results: List[DenseSearchResult]) -> List[AggregatedCandidateResult]:
        """
        Aggregate search results by candidate.
        
        Groups chunks by resume_id and aggregates scores using section weights.
        
        Args:
            results: List of search results to aggregate
            
        Returns:
            List of aggregated candidate results
        """
        if not results:
            return []
        
        # Group results by resume_id
        results_by_resume = defaultdict(list)
        for result in results:
            results_by_resume[result.resume_id].append(result)
        
        # Aggregate each candidate
        aggregated_candidates = []
        for resume_id, candidate_results in results_by_resume.items():
            aggregated = self._aggregate_candidate(candidate_results)
            aggregated_candidates.append(aggregated)
        
        # Sort by final score (descending)
        aggregated_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(
            f"Aggregated {len(aggregated_candidates)} candidates from "
            f"{len(results)} chunks"
        )
        
        return aggregated_candidates
    
    def _aggregate_candidate(self, results: List[DenseSearchResult]) -> AggregatedCandidateResult:
        """
        Aggregate results for a single candidate using the configured strategy.
        
        This method aggregates multiple chunk-level results into a single candidate-level
        score using one of three strategies:
        
        1. MAX Strategy: Takes the maximum score across all chunks
           - Simple and fast
           - Highlights the best matching section
           - Example: If skills=0.94, experience=0.89, projects=0.82, final=0.94
        
        2. AVERAGE Strategy: Takes the average score across all chunks
           - Balances all sections equally
           - Reduces impact of outliers
           - Example: If skills=0.94, experience=0.89, projects=0.82, final=(0.94+0.89+0.82)/3=0.88
        
        3. WEIGHTED Strategy: Takes weighted average based on section importance
           - Prioritizes important sections (e.g., skills weighted more)
           - Configurable weights for different sections
           - Example: If skills=0.94 (weight=0.4), experience=0.89 (weight=0.3), projects=0.82 (weight=0.3)
             final = (0.94*0.4) + (0.89*0.3) + (0.82*0.3) = 0.90
        
        Args:
            results: List of results for a single candidate
            
        Returns:
            Aggregated candidate result with preserved evidence chunks and matched sections
        """
        if not results:
            raise ValueError("Cannot aggregate empty results list")
        
        # Get candidate information
        first_result = results[0]
        candidate_name = first_result.candidate_name
        resume_id = first_result.resume_id
        
        # Group by section to preserve matched sections
        # This allows us to track which sections matched and their scores
        section_scores = defaultdict(list)
        evidence_chunks = []
        matched_sections = set()
        
        for result in results:
            section = result.section.lower()
            section_scores[section].append(result.normalized_score)
            matched_sections.add(result.section)
            
            # Preserve evidence chunks for transparency
            # Each evidence chunk contains: chunk_id, section, score, matched_text, rank
            evidence_chunks.append({
                'chunk_id': result.chunk_id,
                'section': result.section,
                'score': result.normalized_score,
                'matched_text': result.matched_text,
                'rank': result.rank
            })
        
        # Calculate section-level scores (max score for each section)
        # This preserves the best score per section for weighted aggregation
        section_final_scores = {}
        for section, scores in section_scores.items():
            section_final_scores[section] = max(scores)
        
        # Calculate final score based on configured strategy
        if self.strategy == AggregationStrategy.MAX:
            # MAX Strategy: Take the maximum score across all sections
            # This highlights the best matching section for the candidate
            final_score = max(section_final_scores.values())
            
            logger.debug(
                f"MAX aggregation for {candidate_name}: "
                f"section_scores={section_final_scores}, final_score={final_score:.4f}"
            )
            
        elif self.strategy == AggregationStrategy.AVERAGE:
            # AVERAGE Strategy: Take the average score across all sections
            # This balances all sections equally and reduces outlier impact
            all_scores = [score for scores in section_scores.values() for score in scores]
            final_score = sum(all_scores) / len(all_scores)
            
            logger.debug(
                f"AVERAGE aggregation for {candidate_name}: "
                f"num_chunks={len(all_scores)}, final_score={final_score:.4f}"
            )
            
        elif self.strategy == AggregationStrategy.WEIGHTED:
            # WEIGHTED Strategy: Calculate weighted average based on section importance
            # This prioritizes important sections (e.g., skills weighted more than projects)
            final_score = 0.0
            for section, score in section_final_scores.items():
                # Find matching weight (use partial matching for flexibility)
                # Example: 'experience_1' matches 'experience' weight
                weight = self._get_section_weight(section)
                final_score += score * weight
            
            logger.debug(
                f"WEIGHTED aggregation for {candidate_name}: "
                f"section_scores={section_final_scores}, final_score={final_score:.4f}"
            )
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")
        
        # Ensure score is in valid range [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))
        
        # Create aggregated result with preserved evidence and matched sections
        aggregated = AggregatedCandidateResult(
            candidate_name=candidate_name,
            resume_id=resume_id,
            final_score=final_score,
            section_scores=section_final_scores,
            evidence_chunks=evidence_chunks,
            metadata={
                'num_chunks': len(results),
                'matched_sections': list(matched_sections),
                'aggregation_strategy': self.strategy.value
            }
        )
        
        logger.debug(
            f"Aggregated candidate {candidate_name}: final_score={final_score:.4f}, "
            f"chunks={len(results)}, matched_sections={len(matched_sections)}"
        )
        
        return aggregated
    
    def _get_section_weight(self, section: str) -> float:
        """
        Get the weight for a section.
        
        Performs partial matching to find the appropriate weight.
        For example, 'experience_1' would match 'experience'.
        
        Args:
            section: Section name
            
        Returns:
            Weight for the section (default: 0.0 if not found)
        """
        section_lower = section.lower()
        
        # Try exact match first
        if section_lower in self.section_weights:
            return self.section_weights[section_lower]
        
        # Try partial match
        for weight_section, weight in self.section_weights.items():
            if weight_section in section_lower or section_lower in weight_section:
                return weight
        
        # Default weight if no match found
        return 0.0
    
    def set_section_weight(self, section: str, weight: float) -> None:
        """
        Set the weight for a section.
        
        Args:
            section: Section name
            weight: Weight value (should be between 0.0 and 1.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
        
        self.section_weights[section.lower()] = weight
        
        # Renormalize all weights
        total_weight = sum(self.section_weights.values())
        if total_weight > 0:
            self.section_weights = {
                section: w / total_weight
                for section, w in self.section_weights.items()
            }
        
        logger.info(f"Updated section weight for '{section}': {weight:.2f}")
        logger.info(f"Renormalized weights: {self.section_weights}")
    
    def get_section_weights(self) -> Dict[str, float]:
        """
        Get the current section weights.
        
        Returns:
            Dictionary of section weights
        """
        return self.section_weights.copy()
