import time
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional

# In a real project, this would import the actual CrayonVocab
# We use Any/Mock types here to maintain structural integrity without circular deps
from ..core.vocabulary import CrayonVocab 
from .stability import StableVocabularyManager

class AdaptiveVocabularyManager:
    """
    Manages vocabulary adaptation for out-of-distribution text processing.
    
    Implements the control loop defined in Section 8.2:
    dV/dt = eta * grad_V [Performance(V,t) - Complexity(V)][cite: 140].
    """

    def __init__(self, 
                 base_vocab_manager: StableVocabularyManager,
                 core_vocab: Any, # Actually CrayonVocab
                 adaptation_threshold: float = 0.15):
        self.vocab_manager = base_vocab_manager
        self.core_vocab = core_vocab
        self.adaptation_threshold = adaptation_threshold
        
        # Rolling window for effectiveness monitoring [cite: 1106]
        self.unknown_token_rate = deque(maxlen=1000)
        self.candidate_tokens = defaultdict(int)
        
        self.processing_stats = {
            'total_tokens': 0,
            'unknown_tokens': 0,
            'adaptation_events': 0,
            'last_adaptation_time': time.time()
        }

    def tokenize_with_adaptation(self, text: str) -> Tuple[List[int], Dict[str, Any]]:
        """
        Tokenizes text while monitoring for adaptation opportunities[cite: 1120].
        
        Returns:
            Tuple(List[int], MetadataDict)
        """
        # 1. Standard Tokenization (using the core C-extension optimized path)
        # Note: In production, this would call the C extension directly.
        # We simulate the fallback detection logic here.
        tokens = self.core_vocab.tokenize(text)
        
        # 2. Analyze Unknowns
        unk_id = self.core_vocab.unk_token_id
        unknown_count = tokens.count(unk_id)
        total = len(tokens)
        
        # 3. Update Statistics
        self.processing_stats['total_tokens'] += total
        self.processing_stats['unknown_tokens'] += unknown_count
        
        current_rate = unknown_count / total if total > 0 else 0.0
        self.unknown_token_rate.append(current_rate)

        # 4. Extract Candidates (Simplification of Algorithm 3.1)
        # If we have UNKs, we extract the raw substrings causing them
        if unknown_count > 0:
            self._extract_candidates(text, tokens)

        # 5. Trigger Adaptation? [cite: 1157]
        adaptation_metadata = {}
        if self._should_trigger_adaptation():
            adaptation_metadata = self._perform_vocabulary_adaptation()

        return tokens, adaptation_metadata

    def _extract_candidates(self, text: str, tokens: List[int]):
        """
        Heuristic candidate extraction. In reality, we'd map token positions 
        back to text to find the untokenized spans.
        """
        # Logic to extract substrings corresponding to UNK tokens would go here.
        # This is computationally expensive, so it's only done when rate is high.
        pass

    def _should_trigger_adaptation(self) -> bool:
        """
        Determines trigger based on threshold and cooldown[cite: 1157].
        """
        if len(self.unknown_token_rate) < 100:
            return False
        
        recent_rate = sum(self.unknown_token_rate) / len(self.unknown_token_rate)
        
        # Check threshold
        if recent_rate < self.adaptation_threshold:
            return False
            
        # Check cooldown (5 minutes) [cite: 1173]
        if time.time() - self.processing_stats['last_adaptation_time'] < 300:
            return False
            
        return True

    def _rank_candidates_by_utility(self) -> List[Tuple[str, float]]:
        """
        Ranks candidates using the multi-objective utility function[cite: 1224].
        
        Utility = (Compression * 0.4) + (1/Speed * 0.3) + (Coherence * 0.3)
        """
        results = []
        for token, freq in self.candidate_tokens.items():
            if freq < 5: continue # Filter noise
            
            # Proxies for the theoretical values:
            compression_benefit = len(token) * freq  # Bits saved
            speed_impact = 1.0 # Slight penalty for larger vocab
            coherence = 1.0 if token.isalpha() else 0.5
            
            utility = (compression_benefit * 0.4) + \
                      ((1.0/speed_impact) * 0.3) + \
                      (coherence * 0.3)
            
            results.append((token, utility))
            
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _perform_vocabulary_adaptation(self) -> Dict[str, Any]:
        """
        Executes the update[cite: 1179].
        """
        candidates = self._rank_candidates_by_utility()
        # Take top 50
        selected = [c[0] for c in candidates[:50]]
        
        new_ids = self.vocab_manager.add_tokens_incrementally(selected)
        
        # Need to update the core vocabulary (C extension) here
        # self.core_vocab.update(new_ids) 
        
        self.candidate_tokens.clear()
        self.processing_stats['last_adaptation_time'] = time.time()
        self.processing_stats['adaptation_events'] += 1
        
        return {
            "new_tokens": len(new_ids),
            "timestamp": time.time()
        }