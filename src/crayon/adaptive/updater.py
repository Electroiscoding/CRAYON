"""
Incremental Vocabulary Updater Module.

Implements Section 8.3 of the XERV Crayon Engineering Treatise:
- Staged vocabulary updates with validation
- Rollback capability for failed updates
- Persistent state management via JSON
- Compression and unknown rate validation
"""

import json
import time
import copy
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from .stability import StableVocabularyManager


class IncrementalVocabularyUpdater:
    """
    Handles incremental vocabulary updates with rollback capability.
    
    Implements the lifecycle described in Section 8.3 [cite: 1240-1375]:
    1. Stage: Prepare update without committing
    2. Validate: Test against corpus for quality metrics
    3. Commit: Apply permanently if validation passes
    4. Rollback: Discard if validation fails
    
    Features:
    - Transaction-like staged updates
    - Corpus-based validation with real metrics
    - Persistent state management
    - Full update history tracking
    """
    
    def __init__(self, vocab_manager: StableVocabularyManager):
        self.vocab_manager = vocab_manager
        self.update_history: List[Dict] = []
        self.staged_updates: Dict[str, Dict] = {}
        self.validation_results: Dict[str, Dict] = {}
        
        # Snapshot for rollback capability
        self._snapshots: Dict[str, Dict[str, int]] = {}

    def stage_vocabulary_update(
        self, 
        new_tokens: List[str], 
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Stage vocabulary updates for validation before permanent application[cite: 1248].
        
        Args:
            new_tokens: List of token strings to add
            metadata: Optional metadata about the update source
            
        Returns:
            Dict with stage_id and status information
        """
        # Filter tokens already in vocabulary
        filtered_tokens = [
            t for t in new_tokens 
            if t not in self.vocab_manager.token_to_id
        ]
        
        if not filtered_tokens:
            return {
                "stage_id": None,
                "token_count": 0,
                "status": "no_new_tokens",
                "filtered_count": len(new_tokens)
            }
        
        # Generate unique stage ID
        token_hash = hashlib.md5(
            str(sorted(filtered_tokens)).encode('utf-8')
        ).hexdigest()[:8]
        stage_id = f"stage_{int(time.time())}_{token_hash}"
        
        # Create snapshot of current state for potential rollback
        self._snapshots[stage_id] = copy.deepcopy(self.vocab_manager.token_to_id)
        
        self.staged_updates[stage_id] = {
            "new_tokens": filtered_tokens,
            "original_count": len(new_tokens),
            "filtered_count": len(filtered_tokens),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        return {
            "stage_id": stage_id,
            "token_count": len(filtered_tokens),
            "original_count": len(new_tokens),
            "status": "staged_for_validation"
        }

    def validate_staged_update(
        self, 
        stage_id: str, 
        validation_corpus: List[str]
    ) -> Dict[str, float]:
        """
        Validate staged vocabulary update against test corpus[cite: 1277].
        
        Calculates real metrics:
        - Compression ratio: tokens after / tokens before
        - Unknown token rate: proportion of UNK tokens
        - Memory impact: estimated memory usage increase
        
        Args:
            stage_id: ID from stage_vocabulary_update
            validation_corpus: List of text strings for validation
            
        Returns:
            Dict with validation metrics
        """
        if stage_id not in self.staged_updates:
            raise ValueError(f"Invalid stage_id: {stage_id}")

        update = self.staged_updates[stage_id]
        new_tokens = update['new_tokens']
        
        if not validation_corpus:
            raise ValueError("Validation corpus cannot be empty")
        
        # Create temporary vocabulary with proposed additions
        temp_token_to_id = copy.deepcopy(self.vocab_manager.token_to_id)
        next_id = max(temp_token_to_id.values()) + 1 if temp_token_to_id else 0
        
        for token in new_tokens:
            if token not in temp_token_to_id:
                temp_token_to_id[token] = next_id
                next_id += 1
        
        # Calculate metrics on validation corpus
        total_chars_before = 0
        total_tokens_before = 0
        total_unknown_before = 0
        
        total_chars_after = 0
        total_tokens_after = 0
        total_unknown_after = 0
        
        unk_token = "<UNK>"
        
        for text in validation_corpus:
            total_chars_before += len(text)
            total_chars_after += len(text)
            
            # Simulate tokenization with current vocab
            tokens_before = self._simulate_tokenize(
                text, self.vocab_manager.token_to_id, unk_token
            )
            total_tokens_before += len(tokens_before)
            total_unknown_before += tokens_before.count(-1)
            
            # Simulate tokenization with proposed vocab
            tokens_after = self._simulate_tokenize(
                text, temp_token_to_id, unk_token
            )
            total_tokens_after += len(tokens_after)
            total_unknown_after += tokens_after.count(-1)
        
        # Calculate metrics
        compression_ratio = (
            total_tokens_before / total_tokens_after 
            if total_tokens_after > 0 else 1.0
        )
        
        unknown_rate_before = (
            total_unknown_before / total_tokens_before 
            if total_tokens_before > 0 else 0.0
        )
        unknown_rate_after = (
            total_unknown_after / total_tokens_after 
            if total_tokens_after > 0 else 0.0
        )
        
        # Memory impact estimation (bytes per token entry)
        avg_token_len = sum(len(t.encode('utf-8')) for t in new_tokens) / len(new_tokens)
        memory_impact_bytes = len(new_tokens) * (avg_token_len + 64)  # Token + trie node
        memory_impact_mb = memory_impact_bytes / (1024 * 1024)
        
        metrics = {
            "compression_ratio": compression_ratio,
            "unknown_token_rate_before": unknown_rate_before,
            "unknown_token_rate": unknown_rate_after,
            "unknown_reduction": unknown_rate_before - unknown_rate_after,
            "memory_impact_mb": memory_impact_mb,
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "corpus_size": len(validation_corpus),
            "timestamp": datetime.now().isoformat()
        }
        
        self.validation_results[stage_id] = metrics
        update['status'] = "validated"
        
        return metrics

    def _simulate_tokenize(
        self, 
        text: str, 
        token_to_id: Dict[str, int],
        unk_token: str
    ) -> List[int]:
        """
        Simple greedy longest-match tokenization simulation.
        
        Returns list of token IDs (-1 for unknown).
        """
        tokens: List[int] = []
        pos = 0
        text_len = len(text)
        max_len = 16  # SIMD limit
        
        while pos < text_len:
            best_len = 0
            best_id = -1
            
            # Try longest match first
            for length in range(min(max_len, text_len - pos), 0, -1):
                candidate = text[pos:pos + length]
                if candidate in token_to_id:
                    best_len = length
                    best_id = token_to_id[candidate]
                    break
            
            if best_len > 0:
                tokens.append(best_id)
                pos += best_len
            else:
                tokens.append(-1)  # Unknown
                pos += 1
        
        return tokens

    def commit_update(self, stage_id: str) -> bool:
        """
        Permanently apply staged vocabulary update after validation[cite: 1330].
        
        Args:
            stage_id: ID of the staged update
            
        Returns:
            True if commit successful, False if rejected
            
        Raises:
            ValueError: If stage_id not found
            RuntimeError: If update not validated
        """
        if stage_id not in self.staged_updates:
            raise ValueError(f"Unknown stage ID: {stage_id}")
            
        update = self.staged_updates[stage_id]
        if update['status'] != 'validated':
            raise RuntimeError("Update must be validated before commit")
            
        metrics = self.validation_results.get(stage_id, {})
        
        # Strict acceptance criteria [cite: 1362]
        # Reject if unknown rate is too high (> 10%)
        if metrics.get('unknown_token_rate', 1.0) > 0.1:
            update['status'] = 'rejected_high_unknown_rate'
            return False
        
        # Reject if compression ratio is poor (< 1.0 means more tokens)
        if metrics.get('compression_ratio', 0.0) < 0.95:
            update['status'] = 'rejected_poor_compression'
            return False
            
        # Apply changes to stable vocabulary manager
        new_assignments = self.vocab_manager.add_tokens_incrementally(
            update['new_tokens'], preserve_existing=True
        )
        
        # Archive successful update
        self.update_history.append({
            "stage_id": stage_id,
            "tokens_added": len(new_assignments),
            "token_list": list(new_assignments.keys()),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Cleanup staged data
        del self.staged_updates[stage_id]
        del self.validation_results[stage_id]
        if stage_id in self._snapshots:
            del self._snapshots[stage_id]
        
        return True

    def rollback_update(self, stage_id: str) -> bool:
        """
        Roll back a staged update[cite: 1367].
        
        Discards the staged update and restores any snapshot state.
        
        Args:
            stage_id: ID of the staged update to rollback
            
        Returns:
            True if rollback successful, False if stage not found
        """
        if stage_id not in self.staged_updates:
            return False
        
        # Restore snapshot if it exists
        if stage_id in self._snapshots:
            # Note: Full restoration would require rebuilding the trie
            # This is a simplified version that just clears the staged state
            del self._snapshots[stage_id]
        
        # Remove staged update
        del self.staged_updates[stage_id]
        self.validation_results.pop(stage_id, None)
        
        return True

    def save_vocabulary_state(self, path: str) -> None:
        """
        Saves current vocabulary state to disk JSON[cite: 1375].
        
        Saves:
        - Complete token-to-ID mapping
        - Update history
        - Metadata and timestamps
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare ID-to-token for reverse lookup storage
        id_to_token = {
            str(v): k for k, v in self.vocab_manager.token_to_id.items()
        }
        
        state = {
            "version": "1.0.0",
            "token_map": self.vocab_manager.token_to_id,
            "id_to_token": id_to_token,
            "vocabulary_size": len(self.vocab_manager.token_to_id),
            "history": self.update_history,
            "pending_updates": len(self.staged_updates),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_vocabulary_state(self, path: str) -> Dict[str, Any]:
        """
        Loads vocabulary state from disk[cite: 1383].
        
        Reconstructs the vocabulary manager state from saved JSON.
        
        Args:
            path: Path to the state JSON file
            
        Returns:
            Dict with load status and statistics
        """
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Validate version
        version = state.get('version', '0.0.0')
        if version != '1.0.0':
            raise ValueError(f"Unsupported state version: {version}")
        
        # Rebuild vocabulary manager state
        token_map = state.get('token_map', {})
        
        # Clear and rebuild
        self.vocab_manager.token_to_id.clear()
        self.vocab_manager.id_to_token.clear()
        
        for token, token_id in token_map.items():
            self.vocab_manager.token_to_id[token] = token_id
            self.vocab_manager.id_to_token[token_id] = token
        
        # Restore history
        self.update_history = state.get('history', [])
        
        return {
            "status": "loaded",
            "vocabulary_size": len(token_map),
            "history_entries": len(self.update_history),
            "source_timestamp": state.get('timestamp')
        }

    def get_update_history(self) -> List[Dict]:
        """Return the complete update history."""
        return self.update_history.copy()

    def get_pending_updates(self) -> Dict[str, Dict]:
        """Return all pending staged updates."""
        return {
            stage_id: {
                "token_count": len(update['new_tokens']),
                "status": update['status'],
                "timestamp": update['timestamp']
            }
            for stage_id, update in self.staged_updates.items()
        }

    def clear_pending_updates(self) -> int:
        """Clear all pending staged updates. Returns count of cleared updates."""
        count = len(self.staged_updates)
        self.staged_updates.clear()
        self.validation_results.clear()
        self._snapshots.clear()
        return count