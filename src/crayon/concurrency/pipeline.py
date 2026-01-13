import time
import threading
import queue
from collections import deque
from typing import Any, List, Tuple, Optional
from ..core.vocabulary import CrayonVocab
from ..unicode.normalizer import unicode_normalize_nfc_optimized

class PipelineTokenizer:
    """
    Multi-stage pipeline tokenizer achieving high throughput through parallel execution.
    
    Architecture (Section 7.2) [cite: 720-724]:
    1. Input preprocessing & normalization
    2. Vocabulary Lookup & Longest-match
    3. Token ID assignment & Formatting
    """

    def __init__(self, vocab: CrayonVocab, pipeline_depth: int = 4):
        self.vocab = vocab
        self.pipeline_depth = pipeline_depth
        
        # Inter-stage communication queues with backpressure [cite: 730-739]
        # Size = depth * 2 to absorb bursty traffic
        q_size = pipeline_depth * 2
        self.input_queue: queue.Queue = queue.Queue(maxsize=q_size)
        self.normalized_queue: queue.Queue = queue.Queue(maxsize=q_size)
        self.tokenized_queue: queue.Queue = queue.Queue(maxsize=q_size)
        # Output queue is read by external consumers via get_result()
        self.output_queue: queue.Queue = queue.Queue(maxsize=q_size)
        
        # Pipeline stage threads [cite: 741-743]
        # Note: Only 3 stages - output_queue is consumed by user via get_result()
        self.stages: List[threading.Thread] = [
            threading.Thread(target=self._normalize_stage, name="Stage-Normalize", daemon=True),
            threading.Thread(target=self._tokenize_stage, name="Stage-Tokenize", daemon=True),
            threading.Thread(target=self._format_stage, name="Stage-Format", daemon=True),
        ]
        
        # Performance monitoring [cite: 745]
        self.stage_timings: List[deque] = [deque(maxlen=1000) for _ in range(3)]
        self.running = False

    def start_pipeline(self) -> None:
        """Initialize and start all pipeline stages."""
        self.running = True
        for stage in self.stages:
            stage.start()

    def stop_pipeline(self) -> None:
        """Graceful shutdown signal."""
        self.running = False
        # Send sentinel to unblock input
        try:
            self.input_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

    def _normalize_stage(self) -> None:
        """Stage 1: Input preprocessing and Unicode normalization[cite: 752]."""
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None: break # Shutdown
                
                text_id, text = item
                start_time = time.perf_counter()
                
                # Normalize Unicode (CPU intensive)
                normalized_text = unicode_normalize_nfc_optimized(text)
                
                self.stage_timings[0].append(time.perf_counter() - start_time)
                self.normalized_queue.put((text_id, normalized_text))
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pipeline Error (Normalize): {e}")

    def _tokenize_stage(self) -> None:
        """Stage 2: Core tokenization with vocabulary lookup[cite: 769]."""
        while self.running:
            try:
                item = self.normalized_queue.get(timeout=0.1)
                if item is None: break
                
                text_id, normalized_text = item
                start_time = time.perf_counter()
                
                # High-speed tokenization
                # In production, this calls the C-extension via the vocab object
                tokens = self.vocab.tokenize(normalized_text)
                
                self.stage_timings[1].append(time.perf_counter() - start_time)
                self.tokenized_queue.put((text_id, tokens))
                self.normalized_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pipeline Error (Tokenize): {e}")

    def _format_stage(self) -> None:
        """Stage 3: Token formatting and result delivery[cite: 786]."""
        while self.running:
            try:
                item = self.tokenized_queue.get(timeout=0.1)
                if item is None: break
                
                text_id, tokens = item
                start_time = time.perf_counter()
                
                # Format output (e.g., adding special tokens, truncating)
                formatted_result = {
                    "id": text_id,
                    "input_ids": tokens,
                    "length": len(tokens)
                }
                
                self.stage_timings[2].append(time.perf_counter() - start_time)
                # Put result in output queue for external consumers
                self.output_queue.put(formatted_result)
                self.tokenized_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pipeline Error (Format): {e}")

    def submit_text(self, text_id: str, text: str) -> None:
        """Entry point for the pipeline."""
        self.input_queue.put((text_id, text))

    def get_result(self, timeout: float = 10.0) -> Any:
        """Blocking retrieval of next result with timeout."""
        return self.output_queue.get(timeout=timeout)