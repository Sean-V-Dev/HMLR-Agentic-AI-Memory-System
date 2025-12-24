"""
Pre-Chunking Engine - Creates immutable chunks at ingestion time.

This module implements the hierarchical chunking strategy:
- Sentences: Finest granularity (for fact linking)
- Paragraphs: Mid-level granularity (for context)
- Bridge Blocks: Coarsest granularity (for topics)

All chunks receive immutable IDs at creation time to prevent ID drift.
"""
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    Represents a single chunk of text with immutable ID and metadata.
    
    Attributes:
        chunk_id: Immutable identifier (e.g., sent_20251202_143005_abc123)
        chunk_type: 'sentence' | 'paragraph' | 'bridge_block'
        text_verbatim: Original text (preserved for embeddings and display)
        lexical_filters: Keywords extracted for hybrid search (stop words removed)
        parent_chunk_id: Link to parent chunk (sentences → paragraph → block)
        turn_id: Which conversation turn this came from
        span_id: Which conversation span this belongs to
        token_count: Cached token count (approximate)
    """
    chunk_id: str
    chunk_type: str  # 'sentence' | 'paragraph' | 'bridge_block'
    text_verbatim: str
    lexical_filters: List[str]
    parent_chunk_id: Optional[str] = None
    turn_id: Optional[str] = None
    span_id: Optional[str] = None
    token_count: int = 0
    metadata: Dict = field(default_factory=dict)


class ChunkEngine:
    """
    Splits text into hierarchical chunks with immutable IDs.
    
    Key Design Principles:
    1. Chunk at INGESTION time (not Gardener time) - prevents ID drift
    2. Preserve VERBATIM text - modern embeddings need full context
    3. Extract KEYWORDS separately - for lexical (BM25-style) search
    4. Assign IMMUTABLE IDs - chunks never change, even after consolidation
    """
    
    # Stop words to remove from keyword extraction (NOT from verbatim text!)
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
        'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'can', 'my', 'your', 'his', 'her', 'their',
        'our', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'over', 
        'under', 'through', 'into', 'out', 'up', 'down', 'off', 'about'
    }
    
    def chunk_turn(self, text: str, turn_id: str, span_id: str) -> List[Chunk]:
        """
        Chunk a single turn into sentences and paragraphs.
        
        Args:
            text: The raw text to chunk
            turn_id: Unique identifier for this turn
            span_id: Unique identifier for the conversation span
        
        Returns:
            List of Chunk objects (sentences + paragraphs)
            
        Example:
            >>> engine = ChunkEngine()
            >>> chunks = engine.chunk_turn(
            ...     "Hello world. This is a test.\\n\\nNew paragraph here.",
            ...     turn_id="turn_123",
            ...     span_id="span_456"
            ... )
            >>> len(chunks)  # 2 sentences + 2 paragraphs
            4
        """
        if not text or not text.strip():
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Split into paragraphs (by double newline or logical breaks)
        paragraphs = self._split_paragraphs(text)
        
        all_chunks = []
        
        for para_idx, para_text in enumerate(paragraphs):
            # Create paragraph chunk ID
            para_chunk_id = f"para_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Step 2: Split paragraph into sentences
            sentences = self._split_sentences(para_text)
            sentence_chunks = []
            
            for sent_idx, sent_text in enumerate(sentences):
                # Create sentence chunk
                sent_chunk_id = f"sent_{timestamp}_{uuid.uuid4().hex[:8]}"
                
                sent_chunk = Chunk(
                    chunk_id=sent_chunk_id,
                    chunk_type='sentence',
                    text_verbatim=sent_text.strip(),
                    lexical_filters=self._extract_keywords(sent_text),
                    parent_chunk_id=para_chunk_id,
                    turn_id=turn_id,
                    span_id=span_id,
                    token_count=self._estimate_tokens(sent_text),
                    metadata={'para_idx': para_idx, 'sent_idx': sent_idx}
                )
                sentence_chunks.append(sent_chunk)
                all_chunks.append(sent_chunk)
            
            # Create paragraph chunk (contains sentence references)
            para_chunk = Chunk(
                chunk_id=para_chunk_id,
                chunk_type='paragraph',
                text_verbatim=para_text.strip(),
                lexical_filters=self._extract_keywords(para_text),
                parent_chunk_id=None,  
                turn_id=turn_id,
                span_id=span_id,
                token_count=self._estimate_tokens(para_text),
                metadata={
                    'para_idx': para_idx,
                    'sentence_count': len(sentence_chunks),
                    'child_chunks': [c.chunk_id for c in sentence_chunks]
                }
            )
            all_chunks.append(para_chunk)
        
        return all_chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Rules:
        - Split on double newline (\\n\\n)
        - Split long paragraphs (>800 chars) on sentence boundaries
        - Preserve natural paragraph breaks
        
        Args:
            text: Raw text to split
            
        Returns:
            List of paragraph strings
        """
        # Split by double newline
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) > 800:  # Split long paragraphs
                # Try to split on sentence boundaries
                sentences = self._split_sentences(para)
                current_para = []
                current_len = 0
                
                for sent in sentences:
                    if current_len + len(sent) > 600 and current_para:
                        result.append(' '.join(current_para))
                        current_para = [sent]
                        current_len = len(sent)
                    else:
                        current_para.append(sent)
                        current_len += len(sent)
                
                if current_para:
                    result.append(' '.join(current_para))
            else:
                result.append(para)
        
        return [p.strip() for p in result if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Rules:
        - Split on .!? followed by space and capital letter
        - Handle common abbreviations (Dr., Mr., etc.)
        - Preserve sentence integrity
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentence strings
        """
        # Handle common abbreviations to avoid false splits
        text = text.replace("Dr.", "Dr")
        text = text.replace("Mr.", "Mr")
        text = text.replace("Mrs.", "Mrs")
        text = text.replace("Ms.", "Ms")
        text = text.replace("etc.", "etc")
        text = text.replace("i.e.", "ie")
        text = text.replace("e.g.", "eg")
        
        # Split on .!? followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [
            s.replace("Dr", "Dr.")
             .replace("Mr", "Mr.")
             .replace("Mrs", "Mrs.")
             .replace("Ms", "Ms.")
             .replace("etc", "etc.")
             .replace("ie", "i.e.")
             .replace("eg", "e.g.")
             for s in sentences
        ]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords by removing stop words.
        
        CRITICAL: This does NOT modify the verbatim text!
        - Verbatim text preserved for embeddings (modern models need context)
        - Keywords extracted for lexical/BM25-style search
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords (deduplicated, stop words removed)
        """
        # Lowercase and tokenize (preserve numbers, underscores)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words and very short tokens
        keywords = [
            t for t in tokens 
            if t not in self.STOP_WORDS and len(t) > 2
        ]
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate (1 token ≈ 4 characters for English).
        
        Args:
            text: Text to estimate token count for
            
        Returns:
            Approximate token count
        """
        return max(1, len(text) // 4)
    
    def merge_chunks(self, chunks: List[Chunk]) -> str:
        """
        Reconstruct original text from chunks (for testing/validation).
        
        Args:
            chunks: List of chunks to merge
            
        Returns:
            Reconstructed text
        """
        # Group by paragraph
        paragraphs = {}
        for chunk in chunks:
            if chunk.chunk_type == 'paragraph':
                para_idx = chunk.metadata.get('para_idx', 0)
                paragraphs[para_idx] = chunk.text_verbatim
        
        # Join paragraphs with double newline
        return '\n\n'.join(paragraphs[i] for i in sorted(paragraphs.keys()))
