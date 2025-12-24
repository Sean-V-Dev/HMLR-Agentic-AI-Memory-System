# Dead Code Analysis - HMLR/hmlr/memory/

Analysis conducted: December 22, 2025

## Summary

After systematically checking all files in `hmlr/memory/` and its subdirectories, the following components are identified as dead code (defined or exported but not actually used in the codebase).

---

## CONFIRMED DEAD CODE

### 1. `metadata_extractor.py`
**Class: `LLMMetadataExtractor`** (Lines 450-720)
- **Status**: DEAD CODE
- **Evidence**: 
  - Defined but never imported anywhere except in its own test code
  - Already marked as removed in ROADMAP.md (Phase 12)
  - System uses inline metadata parsing via `MetadataExtractor`, not separate LLM calls
- **Recommendation**: DELETE entire class

---

### 2. `retrieval/intent_analyzer.py`
**Class: `IntentAnalyzer`** (Lines 26-482)
- **Status**: DEAD CODE
- **Evidence**: 
  - Exported in `retrieval/__init__.py` but never actually imported by any production code
  - Only usage is in its own test code
  - Already marked as removed in ROADMAP.md (Phase 12)
  - Intent analysis is now handled by other components (TheGovernor, LatticeRetrieval)
- **Recommendation**: DELETE entire file

---

### 3. `retrieval/hybrid_search.py`
**Class: `HybridSearchEngine`** (Lines 43-350)
- **Status**: DEAD CODE
- **Evidence**: 
  - Defined but never imported or instantiated anywhere in the codebase
  - Search functionality is handled by `LatticeCrawler` and `EmbeddingStorage` directly
- **Recommendation**: DELETE entire file

---

### 4. `retrieval/context_assembler.py`
**Class: `ContextAssembler`** (Lines 41-345)
- **Status**: DEAD CODE
- **Evidence**: 
  - Defined but never imported by any production code (only self-test usage)
  - Context assembly is handled by `ContextHydrator` and `Hydrator` classes instead
- **Recommendation**: DELETE entire file

---

### 5. `embeddings/chunker.py`
**Class: `SemanticChunker`** (Lines 12-167)
- **Status**: DEAD CODE
- **Evidence**: 
  - Exported in `embeddings/__init__.py` but never actually imported anywhere
  - Chunking is handled by `ChunkEngine` in the `chunking/` subdirectory instead
- **Recommendation**: DELETE entire file

---

### 6. `chunking/chunk_storage.py`
**Class: `ChunkStorage`** (Lines 15-280)
- **Status**: DEAD CODE (possibly dormant)
- **Evidence**: 
  - Exported in `chunking/__init__.py` but only used internally
  - Not imported by any production code outside of chunking module
  - May be intended for future use
- **Recommendation**: KEEP for now (may be used by ChunkEngine internally), but monitor

---

## ACTIVELY USED CODE (Not Dead)

These components are confirmed as actively used in the codebase:

### Core Memory Files
- ✅ `conversation_manager.py` - ConversationManager (used by component_factory, conversation_engine)
- ✅ `dossier_storage.py` - DossierEmbeddingStorage (used by component_factory, dossier_retriever, dossier_governor, manual_gardener)
- ✅ `fact_scrubber.py` - FactScrubber (used by component_factory, conversation_engine, tests)
- ✅ `id_generator.py` - All ID generation functions (used extensively)
- ✅ `models.py` - All data models (core system data structures)
- ✅ `storage.py` - Storage class (used everywhere)
- ✅ `sliding_window_persistence.py` - SlidingWindowPersistence (used by models.py SlidingWindow)
- ✅ `metadata_extractor.py` - MetadataExtractor class (used in docs, but LLMMetadataExtractor is dead)

### Retrieval Components
- ✅ `retrieval/crawler.py` - LatticeCrawler (used by component_factory, lattice.py)
- ✅ `retrieval/context_hydrator.py` - ContextHydrator (used by component_factory)
- ✅ `retrieval/lattice.py` - LatticeRetrieval, TheGovernor (used by component_factory, conversation_engine)
- ✅ `retrieval/hmlr_hydrator.py` - Hydrator (used by component_factory, conversation_engine)
- ✅ `retrieval/dossier_retriever.py` - DossierRetriever (used by component_factory)

### Synthesis Components
- ✅ `synthesis/synthesis_engine.py` - SynthesisManager (used by component_factory)
- ✅ `synthesis/user_profile_manager.py` - UserProfileManager (used by component_factory, scribe, context_hydrator)
- ✅ `synthesis/scribe.py` - Scribe (used by component_factory)
- ✅ `synthesis/dossier_governor.py` - DossierGovernor (used by component_factory)

### Chunking Components
- ✅ `chunking/chunk_engine.py` - ChunkEngine (used by component_factory)

### Embeddings Components
- ✅ `embeddings/embedding_manager.py` - EmbeddingManager, EmbeddingStorage (used by component_factory, crawler)

### Bridge Models
- ✅ `bridge_models/bridge_block.py` - BridgeBlock, BlockStatus, ExitReason, EmbeddingStatus (exported and used)

### Gardener
- ✅ `gardener/manual_gardener.py` - ManualGardener (used by tests, run_gardener.py)

---

## RECOMMENDATIONS

### Immediate Actions (Delete Dead Code)
1. Delete `hmlr/memory/metadata_extractor.py` lines 450-720 (LLMMetadataExtractor class)
2. Delete `hmlr/memory/retrieval/intent_analyzer.py` (entire file)
3. Delete `hmlr/memory/retrieval/hybrid_search.py` (entire file)
4. Delete `hmlr/memory/retrieval/context_assembler.py` (entire file)
5. Delete `hmlr/memory/embeddings/chunker.py` (entire file)
6. Update `hmlr/memory/retrieval/__init__.py` to remove IntentAnalyzer import
7. Update `hmlr/memory/embeddings/__init__.py` to remove SemanticChunker import

### Follow-up Actions
- Review `chunking/chunk_storage.py` for actual usage - if ChunkEngine doesn't use it, delete it
- Update ROADMAP.md to document Phase 12 completions
- Consider archiving deleted code in a `legacy/` folder if there's any future reference value

---

## Analysis Method

1. Read all files in hmlr/memory/ and subdirectories
2. Identified all classes, functions, and exports
3. Used grep_search to find all imports and usages
4. Used list_code_usages to verify actual references
5. Confirmed dead code by checking:
   - Is it imported anywhere?
   - Is it instantiated anywhere?
   - Is it called anywhere?
   - Is it only used in its own test code?
