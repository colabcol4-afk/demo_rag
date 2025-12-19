#!/usr/bin/env python3
"""
sync_and_chunk_nim.py

Behavior:
- INPUT_DIR (relative) = ../parsed_text  (expects .txt files)
- OUTPUT_DIR (relative) = ../chunks  (creates subfolders with same stem as input files)
- If all input files already have matching chunk folders -> prints and exits.
- Case 1: If a chunk folder exists but input file missing -> remove chunk folder.
- Case 2: If input file exists but chunk folder missing -> chunk that file and write 1..n.txt
- Uses SemanticChunker FIRST for intelligent chunking, then validates and splits oversized chunks
- Pre-processes large documents to avoid token limit errors during semantic analysis
"""

import os
import shutil
from pathlib import Path
from typing import List

# ---------- Paths relative to src directory ----------
# Get the src directory (utils is in src/utils, so parent gets us to src/)
SRC_DIR = Path(__file__).parent.parent
INPUT_DIR = SRC_DIR / "parsed_text"
OUTPUT_DIR = SRC_DIR / "chunks"
# ---------- NVIDIA config (edit if you want) ----------
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NIM_BASE_URL = None  # or "http://localhost:8080/v1"

# ---------- Chunking configuration ----------
# NVIDIA nv-embedqa-e5-v5 has a 512 token limit for embedding
# For FINAL chunks that go to vector store: ~1600 chars max (400 tokens)
MAX_FINAL_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200

# For pre-processing before semantic chunking: split very large docs first
# This prevents semantic chunker from creating sentences that are too long
MAX_SECTION_CHARS = 8000  # Split large docs into sections first
# -------------------------------------------------

# LangChain + splitters
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    raise RuntimeError("langchain_experimental.text_splitter.SemanticChunker is required. Install compatible langchain packages.") from e

# Try NVIDIA embedder first, else fallback to HuggingFace local embedder
_use_nvidia = False
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # type: ignore
    if NVIDIA_API_KEY:
        os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
        _use_nvidia = True
    elif NIM_BASE_URL:
        _use_nvidia = True
    else:
        _use_nvidia = False
except Exception:
    _use_nvidia = False

if not _use_nvidia:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain_huggingface import HuggingFaceEmbeddings


# ---------- helper functions ----------
def list_input_txt_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.glob("*.txt") if p.is_file()])

def list_chunk_subfolders(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return sorted([p for p in output_dir.iterdir() if p.is_dir()])

def read_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_chunks(chunks: List[str], out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks, start=1):
        with open(out_folder / f"{i}.txt", "w", encoding="utf-8") as fh:
            fh.write(chunk)

def delete_folder(folder: Path):
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        print(f"  ✓ Removed chunk folder: {folder}")

def pre_split_large_document(text: str, max_section_chars: int) -> List[str]:
    """
    Pre-split very large documents into manageable sections.
    This prevents semantic chunker from encountering very long sentences.
    
    Args:
        text: Full document text
        max_section_chars: Maximum characters per section
        
    Returns:
        List of text sections
    """
    if len(text) <= max_section_chars:
        return [text]
    
    # Use RecursiveCharacterTextSplitter to split into sections
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_section_chars,
        chunk_overlap=400,  # Larger overlap for sections
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "]
    )
    sections = splitter.split_text(text)
    return sections

def split_oversized_chunk(chunk: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split an oversized chunk using RecursiveCharacterTextSplitter.
    Preserves semantic boundaries as much as possible.
    
    Args:
        chunk: Text chunk to split
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of smaller chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    return splitter.split_text(chunk)

def validate_and_fix_chunks(chunks: List[str], max_chars: int, overlap: int) -> List[str]:
    """
    Validate chunk sizes and split any that exceed the limit.
    
    Args:
        chunks: List of text chunks from semantic chunking
        max_chars: Maximum characters per chunk
        overlap: Overlap for splitting large chunks
        
    Returns:
        List of validated chunks all within size limits
    """
    validated_chunks = []
    oversized_indices = []
    
    for idx, chunk in enumerate(chunks):
        if len(chunk) <= max_chars:
            validated_chunks.append(chunk)
        else:
            oversized_indices.append(idx + 1)  # 1-indexed for display
            # Split oversized chunk preserving semantic boundaries
            sub_chunks = split_oversized_chunk(chunk, max_chars, overlap)
            validated_chunks.extend(sub_chunks)
    
    if oversized_indices:
        print(f"    ⚠️  Found {len(oversized_indices)} oversized chunk(s): {oversized_indices}")
        print(f"    → Split them into smaller pieces while preserving meaning")
    
    return validated_chunks

# ---------- embedder & chunker creation ----------
def make_embedder_and_chunker():
    """
    Returns (embedder, chunker)
    embedder: object usable by SemanticChunker as 'embeddings' argument
    chunker: SemanticChunker instance
    """
    print("\n🔧 Initializing embedder and chunker...")
    
    if _use_nvidia:
        kwargs = {}
        if NIM_BASE_URL:
            kwargs["base_url"] = NIM_BASE_URL
            print(f"  → Using self-hosted NIM at {NIM_BASE_URL}")
        else:
            kwargs["model"] = "NV-Embed-QA"
            print(f"  → Using hosted NVIDIA NIM with model: nvidia/nv-embedqa-e5-v5")
        
        embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=NVIDIA_API_KEY,
            truncate="END",  # Truncate long sentences during semantic analysis
        )
    else:
        print("  → NVIDIA embedder not configured - using local HuggingFace embeddings")
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("  → Creating SemanticChunker with percentile breakpoint strategy")
    print(f"  → Target final chunk size: {MAX_FINAL_CHUNK_CHARS} chars (~{MAX_FINAL_CHUNK_CHARS//4} tokens)")
    
    # Use SemanticChunker for intelligent semantic boundaries
    chunker = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,  # Tune this: higher = fewer, larger chunks
    )
    
    print("  ✓ Embedder and chunker ready\n")
    return embedder, chunker

# ---------- file processing ----------
def process_and_chunk_file(file_path: Path, chunker, file_idx: int, total_files: int):
    print(f"[{file_idx}/{total_files}] 📄 Processing: {file_path.name}")
    
    # Read file
    print(f"  → Reading file...")
    text = read_text(file_path)
    text_length = len(text)
    print(f"  → File size: {text_length:,} characters")
    
    all_chunks = []
    
    # Pre-split large documents into sections
    if text_length > MAX_SECTION_CHARS:
        print(f"  → Large document detected, pre-splitting into sections...")
        sections = pre_split_large_document(text, MAX_SECTION_CHARS)
        print(f"  → Created {len(sections)} sections for processing")
    else:
        sections = [text]
    
    # Process each section with semantic chunking
    print(f"  → Applying semantic chunking to {len(sections)} section(s)...")
    
    for section_idx, section in enumerate(sections, 1):
        if len(sections) > 1:
            print(f"    → Processing section {section_idx}/{len(sections)} ({len(section):,} chars)...")
        
        try:
            # Apply semantic chunking to this section
            docs = chunker.create_documents([section])
            section_chunks = [d.page_content for d in docs]
            
            if len(sections) > 1:
                print(f"      ✓ Created {len(section_chunks)} semantic chunk(s) from section {section_idx}")
            
            all_chunks.extend(section_chunks)
            
        except Exception as e:
            # If semantic chunking fails for this section, use fallback
            print(f"      ⚠️  Semantic chunking failed for section {section_idx}: {str(e)[:80]}")
            print(f"      → Using fallback for this section only")
            
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_FINAL_CHUNK_CHARS,
                chunk_overlap=CHUNK_OVERLAP_CHARS,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
            )
            section_chunks = fallback_splitter.split_text(section)
            print(f"      ✓ Fallback created {len(section_chunks)} chunk(s)")
            all_chunks.extend(section_chunks)
    
    print(f"  ✓ Semantic chunking complete - created {len(all_chunks)} total chunk(s)")
    
    # Validate and fix any oversized chunks
    print(f"  → Validating chunk sizes (max: {MAX_FINAL_CHUNK_CHARS} chars)...")
    validated_chunks = validate_and_fix_chunks(all_chunks, MAX_FINAL_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
    
    if len(validated_chunks) != len(all_chunks):
        print(f"  ✓ Final chunk count: {len(validated_chunks)} (after fixing oversized chunks)")
    else:
        print(f"  ✓ All semantic chunks already within size limit!")
    
    # Check chunk statistics
    max_chunk_len = max(len(c) for c in validated_chunks) if validated_chunks else 0
    avg_chunk_size = sum(len(c) for c in validated_chunks) // len(validated_chunks) if validated_chunks else 0
    min_chunk_len = min(len(c) for c in validated_chunks) if validated_chunks else 0
    
    print(f"  → Final chunk statistics:")
    print(f"    • Total chunks: {len(validated_chunks)}")
    print(f"    • Average size: {avg_chunk_size:,} chars (~{avg_chunk_size//4} tokens)")
    print(f"    • Size range: {min_chunk_len:,} - {max_chunk_len:,} chars")
    print(f"    • Largest chunk: {max_chunk_len:,} chars (~{max_chunk_len//4} tokens)")
    
    if max_chunk_len > MAX_FINAL_CHUNK_CHARS:
        print(f"    ⚠️  WARNING: Largest chunk still exceeds limit!")
    else:
        print(f"    ✓ All chunks safe for embedding (under {MAX_FINAL_CHUNK_CHARS} char limit)")
    
    # Write chunks
    out_folder = OUTPUT_DIR / file_path.stem
    print(f"  → Writing {len(validated_chunks)} chunks to: {out_folder.name}/")
    write_chunks(validated_chunks, out_folder)
    
    print(f"  ✓ Successfully processed {file_path.name}")
    print()

def main():
    print("=" * 70)
    print("🚀 DOCUMENT CHUNKING UTILITY (Semantic + Size Validation)")
    print("=" * 70)
    
    # Ensure folders exist
    print(f"\n📂 Input directory: {INPUT_DIR.absolute()}")
    print(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Scanning directories...")
    input_files = list_input_txt_files(INPUT_DIR)
    chunk_folders = list_chunk_subfolders(OUTPUT_DIR)

    print(f"  → Found {len(input_files)} input file(s)")
    print(f"  → Found {len(chunk_folders)} existing chunk folder(s)")

    input_stems = {p.stem for p in input_files}
    folder_stems = {p.name for p in chunk_folders}

    # Case: remove chunk folders that don't have matching input file
    extra_folders = folder_stems - input_stems
    if extra_folders:
        print(f"\n🗑️  Found {len(extra_folders)} chunk folder(s) without corresponding input files")
        print("   Removing orphaned folders...")
        for folder_name in sorted(extra_folders):
            folder_path = OUTPUT_DIR / folder_name
            delete_folder(folder_path)
        print(f"  ✓ Removed {len(extra_folders)} folder(s)")

    # Case: find inputs that are not chunked yet
    missing_chunks = input_stems - folder_stems

    if not missing_chunks:
        print("\n✅ All files are already chunked. Nothing to do.")
        print("=" * 70)
        return

    print(f"\n📊 Status:")
    print(f"  • Files to process: {len(missing_chunks)}")
    print(f"  • Already processed: {len(input_stems - missing_chunks)}")
    print(f"  • Total files: {len(input_stems)}")

    # Build embedder and chunker only if we need to chunk something
    _, chunker = make_embedder_and_chunker()

    # Process only the new files
    print("=" * 70)
    print("📝 CHUNKING FILES")
    print("=" * 70)
    
    files_to_process = [f for f in input_files if f.stem in missing_chunks]
    total_files = len(files_to_process)
    
    for idx, file_path in enumerate(files_to_process, start=1):
        process_and_chunk_file(file_path, chunker, idx, total_files)

    print("=" * 70)
    print("✅ CHUNKING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  • Files processed: {total_files}")
    print(f"  • Total chunk folders: {len(list_chunk_subfolders(OUTPUT_DIR))}")
    print(f"  • Output location: {OUTPUT_DIR.absolute()}")
    print(f"  • Primary method: Semantic Chunking (intelligent boundaries)")
    print(f"  • Safety: Oversized chunks automatically split")
    print(f"  • Max chunk size: {MAX_FINAL_CHUNK_CHARS} chars (~{MAX_FINAL_CHUNK_CHARS//4} tokens)")
    print(f"  • Safe for NVIDIA embedding (512 token limit)")
    print()

if __name__ == "__main__":
    main()