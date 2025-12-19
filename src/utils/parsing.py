"""
Parallel Document Parser using LLMWhisperer
Supports: PDF, DOCX, HTML files
"""

from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple
import os
import time

# Load environment variables
load_dotenv()

# Configuration
# These are default fallback paths (will be overridden by function parameters)
DATA_FOLDER = Path(__file__).parent.parent / "data"
PARSED_FOLDER = Path(__file__).parent.parent / "parsed_text"
MAX_WORKERS = 4
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.html', '.htm'}

# Initialize LLMWhisperer client
api_key = os.environ.get("LLMWHISPERER_API_KEY")

if not api_key:
    print("❌ ERROR: LLMWHISPERER_API_KEY not found in environment variables!")
    print("Please set your API key in .env file")
    exit(1)

client = LLMWhispererClientV2(api_key=api_key)


def get_files_from_folder(folder: Path, extensions: Set[str]) -> List[Path]:
    """Get all files with specified extensions from a folder."""
    if not folder.exists():
        return []
    
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
    return files


def get_parsed_filenames(folder: Path) -> Set[str]:
    """Get set of parsed filename stems (without .txt extension)."""
    if not folder.exists():
        return set()
    
    return {f.stem for f in folder.glob("*.txt")}


def parse_document(file_path: Path) -> Tuple[bool, str, str]:
    """
    Parse a document using LLMWhisperer.
    Returns: (success, filename, message)
    """
    try:
        print(f"🔄 Parsing: {file_path.name}")
        
        # LLMWhisperer supports PDF directly, and can handle images
        # For DOCX and HTML, we'll still try as LLMWhisperer might handle them
        result = client.whisper(
            file_path=str(file_path),
            wait_for_completion=True,
            wait_timeout=200
        )
        
        if result["status_code"] == 200 and result["status"] == "processed":
            extracted_text = result["extraction"]["result_text"]
            
            # Save to parsed_text folder
            PARSED_FOLDER.mkdir(parents=True, exist_ok=True)
            output_file = PARSED_FOLDER / f"{file_path.stem}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            return True, file_path.name, f"✓ Successfully parsed ({len(extracted_text)} chars)"
        else:
            return False, file_path.name, f"⚠ Unexpected status: {result['status']}"
            
    except LLMWhispererClientException as e:
        return False, file_path.name, f"❌ LLMWhisperer error: {e}"
    except Exception as e:
        return False, file_path.name, f"❌ Error: {str(e)}"


def sync_folders():
    """
    Synchronize data and parsed_text folders:
    1. Check which files need parsing
    2. Remove orphaned parsed files
    3. Parse new files in parallel
    """
    print("=" * 70)
    print("📁 DOCUMENT PARSER - SYNC PROCESS")
    print("=" * 70)
    
    # Create folders if they don't exist
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    PARSED_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Get all files from data folder
    data_files = get_files_from_folder(DATA_FOLDER, SUPPORTED_EXTENSIONS)
    
    if not data_files:
        print(f"\n⚠ No files found in {DATA_FOLDER}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print(f"\n📊 Found {len(data_files)} file(s) in data folder")
    
    # Get parsed filenames (stems only, without extensions)
    parsed_stems = get_parsed_filenames(PARSED_FOLDER)
    data_stems = {f.stem for f in data_files}
    
    print(f"📊 Found {len(parsed_stems)} parsed file(s) in parsed_text folder")
    
    # === CASE 1: Remove orphaned parsed files ===
    orphaned = parsed_stems - data_stems
    if orphaned:
        print(f"\n🗑️  CASE 1: Removing {len(orphaned)} orphaned file(s):")
        for stem in orphaned:
            orphaned_file = PARSED_FOLDER / f"{stem}.txt"
            orphaned_file.unlink()
            print(f"   ✓ Removed: {stem}.txt")
    else:
        print("\n✓ CASE 1: No orphaned files found")
    
    # === CASE 2: Find files that need parsing ===
    files_to_parse = [f for f in data_files if f.stem not in parsed_stems]
    
    if not files_to_parse:
        print("\n✅ All files are already parsed! Nothing to do.")
        print("=" * 70)
        return
    
    print(f"\n🔄 CASE 2: Found {len(files_to_parse)} file(s) to parse:")
    for f in files_to_parse:
        print(f"   • {f.name}")
    
    # === Parse files in parallel ===
    print(f"\n🚀 Starting parallel processing (max {MAX_WORKERS} workers)...")
    print("-" * 70)
    
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all parsing tasks
        future_to_file = {
            executor.submit(parse_document, file_path): file_path 
            for file_path in files_to_parse
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            success, filename, message = future.result()
            print(f"{message} - {filename}")
            
            if success:
                success_count += 1
            else:
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    # === Summary ===
    print("-" * 70)
    print(f"\n📈 SUMMARY:")
    print(f"   ✓ Successfully parsed: {success_count}")
    if failed_count > 0:
        print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"   ⚡ Average: {elapsed_time/len(files_to_parse):.2f} seconds per file")
    print("=" * 70)


def main():
    """Main entry point."""
    try:
        # Verify API connectivity
        print("🔐 Verifying API connection...")
        usage_info = client.get_usage_info()
        print(f"✓ Connected! API usage: {usage_info}\n")
        
        # Run sync process
        sync_folders()
        
    except LLMWhispererClientException as e:
        print(f"❌ API Connection Error: {e}")
        print("\nPlease check:")
        print("1. Your API key is correct")
        print("2. You have remaining quota")
        print("3. Your subscription is active")
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()