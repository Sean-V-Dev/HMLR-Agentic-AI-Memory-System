"""
Manual Gardener Runner - Process Bridge Blocks into Long-term Memory

Run this script to manually trigger ManualGardener on a bridge block.
This will chunk, embed, and store the block in the gardened_memory table.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hmlr.core.component_factory import ComponentFactory
from hmlr.memory.gardener.manual_gardener import ManualGardener


def list_bridge_blocks(storage):
    """List active (unprocessed) bridge blocks"""
    # Use storage's get_active_bridge_blocks() method to only show unprocessed blocks
    active_blocks = storage.get_active_bridge_blocks()
    
    if not active_blocks:
        print("[!] No active bridge blocks found (all blocks have been processed)")
        return []
        
    blocks = []
    print(f"\n[*] Found {len(active_blocks)} active bridge block(s):\n")
    
    for i, block_data in enumerate(active_blocks, 1):
        block_id = block_data.get('block_id', 'unknown')
        topic = block_data.get('topic', 'Unknown Topic')
        created_at = block_data.get('created_at', 'Unknown')
        
        # Store tuple of (block_id, topic, created_at)
        blocks.append((block_id, topic, created_at))
        
        print(f"  [{i}] {block_id}")
        print(f"      Topic: {topic}")
        print(f"      Created: {created_at}\n")
    
    return blocks


async def process_single_block(gardener, block_id):
    """Process a single bridge block"""
    print(f"\n[*] Processing bridge block: {block_id}")
    print("="*70 + "\n")
    
    try:
        result = await gardener.process_bridge_block(block_id)
        
        print("\n" + "="*70)
        print(f"[+] GARDENING COMPLETE: {block_id}")
        print("="*70)
        print(f"\nProcessing Summary:")
        print(f"   Turns processed: {result.get('turns_processed', 0)}")
        print(f"   Total chunks: {result.get('total_chunks', 0)}")
        
        # global_tags might be a count or a list
        tags_val = result.get('global_tags', [])
        tag_count = tags_val if isinstance(tags_val, int) else len(tags_val)
        print(f"   Global tags: {tag_count}")
        
        # Show global tags if it's a list
        if isinstance(tags_val, list) and tags_val:
            print(f"\nGlobal Tags:")
            for tag in tags_val:
                print(f"   - [{tag.get('type')}] {tag.get('value')}")
                
        print("\nChunks stored in: gardened_memory table")
        return True
        
    except Exception as e:
        print(f"\n[!] Error processing block {block_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_manual_gardener(target_block_id: str = None):
    """
    Run ManualGardener on bridge blocks.
    
    Args:
        target_block_id: Optional block ID or 'all'. If None, will prompt user.
    """
    print("\n" + "="*70)
    print("MANUAL GARDENER RUNNER")
    print("="*70 + "\n")
    
    # Initialize components
    print("Initializing components...")
    factory = ComponentFactory()
    components = factory.create_all_components()
    
    # Create ManualGardener with dossier support 
    gardener = ManualGardener(
        storage=components.storage,
        embedding_storage=components.embedding_storage,
        llm_client=components.governor.api_client if components.governor else None,
        dossier_governor=components.dossier_governor,
        dossier_storage=components.dossier_storage
    )
    
    if components.dossier_governor:
        print(f"   Dossier governor connected to gardener")
    else:
        print(f"   Dossier system unavailable (gardener will skip dossier creation)")
    
    # List available blocks
    blocks = list_bridge_blocks(components.storage)
    
    if not blocks:
        return
    
    block_ids_to_process = []
    
    # Determine what to process
    if target_block_id:
        if target_block_id.lower() == 'all':
            block_ids_to_process = [b[0] for b in blocks]
        else:
            block_ids_to_process = [target_block_id]
    else:
        # Interactive selection
        while True:
            try:
                print(f"Options:")
                print(f"  1-{len(blocks)}: Select specific block")
                print(f"  'a' or 'all': Process ALL blocks")
                print(f"  'q': Quit")
                
                choice = input("\nEnter choice: ").strip().lower()
                
                if choice == 'q':
                    print("Exiting...")
                    return
                elif choice in ('a', 'all'):
                    block_ids_to_process = [b[0] for b in blocks]
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(blocks):
                        block_ids_to_process = [blocks[idx][0]]
                        break
                    else:
                        print(f"Invalid choice. Please enter 1-{len(blocks)}")
            except ValueError:
                print("Invalid input.")

    # Execute processing
    print(f"\n[*] Starting processing of {len(block_ids_to_process)} block(s)...\n")
    
    success_count = 0
    for bid in block_ids_to_process:
        if await process_single_block(gardener, bid):
            success_count += 1
            
    print("\n" + "="*70)
    print(f"[+] BATCH COMPLETE: {success_count}/{len(block_ids_to_process)} blocks processed successfully")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_manual_gardener(arg))
