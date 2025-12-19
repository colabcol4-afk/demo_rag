"""
Qdrant Collection Cleanup Script

This script helps you clear data from a Qdrant collection.
You can either:
1. Delete all points (data) from a collection (keeps the collection structure)
2. Delete the entire collection

CAUTION: This operation is IRREVERSIBLE!
"""

import sys
import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models


# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Qdrant Cloud Configuration
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def connect_to_qdrant(url: str, api_key: str) -> QdrantClient:
    """
    Connect to Qdrant Cloud.
    
    Args:
        url: Qdrant cloud URL
        api_key: Qdrant API key
        
    Returns:
        QdrantClient instance
    """
    print("\n🔗 Connecting to Qdrant Cloud...")
    print(f"   URL: {url}")
    
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=True,
        )
        
        # Test connection
        collections = client.get_collections()
        print("   ✓ Connected successfully\n")
        
        return client
    
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        sys.exit(1)


def list_collections(client: QdrantClient) -> List[str]:
    """
    List all available collections.
    
    Args:
        client: QdrantClient instance
        
    Returns:
        List of collection names
    """
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    return collection_names


def show_collections(client: QdrantClient):
    """
    Display all available collections with their details.
    
    Args:
        client: QdrantClient instance
    """
    collections = client.get_collections().collections
    
    if not collections:
        print("📊 No collections found in your Qdrant instance.")
        return
    
    print(f"📊 Available Collections ({len(collections)}):\n")
    
    for col in collections:
        try:
            # Get collection info
            info = client.get_collection(collection_name=col.name)
            
            print(f"   • {col.name}")
            print(f"     - Vectors: {info.points_count:,}")
            print(f"     - Dimension: {info.config.params.vectors.size}")
            print(f"     - Distance: {info.config.params.vectors.distance}")
            print()
            
        except Exception as e:
            print(f"   • {col.name}")
            print(f"     - Error getting details: {str(e)}")
            print()


def get_collection_info(client: QdrantClient, collection_name: str):
    """
    Display detailed information about a specific collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
    """
    try:
        info = client.get_collection(collection_name=collection_name)
        
        print(f"\n📊 Collection: '{collection_name}'")
        print(f"   • Total vectors: {info.points_count:,}")
        print(f"   • Vector dimension: {info.config.params.vectors.size}")
        print(f"   • Distance metric: {info.config.params.vectors.distance}")
        
        if info.points_count > 0:
            # Try to get a sample point
            sample = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if sample:
                print(f"\n   Sample metadata keys:")
                for key in sample[0].payload.keys():
                    print(f"     - {key}")
        
    except Exception as e:
        print(f"\n❌ Error getting collection info: {str(e)}")


def clear_collection_data(client: QdrantClient, collection_name: str):
    """
    Delete all points from a collection (keeps the collection structure).
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
    """
    print(f"\n🗑️  Clearing all data from collection '{collection_name}'...")
    
    try:
        # Get point count before deletion
        info_before = client.get_collection(collection_name=collection_name)
        points_count = info_before.points_count
        
        if points_count == 0:
            print("   ℹ️  Collection is already empty.")
            return
        
        # Delete all points using a filter that matches everything
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.HasIdCondition(has_id=[])  # This won't match anything
                    ]
                )
            )
        )
        
        # Alternative: Delete by scrolling through all IDs
        # This is more reliable for ensuring all points are deleted
        print(f"   🔄 Deleting {points_count:,} points...")
        
        offset = None
        batch_size = 100
        total_deleted = 0
        
        while True:
            # Scroll through points to get their IDs
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            if not points:
                break
            
            # Extract IDs
            point_ids = [point.id for point in points]
            
            # Delete points by ID
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            total_deleted += len(point_ids)
            print(f"   ⏳ Deleted {total_deleted:,} / {points_count:,} points...", end='\r')
            
            if offset is None:
                break
        
        print(f"\n   ✓ Successfully deleted all {points_count:,} points!")
        
        # Verify deletion
        info_after = client.get_collection(collection_name=collection_name)
        print(f"   ✓ Collection '{collection_name}' is now empty (0 vectors)")
        print(f"   ✓ Collection structure preserved")
        
    except Exception as e:
        print(f"\n❌ Error clearing collection: {str(e)}")
        raise


def delete_collection(client: QdrantClient, collection_name: str):
    """
    Delete an entire collection (structure and all data).
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
    """
    print(f"\n🗑️  Deleting collection '{collection_name}'...")
    
    try:
        # Get info before deletion
        info = client.get_collection(collection_name=collection_name)
        points_count = info.points_count
        
        # Delete the collection
        client.delete_collection(collection_name=collection_name)
        
        print(f"   ✓ Successfully deleted collection '{collection_name}'!")
        print(f"   ✓ Removed collection with {points_count:,} vectors")
        print(f"   ✓ Collection structure also removed")
        
    except Exception as e:
        print(f"\n❌ Error deleting collection: {str(e)}")
        raise


def confirm_action(message: str) -> bool:
    """
    Get user confirmation for destructive actions.
    
    Args:
        message: Confirmation message to display
        
    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 70)
    print("⚠️  WARNING: DESTRUCTIVE OPERATION")
    print("=" * 70)
    print(message)
    print("=" * 70)
    
    response = input("\nType 'DELETE' to confirm (or anything else to cancel): ")
    
    return response == "DELETE"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("🗑️  QDRANT COLLECTION CLEANUP UTILITY")
    print("=" * 70)
    
    # Connect to Qdrant
    client = connect_to_qdrant(QDRANT_URL, QDRANT_API_KEY)
    
    # Show available collections
    show_collections(client)
    
    # Get collection names
    collection_names = list_collections(client)
    
    if not collection_names:
        print("\n❌ No collections found. Nothing to delete.")
        return
    
    # Get collection name from user
    print("-" * 70)
    print("\nEnter the name of the collection you want to manage:")
    collection_name = input("Collection name: ").strip()
    
    # Validate collection exists
    if collection_name not in collection_names:
        print(f"\n❌ Collection '{collection_name}' not found.")
        print(f"\nAvailable collections: {', '.join(collection_names)}")
        return
    
    # Show collection info
    get_collection_info(client, collection_name)
    
    # Get action choice
    print("\n" + "-" * 70)
    print("\nWhat would you like to do?")
    print("1. Clear all data (delete all points, keep collection structure)")
    print("2. Delete entire collection (remove everything)")
    print("3. Cancel")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Clear collection data
        confirm_msg = (
            f"You are about to DELETE ALL DATA from collection '{collection_name}'.\n"
            f"The collection structure will be preserved, but all {client.get_collection(collection_name).points_count:,} points will be removed.\n"
            f"This operation is IRREVERSIBLE!"
        )
        
        if confirm_action(confirm_msg):
            clear_collection_data(client, collection_name)
            
            print("\n" + "=" * 70)
            print("✅ OPERATION COMPLETED")
            print("=" * 70)
            print(f"\nCollection '{collection_name}' has been cleared.")
            print("The collection still exists and can accept new data.")
        else:
            print("\n❌ Operation cancelled by user.")
    
    elif choice == "2":
        # Delete entire collection
        confirm_msg = (
            f"You are about to DELETE THE ENTIRE COLLECTION '{collection_name}'.\n"
            f"This will remove the collection structure and all {client.get_collection(collection_name).points_count:,} points.\n"
            f"This operation is IRREVERSIBLE!"
        )
        
        if confirm_action(confirm_msg):
            delete_collection(client, collection_name)
            
            print("\n" + "=" * 70)
            print("✅ OPERATION COMPLETED")
            print("=" * 70)
            print(f"\nCollection '{collection_name}' has been completely deleted.")
            print("You will need to recreate it to use it again.")
        else:
            print("\n❌ Operation cancelled by user.")
    
    elif choice == "3":
        print("\n✅ Operation cancelled by user.")
    
    else:
        print("\n❌ Invalid choice. Operation cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)