import sys
import os
sys.path.append(r'D:\Projects\om\backend')

try:
    from pinecone import Pinecone, ServerlessSpec
    print("✓ Pinecone imports work")
    
    from src.config import PINECONE_API_KEY
    print("✓ Config imports work")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✓ Pinecone client initialized")
    
    print("All imports and initialization successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
