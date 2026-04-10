
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.database import DatabaseService

print("Testing DatabaseService initialization...")
try:
    db = DatabaseService()
    print(" DatabaseService initialized")
    
    print("Testing end_session()...")
    db.end_session()
    print(" end_session() called successfully")
    
except AttributeError as e:
    print(f" AttributeError: {e}")
except Exception as e:
    print(f" Exception: {e}")
