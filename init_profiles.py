
from crayon.resources import build_and_cache_profile
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Building LITE profile...")
    path = build_and_cache_profile("lite", prefer_local_only=True)
    print(f"Created: {path}")

if __name__ == "__main__":
    main()
