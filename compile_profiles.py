
from pathlib import Path
import json
import logging
import sys
import time

# Add src to sys.path
sys.path.append("src")
from crayon.c_ext.dat_builder import DATBuilder
from crayon.core.profiles import PROFILES

logging.basicConfig(level=logging.INFO)

def compile_all():
    cache_dir = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("XERV CRAYON V2.1: OFFLINE DAT COMPILER")
    print("="*80)
    print(f"Target Directory: {cache_dir}")
    print("-" * 80)
    
    for name, profile in PROFILES.items():
        # Source JSON (Versioned)
        json_filename = f"vocab_{name}_{profile.version}.json"
        json_path = cache_dir / json_filename
        
        # Target DAT (Canonical for Engine V2)
        dat_path = cache_dir / f"vocab_{name}.dat"
        
        if not json_path.exists():
            print(f"[-] SKIPPING {name}: {json_path} not found.")
            # Trigger build_and_cache if needed? 
            # For now we assume they exist or user runs build_all_profiles.py first.
            continue
            
        print(f"[+] Compiling {name.upper()}...")
        try:
            start = time.time()
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                vocab = data
            elif isinstance(data, dict):
                # Sort by value
                vocab = [k for k, v in sorted(data.items(), key=lambda x: x[1])]
            
            # Use V2.1 Builder
            builder = DATBuilder()
            builder.build(vocab)
            builder.save(str(dat_path))
            end = time.time()
            
            print(f"    -> Success! ({end-start:.2f}s)")
            print(f"    -> Output: {dat_path} ({dat_path.stat().st_size/1024:.1f} KB)")
            
        except Exception as e:
            print(f"[!] FAILED {name}: {e}")

if __name__ == "__main__":
    compile_all()
