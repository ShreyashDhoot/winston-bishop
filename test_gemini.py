#!/usr/bin/env python3
"""
test_gemini.py
──────────────
Quick diagnostic script to verify local Qwen checker functionality.
"""

import sys
import os

# Add required paths
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./JailbreakDiffusionBench"))

try:
    from jailbreak_diffusion.judger.pre_checker.gemini import GeminiChecker
    print("✅ Successfully imported GeminiChecker.")
except ImportError as e:
    print(f"❌ Failed to import GeminiChecker: {e}")
    sys.exit(1)

def main():
    print("Initializing Qwen-backed GeminiChecker...")
    try:
        checker = GeminiChecker(prompt_file="simple")
        print(f"✅ Checker initialized successfully. Using model: {checker.model}")
    except Exception as e:
        print(f"❌ Failed to initialize GeminiChecker: {e}")
        sys.exit(1)

    test_prompts = [
        "A peaceful garden with blooming cherry blossoms and a small pond.",
        "A highly toxic and violent prompt containing illegal hate speech."
    ]

    print("\n--- Running API Diagnosis ---")
    for prompt in test_prompts:
        print(f"\nChecking Prompt: '{prompt}'")
        try:
            res = checker.check(prompt)
            print(f"Response: {res}")
            if isinstance(res, dict) and "flagged" in res:
                status = "🚨 UNSAFE" if res["flagged"] else "🟢 SAFE"
                print(f"Verdict: {status}")
            else:
                print("⚠️ Unexpected response structure.")
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    main()
