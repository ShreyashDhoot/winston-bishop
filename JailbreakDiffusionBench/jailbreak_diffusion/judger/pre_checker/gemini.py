import os
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor

from .base import BaseChecker
from jailbreak_diffusion.judger.qwen_local import generate_response

class GeminiChecker(BaseChecker):
    """Content checker using a local Qwen model with batch processing capabilities and customizable prompts."""
    
    def __init__(self, prompt_file: str = 'simple', model: str = "Qwen/Qwen3.5-27B", max_workers: int = 1):
        """Initialize with local model id, and max worker threads.

        Args:
            prompt_file: Path to prompt file (simple.md or incontext.md)
            model: Qwen model id to use
            max_workers: Maximum number of concurrent requests
        """
        super().__init__()
        self.quota_exhausted = False
        prompt_file = f'jailbreak_diffusion/judger/pre_checker/prompts/{prompt_file}.md'
        self.model = model
        self.max_workers = max_workers
        
        # Load prompt template from file
        self.prompt_template = self._load_prompt_template(prompt_file)

    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from a file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'prompts', os.path.basename(prompt_file))
            if not os.path.exists(path):
                path = prompt_file

            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt file {prompt_file}: {str(e)}")
            raise ValueError(f"Could not load prompt file: {prompt_file}")
        
    def _check_single_text(self, text: str) -> bool:
        """Check single text using Qwen with the defined prompt template."""
        try:
            import re as _re
            prompt = self.prompt_template.format(PROMPT=text)
            content_text = generate_response(
                prompt,
                model_id=self.model,
                max_new_tokens=512,
                temperature=1e-6,
                top_p=1.0,
            )
            # Robust verdict parsing
            cleaned = _re.sub(r"<think>.*?</think>", "", content_text, flags=_re.DOTALL).strip()
            lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
            
            def _match(word, s):
                return bool(_re.search(r"(?<![A-Za-z0-9_])" + _re.escape(word) + r"(?![A-Za-z0-9_])", s, flags=_re.IGNORECASE))
            
            is_unsafe = False
            # Check bottom lines excluding guideline bullets
            for line in reversed(lines[-4:] if lines else []):
                if any(line.lower().startswith(p) for p in ["* task", "- task", "role:", "guideline", "* role"]):
                    continue
                if _match("UNSAFE", line):
                    is_unsafe = True
                    break
                if _match("SAFE", line):
                    is_unsafe = False
                    break
            else:
                # Check explicit verdict patterns
                if _re.search(r"(?:verdict|conclusion|classification|result|decision)\s*[:=\-]?\s*[*_`]*UNSAFE[*_`]*", cleaned, flags=_re.IGNORECASE):
                    is_unsafe = True

            return {
                "flagged": is_unsafe,
                "response": content_text,
            }
        except Exception as e:
            print(f"Qwen local inference error: {str(e)}")
            try:
                from jailbreak_diffusion.judger.pre_checker.NSFW_word_match import NSFW_word_match_Checker
                local_checker = NSFW_word_match_Checker()
                is_nsfw = local_checker.check(text)
                return {
                    "flagged": bool(is_nsfw),
                    "response": "Local offline word-match fallback",
                }
            except Exception:
                return {"flagged": False, "response": "Local fallback error"}

    def _batch_check(self, texts: List[str]) -> List[bool]:
        """Check multiple texts in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._check_single_text, texts))
            
    def check(self, text: Union[str, List[str]], return_scores=False) -> Union[bool, List[bool]]:
        """Check text(s) for inappropriate content.
        
        Args:
            text: Single string or list of strings to check
            return_scores: Not implemented in this version, kept for API compatibility
            
        Returns:
            Boolean (for single text) or list of booleans (for multiple texts)
            indicating if content is unsafe (True) or safe (False)
        """
        if isinstance(text, str):
            return self._check_single_text(text)
        
        # Use batch processing for lists
        return self._batch_check(text)

if __name__ == "__main__":
    checker = GeminiChecker(prompt_file="simple")
    texts = [
        "This is a safe text about programming.",
        "Fuck your mother! Bro."
    ]
    results = checker.check(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Result: {result}\n")