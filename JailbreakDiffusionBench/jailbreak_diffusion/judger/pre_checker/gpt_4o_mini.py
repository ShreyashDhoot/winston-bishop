from typing import Union, List
from concurrent.futures import ThreadPoolExecutor

from .base import BaseChecker
from jailbreak_diffusion.judger.qwen_local import generate_response

class GPTChecker(BaseChecker):
    """Content checker using GPT API with batch processing capabilities and customizable prompts."""
    
    def __init__(self, prompt_file: str = 'gpt_4o_mini_simple', model: str = "Qwen/Qwen3.5-27B", max_workers: int = 512):
        """Initialize with local model id and max worker threads.
        
        Args:
            prompt_file: Path to prompt file (simple.md or incontext.md)
            model: Qwen model id to use
            max_workers: Maximum number of concurrent API requests
        """
        super().__init__()
        prompt_file = f'jailbreak_diffusion/judger/pre_checker/prompts/{prompt_file}.md'
        self.model = model
        self.max_workers = max_workers
        
        # Load prompt from file if provided
        self.prompt_template = self._load_prompt_template(prompt_file)
        
    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from a file."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt file {prompt_file}: {str(e)}")
            raise ValueError(f"Could not load prompt file: {prompt_file}")
        
    def _check_single_text(self, text: str) -> bool:
        """Check single text using GPT with the defined prompt template."""
        try:
            # Format the prompt template with the input text
            prompt = self.prompt_template.format(PROMPT=text)
            
            content_text = generate_response(
                prompt,
                model_id=self.model,
                max_new_tokens=10,
                temperature=1e-6,
                top_p=1.0,
            )
            return {
                "flagged": "UNSAFE" in content_text,
                "response": content_text,
            }
            
        except Exception as e:
            print(f"Qwen local inference error: {str(e)}")
            return False

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


# test
if __name__ == "__main__":
    # Example using default prompt
    default_checker = GPTChecker(max_workers=3)
    
    # Example using simple.md prompt
    simple_checker = GPTChecker(prompt_file="simple.md", max_workers=3)
    
    # Example using incontext.md prompt
    incontext_checker = GPTChecker(prompt_file="incontext.md", max_workers=3)
    
    texts = [
        "This is a safe text about programming.",
        "Some inappropriate adult content here",
        "Fuck your mother! Bro."
    ]
    
    print("Testing with default prompt:")
    results = default_checker.check(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Unsafe: {result}\n")