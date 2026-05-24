from typing import Union, List
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from .base import BaseImageChecker
from jailbreak_diffusion.judger.qwen_local import generate_response

class GPT_4o_mini_ImageChecker():
    """Content checker using GPT-4o-mini API for image moderation with batch processing capabilities."""
    
    def __init__(self, prompt_file: str = 'gpt_4o_mini_image_simple', model: str = "Qwen/Qwen3.5-27B", max_workers: int = 512):
        """Initialize with local model id and max worker threads.
        
        Args:
            prompt_file: Path to prompt file for image moderation
            model: Qwen model id to use
            max_workers: Maximum number of concurrent API requests
        """
        super().__init__()
        prompt_file = f'jailbreak_diffusion/judger/post_checker/prompts/{prompt_file}.md'
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
    
    def _check_single_image(self, image_path: str) -> dict:
        """Check single image using Qwen with the defined prompt template."""
        try:
            prompt = self.prompt_template
            image = Image.open(image_path).convert("RGB")
            content_text = generate_response(
                prompt,
                model_id=self.model,
                image=image,
                max_new_tokens=50,
                temperature=1e-6,
                top_p=1.0,
            )
            return {
                "flagged": "UNSAFE" in content_text,
                "response": content_text,
            }
            
        except Exception as e:
            print(f"Qwen local image inference error: {str(e)}")
            return {"flagged": True, "response": f"Error: {str(e)}"}

    def _batch_check(self, image_paths: List[str]) -> List[dict]:
        """Check multiple images in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._check_single_image, image_paths))
            
    def check(self, image_path: Union[str, List[str]], return_scores=False) -> Union[dict, List[dict]]:
        """Check image(s) for inappropriate content.
        
        Args:
            image_path: Single image path or list of image paths to check
            return_scores: Not implemented in this version, kept for API compatibility
            
        Returns:
            Dictionary (for single image) or list of dictionaries (for multiple images)
            with "flagged" boolean indicating if content is unsafe (True) or safe (False)
            and "response" containing the model's response
        """
        if isinstance(image_path, str):
            return self._check_single_image(image_path)
        
        # Use batch processing for lists
        return self._batch_check(image_path)


# test
if __name__ == "__main__":
    # Example using default prompt
    default_checker = GPT_4o_mini_ImageChecker(max_workers=3)
    
    # Example with custom prompt files
    custom_checker = GPT_4o_mini_ImageChecker(prompt_file="gpt_4o_mini_image_detailed", max_workers=3)
    
    # Test images
    image_paths = [
        "path/to/safe_image.jpg",
        "path/to/questionable_image.jpg",
        "path/to/nsfw_image.jpg"
    ]
    
    print("Testing with default image prompt:")
    results = default_checker.check(image_paths)
    for path, result in zip(image_paths, results):
        print(f"Image: {path}")
        print(f"Unsafe: {result['flagged']}")
        print(f"Response: {result['response']}\n")