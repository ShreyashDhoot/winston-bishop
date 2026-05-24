# jailbreak_diffusion/judger/post_checker/__init__.py

def _safe_import(module, name):
    try:
        # Use importlib for cleaner dynamic imports if needed, 
        # but relative dots are tricky with __import__
        from importlib import import_module
        # Get the current package name
        pkg = __name__
        mod = import_module(module, package=pkg)
        return getattr(mod, name)
    except Exception as e:
        # print(f"Warning: Could not import {name} from {module}: {e}")
        return None

MultiheadDetector = _safe_import('.MultiheadDetector', 'MultiheadDetector')
Q16Detector = _safe_import('.Q16', 'Q16Detector')
FinetunedQ16Detector = _safe_import('.Q16', 'FinetunedQ16Detector')
SD_SafetyCheckerDetector = _safe_import('.SD_safety_checker', 'SD_SafetyCheckerDetector')
OpenAIImageDetector = _safe_import('.openai_image_moderation', 'OpenAIImageDetector')
AzureContentSafetyDetector = _safe_import('.azure_image_checker', 'AzureContentSafetyDetector')
LlavaGuardChecker = _safe_import('.llava_guard', 'LlavaGuardChecker')
GPT_4o_mini_ImageChecker = _safe_import('.gpt_4o_mini', 'GPT_4o_mini_ImageChecker')
GPT_4o_ImageChecker = _safe_import('.gpt_4o', 'GPT_4o_ImageChecker')

__all__ = [
    'MultiheadDetector',
    'Q16Detector', 
    'FinetunedQ16Detector',
    'SD_SafetyCheckerDetector',
    'OpenAIImageDetector',
    'AzureContentSafetyDetector', 
    'LlavaGuardChecker',
    'GPT_4o_mini_ImageChecker',
    'GPT_4o_ImageChecker'
]

__version__ = '1.0.0'