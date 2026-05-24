# jailbreak_diffusion/judger/pre_checker/__init__.py

from .base import BaseChecker

def _safe_import(module, name):
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except Exception as e:
        # print(f"Warning: Could not import {name} from {module}: {e}")
        return None

NSFW_text_classifier_Checker = _safe_import('.NSFW_text_classifier', 'NSFW_text_classifier_Checker')
NSFW_word_match_Checker = _safe_import('.NSFW_word_match', 'NSFW_word_match_Checker')
AzureTextDetector = _safe_import('.azure_text_checker', 'AzureTextDetector')
CompositeChecker = _safe_import('.composite', 'CompositeChecker')
distilbert_nsfw_text_checker = _safe_import('.distilbert_nsfw_text_checker', 'distilbert_nsfw_text_checker')
distilroberta_nsfw_text_checker = _safe_import('.distilroberta_nsfw_text_checker', 'distilroberta_nsfw_text_checker')
GoogleTextModerator = _safe_import('.google_text_checker', 'GoogleTextModerator')
GPTChecker = _safe_import('.gpt', 'GPTChecker')
GeminiChecker = _safe_import('.gemini', 'GeminiChecker')
LlamaGuardChecker = _safe_import('.llama_guard', 'LlamaGuardChecker')
OpenAITextDetector = _safe_import('.openai_text_moderation', 'OpenAITextDetector')
DetoxifyChecker = _safe_import('.detoxify', 'DetoxifyChecker')
NvidiaAegisChecker = _safe_import('.nvidia_aegis', 'NvidiaAegisChecker')

__all__ = [
    'BaseChecker',
    'NSFW_text_classifier_Checker',
    'NSFW_word_match_Checker',
    'AzureTextDetector',
    'CompositeChecker',
    'distilbert_nsfw_text_checker',
    'distilroberta_nsfw_text_checker',
    'GoogleTextModerator',
    'GPTChecker',
    'GeminiChecker',
    'LlamaGuardChecker',
    'OpenAITextDetector',
    'DetoxifyChecker',
    'NvidiaAegisChecker'
]