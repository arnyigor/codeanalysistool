"""
Code Analyzer package for analyzing Android code
"""

__version__ = "0.1"

from src.code_analyzer.code_analyzer import CodeAnalyzer
from src.code_analyzer.file_processor import main as run_analyzer

__all__ = ["CodeAnalyzer", "run_analyzer"] 