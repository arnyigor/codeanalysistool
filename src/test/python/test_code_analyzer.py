import unittest
import asyncio
from pathlib import Path
from src.llm.code_analyzer import CodeAnalyzer

class TestCodeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.code_analyzer = CodeAnalyzer()
        self.test_resources = Path('src/test/resources')
        
    def test_build_llm_prompt(self):
        """Тест создания промпта для LLM"""
        ast_info = {
            'name': 'TestClass',
            'methods': [
                {
                    'name': 'testMethod',
                    'params': ['param1', 'param2'],
                    'return_type': 'void'
                }
            ],
            'fields': [
                {
                    'name': 'testField',
                    'type': 'String'
                }
            ],
            'imports': ['java.util.List']
        }
        
        prompt = self.code_analyzer._build_llm_prompt(ast_info)
        
        # Проверка наличия основных элементов в промпте
        self.assertIn('TestClass', prompt)
        self.assertIn('testMethod', prompt)
        self.assertIn('testField', prompt)
        self.assertIn('java.util.List', prompt)
        
    def test_combine_analysis(self):
        """Тест объединения результатов анализа"""
        ast_info = {
            'name': 'TestClass',
            'methods': [],
            'fields': [],
            'imports': []
        }
        
        llm_analysis = {
            'description': 'Test description',
            'complexity': 'Low',
            'recommendations': []
        }
        
        result = self.code_analyzer._combine_analysis(ast_info, llm_analysis)
        
        # Проверка структуры результата
        self.assertIn('structure', result)
        self.assertIn('semantic_analysis', result)
        self.assertIn('relationships', result)
        self.assertEqual(result['semantic_analysis'], llm_analysis)
        
    def test_analyze_java_file(self):
        """Тест анализа Java файла"""
        async def run_test():
            java_file = self.test_resources / 'DataProcessor.java'
            result = await self.code_analyzer.analyze_file(str(java_file))
            
            # Проверка результата
            self.assertIsNotNone(result)
            self.assertIn('structure', result)
            self.assertIn('semantic_analysis', result)
            self.assertIn('relationships', result)
            self.assertEqual(result['structure']['name'], 'DataProcessor')
            self.assertIsNotNone(result['semantic_analysis'])
            
        asyncio.run(run_test())
        
    def test_analyze_kotlin_file(self):
        """Тест анализа Kotlin файла"""
        async def run_test():
            kotlin_file = self.test_resources / 'KotlinAnalyzer.kt'
            result = await self.code_analyzer.analyze_file(str(kotlin_file))
            
            # Проверка результата
            self.assertIsNotNone(result)
            self.assertIn('structure', result)
            self.assertIn('semantic_analysis', result)
            self.assertEqual(result['structure']['name'], 'KotlinAnalyzer')
            self.assertIsNotNone(result['semantic_analysis'])
            
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main() 