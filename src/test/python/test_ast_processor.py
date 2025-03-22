import unittest
from pathlib import Path
from src.code_analyzer.ast_processor import ASTProcessor

class TestASTProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ASTProcessor()
        self.test_resources = Path('src/test/resources')
        
    def test_process_java_file(self):
        """Тест обработки Java файла"""
        java_file = self.test_resources / 'DataProcessor.java'
        result = self.processor.process_java_file(str(java_file))
        
        # Проверка структуры
        self.assertEqual(result['name'], 'DataProcessor')
        
        # Проверка дженериков
        self.assertIn('T', result['generics'])
        
        # Проверка аннотаций
        self.assertIn('SuppressWarnings', result['annotations'])
        
        # Проверка методов
        method_names = [m['name'] for m in result['methods']]
        self.assertIn('processDataAsync', method_names)
        self.assertIn('validateInput', method_names)
        
        # Проверка параметров и типов
        process_method = next(m for m in result['methods'] if m['name'] == 'processDataAsync')
        self.assertEqual(len(process_method['params']), 1)
        self.assertEqual(process_method['return_type'], 'CompletableFuture')
        
        # Проверка полей
        field_names = [f['name'] for f in result['fields']]
        self.assertIn('analyzer', field_names)
        self.assertIn('processedData', field_names)
        
        # Проверка импортов
        self.assertIn('com.example.KotlinAnalyzer', result['imports'])
        self.assertIn('java.util.concurrent.CompletableFuture', result['imports'])
        
    def test_process_kotlin_file(self):
        """Тест обработки Kotlin файла"""
        kotlin_file = self.test_resources / 'KotlinAnalyzer.kt'
        result = self.processor.process_kotlin_file(str(kotlin_file))
        
        # Проверка структуры
        self.assertEqual(result['name'], 'KotlinAnalyzer')
        
        # Проверка пакета
        self.assertEqual(result['package'], 'com.example')
        
        # Проверка дженериков
        self.assertTrue(any('T' in g for g in result['generics']))
        
        # Проверка аннотаций
        self.assertIn('Suppress', result['annotations'])
        
        # Проверка методов
        method_names = [m['name'] for m in result['methods']]
        self.assertIn('analyzeData', method_names)
        self.assertIn('addPattern', method_names)
        self.assertIn('getProcessingMetrics', method_names)
        
        # Проверка параметров и возвращаемых типов
        analyze_method = next(m for m in result['methods'] if m['name'] == 'analyzeData')
        self.assertEqual(len(analyze_method['params']), 1)
        self.assertIn('AnalysisResult', analyze_method['return_type'])
        
        # Проверка полей
        field_names = [f['name'] for f in result['fields']]
        self.assertIn('patterns', field_names)
        self.assertIn('processingTimes', field_names)
        self.assertIn('analysisCache', field_names)
        
        # Проверка типов полей
        patterns_field = next(f for f in result['fields'] if f['name'] == 'patterns')
        self.assertIn('ConcurrentHashMap', patterns_field['type'])
        
    def test_extract_relationships(self):
        """Тест извлечения связей между классами"""
        processor = ASTProcessor()
        
        # Тестируем Java файл
        java_relationships = processor.extract_relationships({
            'imports': ['com.example.KotlinAnalyzer'],
            'fields': [{'name': 'analyzer', 'type': 'KotlinAnalyzer'}]
        })
        self.assertTrue(len(java_relationships) > 0)
        
        # Тестируем Kotlin файл
        kotlin_relationships = processor.extract_relationships({
            'imports': ['java.util.concurrent.ConcurrentHashMap'],
            'fields': [{'name': 'map', 'type': 'ConcurrentHashMap<String, Int>'}]
        })
        self.assertTrue(len(kotlin_relationships) > 0)

    def test_generate_ast_report(self):
        """Тест генерации AST отчета"""
        processor = ASTProcessor()
        
        # Подготавливаем тестовые данные
        test_class_info = {
            'name': 'TestClass',
            'package': 'com.example.test',
            'imports': ['java.util.List', 'kotlin.collections.Map'],
            'interfaces': ['TestInterface'],
            'fields': [
                {
                    'name': 'testField',
                    'type': 'String',
                    'annotations': ['NotNull']
                }
            ],
            'methods': [
                {
                    'name': 'testMethod',
                    'params': [('param1', 'String'), ('param2', 'Int')],
                    'return_type': 'Boolean',
                    'is_suspend': True
                }
            ],
            'inner_classes': [
                {
                    'name': 'InnerTest',
                    'description': 'Test inner class',
                    'methods': []
                }
            ]
        }
        
        processor.class_info = test_class_info
        
        # Генерируем отчет в строку
        report = processor.generate_ast_report()
        
        # Проверяем наличие основных секций
        self.assertIn("# AST Analysis Report", report)
        self.assertIn("## Class Information", report)
        self.assertIn("## Import Structure", report)
        self.assertIn("## Field Structure", report)
        self.assertIn("## Method Structure", report)
        
        # Проверяем содержимое
        self.assertIn("TestClass", report)
        self.assertIn("com.example.test", report)
        self.assertIn("testField", report)
        self.assertIn("testMethod", report)
        self.assertIn("Is Suspend Function: Yes", report)
        
        # Тестируем сохранение в файл
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as tmp:
            report_file = tmp.name
            
        try:
            processor.generate_ast_report(report_file)
            self.assertTrue(os.path.exists(report_file))
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertEqual(report, content)
        finally:
            os.unlink(report_file)

    def test_prepare_llm_data(self):
        """Тест подготовки данных для LLM"""
        processor = ASTProcessor()
        
        # Подготавливаем тестовые данные
        test_class_info = {
            'name': 'TestClass',
            'package': 'com.example.test',
            'imports': ['kotlin.collections.List'],
            'interfaces': ['TestInterface'],
            'fields': [
                {
                    'name': 'testField',
                    'type': 'String',
                    'annotations': ['NotNull']
                }
            ],
            'methods': [
                {
                    'name': 'testMethod',
                    'params': [('param1', 'String')],
                    'return_type': 'Boolean',
                    'is_suspend': True
                }
            ]
        }
        
        processor.class_info = test_class_info
        
        # Получаем данные для LLM
        llm_data = processor.prepare_llm_data()
        
        # Проверяем структуру данных
        self.assertIn('file_info', llm_data)
        self.assertIn('structure', llm_data)
        self.assertIn('relationships', llm_data)
        
        # Проверяем содержимое
        self.assertEqual(llm_data['file_info']['class_name'], 'TestClass')
        self.assertEqual(llm_data['file_info']['type'], 'Kotlin')
        
        # Проверяем структуру
        self.assertEqual(len(llm_data['structure']['fields']), 1)
        self.assertEqual(len(llm_data['structure']['methods']), 1)
        
        # Проверяем поле
        test_field = llm_data['structure']['fields'][0]
        self.assertEqual(test_field['name'], 'testField')
        self.assertEqual(test_field['type'], 'String')
        
        # Проверяем метод
        test_method = llm_data['structure']['methods'][0]
        self.assertEqual(test_method['name'], 'testMethod')
        self.assertTrue(test_method['is_suspend'])

    def test_generate_documentation(self):
        """Тест генерации документации"""
        # Тест документации Java класса
        java_file = self.test_resources / 'DataProcessor.java'
        java_info = self.processor.process_java_file(str(java_file))
        java_doc = self.processor.generate_documentation(java_info)
        
        # Проверка основных разделов
        self.assertIn('# Класс DataProcessor', java_doc)
        self.assertIn('## Параметры типа', java_doc)
        self.assertIn('## Методы', java_doc)
        self.assertIn('## Поля', java_doc)
        self.assertIn('## Зависимости', java_doc)
        
        # Проверка деталей документации
        self.assertIn('CompletableFuture', java_doc)
        self.assertIn('ProcessingException', java_doc)
        self.assertIn('ErrorType', java_doc)
        
        # Тест документации Kotlin класса
        kotlin_file = self.test_resources / 'KotlinAnalyzer.kt'
        kotlin_info = self.processor.process_kotlin_file(str(kotlin_file))
        kotlin_doc = self.processor.generate_documentation(kotlin_info)
        
        # Проверка основных разделов
        self.assertIn('# Класс KotlinAnalyzer', kotlin_doc)
        self.assertIn('## Параметры типа', kotlin_doc)
        self.assertIn('## Методы', kotlin_doc)
        self.assertIn('## Поля', kotlin_doc)
        
        # Проверка деталей документации
        self.assertIn('suspend fun', kotlin_doc)
        self.assertIn('Flow<ProcessingMetrics>', kotlin_doc)
        self.assertIn('ConcurrentHashMap', kotlin_doc)
        
    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Тест несуществующего файла
        with self.assertRaises(Exception) as context:
            self.processor.process_file('nonexistent.java')
        self.assertIn('Ошибка при обработке', str(context.exception))
        
        # Тест неподдерживаемого типа файла
        with self.assertRaises(ValueError) as context:
            self.processor.process_file('test.cpp')
        self.assertIn('Неподдерживаемый тип файла', str(context.exception))

if __name__ == '__main__':
    unittest.main() 