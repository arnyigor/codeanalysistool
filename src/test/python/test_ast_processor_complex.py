import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List

# Добавляем путь к src в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from code_analyzer.ast_processor import ASTProcessor

class TestASTProcessorComplex(unittest.TestCase):
    def setUp(self):
        self.processor = ASTProcessor()
        self.test_app_path = Path('src/test/resources/testapp')
        self.addedit_path = self.test_app_path / 'addedit'
        self.view_path = self.addedit_path / 'view'
        self.presenter_path = self.addedit_path / 'presenter'

    def test_process_complex_kotlin_file(self):
        """Тест обработки сложного Kotlin файла с множеством зависимостей"""
        # Тестируем AddEditPresenter.kt - самый сложный файл
        presenter_file = self.presenter_path / 'AddEditPresenter.kt'
        class_info = self.processor.process_kotlin_file(str(presenter_file))

        # Проверяем базовую структуру
        self.assertIn('name', class_info)
        self.assertIn('imports', class_info)
        self.assertIn('fields', class_info)
        self.assertIn('methods', class_info)

        # Проверяем импорты на наличие Moxy и Interactor
        imports = class_info['imports']
        self.assertTrue(any('Interactor' in imp for imp in imports), 
                       "Должен быть импорт Interactor")
        self.assertTrue(any('moxy' in imp.lower() for imp in imports),
                       "Должен быть импорт Moxy")

        # Проверяем аннотации Moxy
        self.assertTrue(any('@InjectViewState' in ann for ann in class_info['annotations']),
                       "Должна быть аннотация @InjectViewState")

    def test_process_view_files(self):
        """Тест обработки файлов из папки view"""
        view_files = [
            'AddEditView.kt',
            'AddEditFragment.kt',
            'AircraftSpinnerAdapter.kt',
            'CustomFieldValuesAdapter.kt',
            'MultiAutoCompleteAdapter.kt'
        ]

        for file_name in view_files:
            with self.subTest(file=file_name):
                file_path = self.view_path / file_name
                class_info = self.processor.process_kotlin_file(str(file_path))

                # Проверяем основные поля
                self.assertIsNotNone(class_info['name'])
                self.assertIsNotNone(class_info['package'])
                
                # Проверяем наличие методов
                self.assertGreater(len(class_info['methods']), 0,
                                 f"Файл {file_name} должен содержать методы")

                # Для AddEditView проверяем, что это интерфейс MVP
                if file_name == 'AddEditView.kt':
                    self.assertTrue(any('MvpView' in intf for intf in class_info['interfaces']),
                                  "AddEditView должен реализовывать MvpView")

    def test_cross_file_relationships(self):
        """Тест анализа связей между файлами"""
        # Анализируем AddEditFragment и его связи с Presenter
        fragment_file = self.view_path / 'AddEditFragment.kt'
        fragment_info = self.processor.process_kotlin_file(str(fragment_file))
        
        # Проверяем связь с Presenter
        relationships = self.processor.extract_relationships(fragment_info)
        presenter_refs = [r for r in relationships if 'Presenter' in r]
        self.assertTrue(len(presenter_refs) > 0,
                       "AddEditFragment должен иметь связь с Presenter")

    def test_adapter_pattern_detection(self):
        """Тест обнаружения паттерна Adapter"""
        adapter_files = [
            'AircraftSpinnerAdapter.kt',
            'CustomFieldValuesAdapter.kt',
            'MultiAutoCompleteAdapter.kt'
        ]

        for file_name in adapter_files:
            with self.subTest(file=file_name):
                file_path = self.view_path / file_name
                class_info = self.processor.process_kotlin_file(str(file_path))
                
                # Проверяем наличие типичных методов адаптера
                method_names = [m['name'] for m in class_info['methods']]
                self.assertTrue(
                    'getItemTitle' in method_names or 'onBindViewHolder' in method_names,
                    f"Файл {file_name} должен содержать методы адаптера"
                )

    def test_missing_imports_handling(self):
        """Тест обработки отсутствующих импортов"""
        presenter_file = self.presenter_path / 'AddEditPresenter.kt'
        class_info = self.processor.process_kotlin_file(str(presenter_file))
        
        # Собираем все импортируемые файлы
        imported_files = []
        for imp in class_info['imports']:
            if '.kt' in imp:
                imported_files.append(imp)

        # Проверяем, что отсутствующие файлы не вызывают ошибок
        for imp_file in imported_files:
            relationships = self.processor.extract_relationships({
                'imports': [imp_file],
                'fields': [],
                'methods': []
            })
            self.assertIsInstance(relationships, list,
                                f"Обработка импорта {imp_file} должна возвращать список")

    def test_complex_field_types(self):
        """Тест обнаружения сложных типов полей"""
        presenter_file = self.presenter_path / 'AddEditPresenter.kt'
        class_info = self.processor.process_kotlin_file(str(presenter_file))
        
        fields = class_info['fields']
        field_types = [field['type'] for field in fields]
        
        # Проверяем наличие сложных типов
        complex_types = [
            'MutableList<CustomFieldValue>',
            'Disposable',
            'Provider<AddEditPresenter>'
        ]
        
        for type_name in complex_types:
            self.assertTrue(
                any(type_name in ft for ft in field_types),
                f"Должно быть поле типа {type_name}"
            )

    def test_suspend_functions(self):
        """Тест обнаружения suspend функций"""
        presenter_file = self.presenter_path / 'AddEditPresenter.kt'
        class_info = self.processor.process_kotlin_file(str(presenter_file))
        
        methods = class_info['methods']
        suspend_methods = [m for m in methods if m.get('is_suspend', False)]
        
        self.assertTrue(len(suspend_methods) > 0,
                       "Должны быть найдены suspend функции")

    def test_generate_complex_ast_report(self):
        """Тест генерации отчета по AST"""
        # Анализируем все файлы
        for file_path in self.view_path.glob('*.kt'):
            self.processor.process_kotlin_file(str(file_path))
        
        # Генерируем отчет
        report = self.processor.generate_ast_report()
        
        # Проверяем наличие ключевых компонентов в отчете
        self.assertIn('MVP Architecture', report)
        self.assertIn('Presenter', report)
        self.assertIn('View', report)
        self.assertIn('Adapters', report)

if __name__ == '__main__':
    unittest.main() 