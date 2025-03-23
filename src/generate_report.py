import os
import sys

# Добавляем путь к src в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.code_analyzer.ast_processor import ASTProcessor

def main():
    processor = ASTProcessor()
    
    # Анализируем файл
    kotlin_file = 'test/resources/KotlinAnalyzer.kt'
    info = processor.process_kotlin_file(kotlin_file)
    processor.class_info = info
    
    # Генерируем отчет
    report_file = '../ast_report.md'
    processor.generate_ast_report(report_file)
    print(f'Отчет сохранен в {report_file}')