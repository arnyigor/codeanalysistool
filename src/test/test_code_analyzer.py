import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.code_analyzer.code_analyzer import CodeAnalyzer


@pytest.fixture
def mock_ollama_client():
    """Фикстура для мока клиента Ollama."""
    mock_client = MagicMock()
    mock_client.analyze_code.return_value = {
        "documentation": "/**\n * Тестовая документация\n */",
        "status": "success",
        "metrics": {
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_duration": 5000000000,  # 5 сек в наносекундах
            "generation_speed": 4.0
        }
    }
    return mock_client


def test_code_analyzer_init():
    """Тест инициализации анализатора кода."""
    with patch('src.code_analyzer.code_analyzer.OllamaClient') as mock_ollama:
        mock_client = MagicMock()
        mock_ollama.return_value = mock_client
        
        analyzer = CodeAnalyzer()
        
        assert analyzer.client == mock_client
        assert '.kt' in analyzer.supported_extensions
        assert analyzer.supported_extensions['.kt'] == 'kotlin'


def test_analyze_file(mock_ollama_client, tmp_path):
    """Тест анализа файла."""
    # Создаем временный файл
    test_file = tmp_path / "TestClass.kt"
    test_file.write_text("""
    class TestClass {
        fun hello() {
            println("Hello, World!")
        }
    }
    """)
    
    # Создаем анализатор с моком клиента
    analyzer = CodeAnalyzer(mock_ollama_client)
    
    # Запускаем анализ
    result = analyzer.analyze_file(str(test_file))
    
    # Проверяем результат
    assert result is not None
    assert "documentation" in result
    assert result["status"] == "success"
    assert result["file_path"] == str(test_file)
    assert result["file_type"] == "kotlin"
    assert result["file_name"] == "TestClass.kt"
    
    # Проверяем, что метод был вызван с правильными параметрами
    mock_ollama_client.analyze_code.assert_called_once()
    args, kwargs = mock_ollama_client.analyze_code.call_args
    assert "class TestClass" in args[0]
    assert args[1] == "kotlin"
    assert kwargs.get("context") is None


def test_analyze_file_with_context(mock_ollama_client, tmp_path):
    """Тест анализа файла с контекстом."""
    # Создаем временные файлы
    test_file = tmp_path / "TestClass.kt"
    test_file.write_text("""
    class TestClass {
        fun hello() {
            println("Hello, World!")
        }
    }
    """)
    
    context_file = tmp_path / "Context.kt"
    context_file.write_text("""
    interface Context {
        fun provide(): String
    }
    """)
    
    # Создаем анализатор с моком клиента
    analyzer = CodeAnalyzer(mock_ollama_client)
    
    # Запускаем анализ с контекстом
    result = analyzer.analyze_file(str(test_file), [str(context_file)])
    
    # Проверяем результат
    assert result is not None
    assert "documentation" in result
    assert result["status"] == "success"
    
    # Проверяем, что метод был вызван с правильными параметрами
    mock_ollama_client.analyze_code.assert_called_once()
    args, kwargs = mock_ollama_client.analyze_code.call_args
    assert kwargs.get("context") is not None
    assert "Context.kt" in kwargs.get("context")


def test_analyze_unsupported_file(mock_ollama_client, tmp_path):
    """Тест анализа неподдерживаемого файла."""
    # Создаем временный файл с неподдерживаемым расширением
    test_file = tmp_path / "test.txt"
    test_file.write_text("Это текстовый файл.")
    
    # Создаем анализатор с моком клиента
    analyzer = CodeAnalyzer(mock_ollama_client)
    
    # Запускаем анализ
    result = analyzer.analyze_file(str(test_file))
    
    # Проверяем результат
    assert result is None
    # Проверяем, что метод не был вызван
    mock_ollama_client.analyze_code.assert_not_called()


def test_analyze_path_file(mock_ollama_client, tmp_path):
    """Тест анализа пути (файл)."""
    # Создаем временный файл
    test_file = tmp_path / "TestClass.kt"
    test_file.write_text("class TestClass {}")
    
    # Создаем анализатор с моком клиента
    analyzer = CodeAnalyzer(mock_ollama_client)
    
    # Запускаем анализ
    results = analyzer.analyze_path(str(test_file))
    
    # Проверяем результат
    assert len(results) == 1
    assert str(test_file) in results
    
    # Запускаем анализ с выходной директорией
    output_dir = tmp_path / "output"
    results = analyzer.analyze_path(str(test_file), output_dir=str(output_dir))
    
    # Проверяем, что выходной файл создан
    expected_output = output_dir / "TestClass_doc.kt"
    assert expected_output.exists()


def test_analyze_path_directory(mock_ollama_client, tmp_path):
    """Тест анализа пути (директория)."""
    # Создаем временные файлы
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    
    kotlin_file = test_dir / "TestClass.kt"
    kotlin_file.write_text("class TestClass {}")
    
    java_file = test_dir / "TestJava.java"
    java_file.write_text("class TestJava {}")
    
    text_file = test_dir / "text.txt"
    text_file.write_text("Текстовый файл")
    
    # Создаем анализатор с моком клиента
    analyzer = CodeAnalyzer(mock_ollama_client)
    
    # Запускаем анализ без рекурсии
    results = analyzer.analyze_path(str(test_dir), recursive=False)
    
    # Проверяем результат (должен найти 2 файла: .kt и .java)
    assert len(results) == 2
    assert str(kotlin_file) in results
    assert str(java_file) in results
    
    # Проверяем, что .txt файл не обработан
    assert str(text_file) not in results 