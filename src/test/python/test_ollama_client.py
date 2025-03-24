import logging
import os
import re
import sys
from pathlib import Path

# Корректная настройка путей
ROOT_DIR = Path(__file__).parent.parent.parent  # Указываем на src/
sys.path.append(str(ROOT_DIR.parent))

import ollama
import pytest
from src.llm.llm_client import OllamaClient

LOG_FILE = os.path.abspath("test_ollama.log")


def setup_logging():
    """Настройка логгера для тестов"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s '
        '[%(filename)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Очистка файла перед запуском
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("")

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s → %(filename)s:%(lineno)d'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


# Настройка логгирования перед тестами
setup_logging()


@pytest.fixture(scope="session", autouse=True)
def check_ollama_available():
    """Проверка доступности Ollama перед тестами"""
    try:
        models = ollama.list().get('models', [])
        assert len(models) > 0, "Нет доступных моделей Ollama"
    except Exception as e:
        pytest.exit(f"Ошибка подключения к Ollama: {str(e)}")


@pytest.fixture
def ollama_client():
    """Фикстура для реального клиента Ollama"""
    return OllamaClient()


def test_real_model_initialization(ollama_client):
    """Проверка инициализации с реальной моделью"""
    models = ollama.list().get('models', [])
    model_names = [m['model'] for m in models]  # Проверяем структуру ответа
    assert ollama_client.model in model_names, "Модель не найдена в списке доступных"


def test_real_analyze_code_java(ollama_client):
    code = """
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
    }
    """
    result = ollama_client.analyze_code(code, "java")

    # Исправьте проверку на наличие ключа "documentation"
    assert "documentation" in result, "Отсутствует ключ 'documentation' в ответе"

    # Добавьте проверку содержимого документации
    assert "Calculator" in result["documentation"], "Документация не содержит ожидаемый класс"
    assert "add" in result["documentation"], "Документация не содержит описание метода"

    # Проверьте метрики
    assert "metrics" in result, "Отсутствуют метрики в ответе"
    assert result["metrics"]["time"] > 0, "Время выполнения не указано"


def test_real_analyze_code_kotlin(ollama_client):
    code = """
    class DataProcessor {
        fun filterData(data: List<Int>): List<Int> {
            return data.filter { it > 0 }
        }
    }
    """
    result = ollama_client.analyze_code(code, "kotlin")

    # 1. Проверка наличия ключа 'documentation'
    assert "documentation" in result, "Отсутствует ключ 'documentation' в ответе"

    # 2. Проверка корректности документации
    assert "DataProcessor" in result["documentation"], "Документация не содержит класс DataProcessor"
    assert "filterData" in result["documentation"], "Документация не содержит описание метода filterData"
    assert "KDoc" in result["documentation"], "Формат документации не соответствует KDoc"

    # 3. Проверка метрик
    assert "metrics" in result, "Отсутствуют метрики в ответе"
    assert result["metrics"]["time"] > 0, "Время выполнения не указано"
#
#
# def test_real_model_parameters(ollama_client):
#     """Проверка параметров модели"""
#     model_info = ollama.show(ollama_client.model)
#     assert 'model_info' in model_info, "Отсутствует model_info"
#     assert 'general' in model_info['model_info'], "Отсутствует раздел general"
#     assert 'parameter_count' in model_info['model_info']['general'], "Отсутствует parameter_count"
#
#
# def test_large_code_java(ollama_client):
#     """Тест на большие объемы кода (500 строк)"""
#     code = "class BigClass {\n" + "    void method() {}\n" * 500 + "}"
#     result = ollama_client.analyze_code(code, "java")
#
#     assert "error" not in result, "Ошибка при обработке большого кода"
#     assert len(result['code']) > len(code), "Документация не добавлена"
#
#
# def test_invalid_file_type(ollama_client):
#     """Проверка обработки неподдерживаемых типов"""
#     with pytest.raises(ValueError):
#         ollama_client.analyze_code("...", "python")
#
#
# def test_missing_code(ollama_client):
#     """Тест на пустой код"""
#     result = ollama_client.analyze_code("", "java")
#     assert "error" in result, "Не обнаружена ошибка пустого кода"
#     assert "пустой код" in result['error'], "Неверное сообщение об ошибке"
#
#
# def test_real_api_response(ollama_client):
#     """Проверка структуры ответа модели"""
#     code = "class Test {}"
#     result = ollama_client.analyze_code(code, "java")
#
#     assert isinstance(result, dict), "Ответ не является словарем"
#     assert "generation_info" in result, "Отсутствует информация о генерации"
#     assert "chunks_received" in result["generation_info"], "Отсутствует статистика по чанкам"
#
#
# @pytest.mark.slow
# def test_context_limit(ollama_client):
#     """Тест превышения лимита контекста (16KB)"""
#     long_code = "public class Overflow {\n" + "    void method() { /* " + "x" * 16384 + " */ }\n}"
#     result = ollama_client.analyze_code(long_code, "java")
#
#     assert "error" in result, "Не обнаружена ошибка превышения контекста"
#     assert "превышен лимит контекста" in result['error'], "Неверное сообщение об ошибке"
#
#
# def test_documentation_presence(ollama_client):
#     """Проверка наличия документации в ответе"""
#     code = "class Logger {\n    void log(String msg) {}\n}"
#     result = ollama_client.analyze_code(code, "java")
#
#     assert re.search(r'/\*\*\n.*?\*/', result['code'], flags=re.DOTALL), \
#         "Отсутствует многострочная документация"
#     assert "log(String msg)" in result['code'], "Метод не документирован"