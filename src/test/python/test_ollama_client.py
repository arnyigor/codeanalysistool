import logging
import os
import re
import sys
from pathlib import Path
import json

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


def test_real_model_parameters(ollama_client):
    """Проверка параметров модели"""
    model_info = ollama.show(ollama_client.model)
    
    # Подробное логирование информации о модели
    logging.info("=== Информация о модели ===")
    if 'model' in model_info:
        logging.info(f"Название модели: {model_info.get('model', 'Неизвестно')}")
    if 'parameters' in model_info:
        logging.info(f"Параметры модели: {model_info['parameters']}")
    logging.info("========================")
    
    assert model_info is not None, "Не удалось получить информацию о модели"


def test_invalid_file_type(ollama_client):
    """Проверка обработки неподдерживаемых типов файлов"""
    with pytest.raises(ValueError, match="Поддерживается только Kotlin"):
        ollama_client.analyze_code("...", "java")


def test_missing_code(ollama_client):
    """Тест на пустой код"""
    result = ollama_client.analyze_code("", "kotlin")
    assert "error" in result, "Не обнаружена ошибка пустого кода"
    assert "пустой код" in result['error'], "Неверное сообщение об ошибке"


def test_real_api_response(ollama_client):
    """Проверка структуры ответа модели"""
    code = """
    class Test {
        fun test() {}
    }
    """
    result = ollama_client.analyze_code(code, "kotlin")

    assert isinstance(result, dict), "Ответ не является словарем"
    assert "documentation" in result, "Отсутствует документация"
    assert "metrics" in result, "Отсутствуют метрики"
    assert "time" in result["metrics"], "Отсутствует время выполнения"


def test_documentation_presence(ollama_client):
    """Проверка наличия документации в ответе"""
    code = """
    class Logger {
        fun log(msg: String) {}
    }
    """
    result = ollama_client.analyze_code(code, "kotlin")

    assert "documentation" in result, "Отсутствует поле documentation в ответе"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc
    assert "/**" in doc, "Отсутствует начало KDoc блока"
    assert "*/" in doc, "Отсутствует конец KDoc блока"
    assert "@property" in doc or "@constructor" in doc, "Отсутствуют основные KDoc аннотации"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_real_android_code(ollama_client):
    """Тест документирования реального Android кода на Kotlin."""
    android_code = """
    class HomeFragment : Fragment() {
        private lateinit var binding: FragmentHomeBinding
        private val viewModel: HomeViewModel by viewModels()
        
        override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
        ): View {
            binding = FragmentHomeBinding.inflate(inflater, container, false)
            return binding.root
        }
    }
    """
    result = ollama_client.analyze_code(android_code, "kotlin")

    # Проверяем основные элементы документации
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    assert "@see" in doc, "Отсутствуют ссылки на связанные классы"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_real_analyze_code_kotlin(ollama_client):
    """Тест анализа простого Kotlin класса"""
    code = """
    class DataProcessor {
        private val cache = mutableMapOf<String, Int>()
        
        fun processData(data: List<String>): Map<String, Int> {
            return data.groupBy { it }.mapValues { it.value.size }
        }
    }
    """
    result = ollama_client.analyze_code(code, "kotlin")

    # Проверка наличия документации
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверка структуры KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    
    # Проверка обязательных секций
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"

    # Проверка метрик
    assert "metrics" in result, "Отсутствуют метрики"
    assert result["metrics"]["time"] > 0, "Некорректное время выполнения"
