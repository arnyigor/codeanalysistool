import logging
import os
import re
import sys
from pathlib import Path
import json
from typing import Optional, Dict

# Корректная настройка путей
ROOT_DIR = Path(__file__).parent.parent.parent  # Указываем на src/
sys.path.append(str(ROOT_DIR.parent))

import ollama
import pytest
from src.llm.llm_client import OllamaClient

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
    
    log_model_info(model_info)
    
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
    assert result["metrics"]["tokens"] > 0, "Некорректное количество токенов"
    assert result["metrics"]["speed"] > 0, "Некорректная скорость обработки"


def log_context_info(context: Optional[Dict[str, str]] = None):
    """Логирует информацию о контексте в структурированном виде"""
    if not context:
        logging.info("Контекст: отсутствует")
        return

    logging.info("\nКонтекст:")
    for title, content in context.items():
        if content and content.strip():
            content_preview = content.strip()[:100] + "..." if len(content) > 100 else content
            logging.info(f"- {title}:")
            logging.info(f"  {content_preview}")

def log_metrics(metrics: Dict):
    """Логирует метрики в структурированном виде"""
    logging.info("\nМетрики выполнения:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logging.info(f"- {key}: {value:.2f}")
        else:
            logging.info(f"- {key}: {value}")
    
    # Добавляем информацию о скорости обработки
    if "total_duration" in metrics and "total_tokens" in metrics:
        tokens_per_second = metrics["total_tokens"] / (metrics["total_duration"] / 1e9)  # наносекунды в секунды
        logging.info(f"- Скорость обработки: {tokens_per_second:.2f} токенов/сек")
        logging.info(f"- Среднее время на токен: {(metrics['total_duration'] / 1e9 / metrics['total_tokens'])*1000:.2f} мс")

def log_model_info(model_info: Dict):
    """Логирует информацию о модели в структурированном виде"""
    logging.info("\nИнформация о модели:")
    if 'model' in model_info:
        logging.info(f"- Название: {model_info['model']}")
    if 'parameters' in model_info:
        logging.info("- Параметры:")
        for param, value in model_info['parameters'].items():
            logging.info(f"  • {param}: {value}")
    
    # Добавляем информацию о размере модели и использовании памяти
    if 'details' in model_info:
        details = model_info['details']
        if 'parameter_size' in details:
            logging.info(f"- Размер модели: {details['parameter_size']}")
        if 'memory_per_token' in details:
            logging.info(f"- Память на токен: {details['memory_per_token']} байт")
        if 'vocab_size' in details:
            logging.info(f"- Размер словаря: {details['vocab_size']} токенов")


def test_documentation_with_context(ollama_client):
    """Тест генерации документации с учетом контекста"""
    # Основной код для документирования
    main_code = """
    class UserRepository {
        private val userDao: UserDao
        
        fun getUser(id: String): User? {
            return userDao.findById(id)
        }
    }
    """
    
    # Контекст - интерфейс и связанные классы
    context = {
        "Интерфейс": """
        interface UserDao {
            fun findById(id: String): User?
            fun save(user: User)
            fun delete(id: String)
        }
        """,
        "Модель": """
        data class User(
            val id: String,
            val name: String,
            val email: String
        )
        """
    }
    
    log_context_info(context)
    
    result = ollama_client.analyze_code(main_code, "kotlin", context=context)
    
    logging.info("\nРезультаты анализа кода:")
    log_metrics(result["metrics"])
    model_info = ollama.show(ollama_client.model)
    log_model_info(model_info)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc с учетом контекста
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    assert "UserDao" in doc, "Отсутствует упоминание интерфейса из контекста"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"

    logging.info("\nСгенерированная документация:")
    logging.info(doc)


def test_documentation_with_implementation_context(ollama_client):
    """Тест генерации документации с учетом контекста реализации"""
    # Интерфейс для документирования
    interface_code = """
    interface PaymentProcessor {
        fun processPayment(amount: Double): Boolean
        fun validatePayment(amount: Double): Boolean
    }
    """
    
    # Контекст - реализация интерфейса
    context = {
        "Реализация": """
        class StripePaymentProcessor : PaymentProcessor {
            override fun processPayment(amount: Double): Boolean {
                return if (validatePayment(amount)) {
                    // Обработка платежа через Stripe
                    true
                } else {
                    false
                }
            }
            
            override fun validatePayment(amount: Double): Boolean {
                return amount > 0 && amount < 1000000
            }
        }
        """
    }
    
    result = ollama_client.analyze_code(interface_code, "kotlin", context=context)
    
    logging.info("\nРезультаты анализа кода:")
    log_metrics(result["metrics"])
    model_info = ollama.show(ollama_client.model)
    log_model_info(model_info)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc с учетом контекста реализации
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@see" in doc, "Отсутствуют ссылки на связанные классы"
    assert "Stripe" in doc, "Отсутствует информация о реализации из контекста"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_context_size_calculation(ollama_client):
    """Тест корректности расчета размера контекста"""
    # Контекст с фиксированным размером
    context = {
        "Файл1": "A" * 100,  # 100 байт
        "Файл2": "B" * 100   # 100 байт
    }
    
    code = "class Test {}"
    result = ollama_client.analyze_code(code, "kotlin", context=context)
    
    # Проверяем метрики
    assert "metrics" in result, "Отсутствуют метрики"
    metrics = result["metrics"]
    
    # Проверяем базовые метрики
    assert metrics["time"] > 0, "Некорректное время выполнения"
    assert metrics["tokens"] > 0, "Некорректное количество токенов"
    assert metrics["prompt_tokens"] > 0, "Некорректное количество токенов промпта"
    assert metrics["completion_tokens"] >= 0, "Некорректное количество токенов ответа"
    assert metrics["speed"] > 0, "Некорректная скорость обработки"


def test_documentation_with_multiple_contexts(ollama_client):
    """Тест генерации документации с множественным контекстом"""
    main_code = """
    class OrderProcessor {
        private val paymentService: PaymentService
        private val notificationService: NotificationService
        
        fun processOrder(order: Order): Boolean {
            return if (paymentService.processPayment(order.total)) {
                notificationService.notify(order.userId, "Заказ оплачен")
                true
            } else {
                false
            }
        }
    }
    """
    
    context = {
        "Сервис оплаты": """
        interface PaymentService {
            fun processPayment(amount: Double): Boolean
        }
        """,
        "Сервис уведомлений": """
        interface NotificationService {
            fun notify(userId: String, message: String)
        }
        """,
        "Модель заказа": """
        data class Order(
            val id: String,
            val userId: String,
            val total: Double,
            val items: List<OrderItem>
        )
        """
    }
    
    result = ollama_client.analyze_code(main_code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем упоминание основных зависимостей
    assert "PaymentService" in doc, "Отсутствует упоминание PaymentService"
    assert "NotificationService" in doc, "Отсутствует упоминание NotificationService"
    assert "Order" in doc, "Отсутствует упоминание Order"
    
    # Проверяем связи между компонентами
    assert "processPayment" in doc, "Отсутствует упоминание метода processPayment"
    assert "notify" in doc, "Отсутствует упоминание метода notify"
    
    # Проверяем структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_documentation_with_empty_context(ollama_client):
    """Тест генерации документации с пустым контекстом"""
    code = """
    class SimpleClass {
        fun test() {}
    }
    """
    
    # Проверяем разные варианты пустого контекста
    empty_contexts = [
        None,  # Нет контекста
        {},    # Пустой словарь
        {"Context": ""},  # Пустая строка
        {"Context": None},  # None значение
        {"Context": "   "}  # Только пробелы
    ]
    
    for empty_context in empty_contexts:
        result = ollama_client.analyze_code(code, "kotlin", context=empty_context)
        
        # Проверяем базовую структуру ответа
        assert "documentation" in result, "Отсутствует документация"
        assert "metrics" in result, "Отсутствуют метрики"
        
        doc = result["documentation"]
        
        # Проверяем структуру KDoc
        assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
        assert "@constructor" in doc, "Отсутствует описание конструктора"
        
        # Проверяем обязательные секции
        assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
        assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_context_parameter_validation(ollama_client):
    """Тест валидации параметров контекста"""
    code = """
    class TestClass {
        fun test() {}
    }
    """
    
    # Тест с None
    result1 = ollama_client.analyze_code(code, "kotlin", context=None)
    assert "documentation" in result1, "Отсутствует документация при context=None"
    
    # Тест с пустым словарем
    result2 = ollama_client.analyze_code(code, "kotlin", context={})
    assert "documentation" in result2, "Отсутствует документация при пустом контексте"
    
    # Тест с некорректными значениями в контексте
    invalid_context = {
        "Файл1": None,
        "Файл2": "",
        "Файл3": "   ",
    }
    result3 = ollama_client.analyze_code(code, "kotlin", context=invalid_context)
    assert "documentation" in result3, "Отсутствует документация при некорректном контексте"


def test_documentation_quality_without_context(ollama_client):
    """Тест качества документации без контекстной информации"""
    code = """
    class CalcSum{
        fun mainCalc() {
            val numbers = Array(5) { i -> i * 2 }  // [0, 2, 4, 6, 8]
            println("Исходный массив: ${numbers.joinToString()}")
            val first = numbers.get(0)  // 0
            numbers.set(2, 10)         // Изменение элемента по индексу
            println("Элемент по индексу 0: $first")
            println("Модифицированный массив: ${numbers.joinToString()}")
            calculateSum(5, 7)
        }

        fun calculateSum(a: Int, b: Int) {
            val result = a + b
            println("Сумма $a и $b: $result")
        }
    }
    """
    
    result = ollama_client.analyze_code(code, "kotlin")
    
    logging.info("\nРезультаты анализа кода:")
    log_metrics(result["metrics"])
    model_info = ollama.show(ollama_client.model)
    log_model_info(model_info)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем базовую структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    
    # Проверяем наличие описания методов
    assert "mainCalc" in doc, "Отсутствует документация метода mainCalc"
    assert "calculateSum" in doc, "Отсутствует документация метода calculateSum"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logging.info("\n=== Документация без контекста ===\n")
    logging.info(doc)


def test_documentation_quality_with_context(ollama_client):
    """Тест качества документации с контекстной информацией"""
    code = """
    class Calculator {
        fun mainCalc(numbers: Array<Int>): Int {
            return numbers.getOrNull(0) ?: 0
        }
        
        fun calculateSum(a: Int, b: Int): Int {
            return a + b
        }
    }
    """
    
    context = {
        "Описание работы": """
        Создание массива
        Используется конструктор Array для создания массива фиксированного размера.
        Доступ к элементам осуществляется через индексы.
        Безопасное получение элемента через getOrNull().
        """,
        "Особенности реализации": """
        Особенности реализации:
        - Синтаксис массивов в Kotlin
        - Использование лямбда-выражений
        - Безопасная работа с null через оператор ?:
        - Методы-расширения Array: getOrNull()
        """
    }
    
    result = ollama_client.analyze_code(code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем базовую структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    
    # Проверяем наличие информации из контекста
    assert "Array" in doc, "Отсутствует информация о работе с Array"
    assert "лямбда" in doc.lower(), "Отсутствует информация о лямбда-выражениях"
    assert "getOrNull" in doc, "Отсутствует информация о безопасных методах"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


# Добавляем функцию для логирования разделителей
def log_test_separator(test_name: str):
    """Добавляет разделитель между тестами для лучшей читаемости логов"""
    logging.info(f"\n{'='*50}")
    logging.info(f"Запуск теста: {test_name}")
    logging.info(f"{'='*50}\n")


@pytest.fixture(autouse=True)
def log_test_name(request):
    """Автоматически логирует название каждого теста"""
    log_test_separator(request.node.name)
