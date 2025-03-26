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
    
    result = ollama_client.analyze_code(main_code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc с учетом контекста
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    assert "UserDao" in doc, "Отсутствует упоминание интерфейса из контекста"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


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
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем структуру KDoc с учетом контекста реализации
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@see" in doc, "Отсутствуют ссылки на связанные классы"
    assert "Stripe" in doc, "Отсутствует информация о реализации из контекста"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"


def test_context_size_calculation(ollama_client):
    """Тест расчета размера контекста при наличии дополнительной информации"""
    code = """
    class Logger {
        fun log(msg: String) {}
    }
    """
    
    large_context = {
        "Файл1": "A" * 1000,  # Большой контекст
        "Файл2": "B" * 1000,
    }
    
    result = ollama_client.analyze_code(code, "kotlin", context=large_context)
    
    # Проверяем, что метрики содержат информацию о размере контекста
    assert "metrics" in result, "Отсутствуют метрики"
    assert "context_size" in result["metrics"], "Отсутствует информация о размере контекста"
    assert result["metrics"]["context_size"] > len(code), "Неверный расчет размера контекста"


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
        """,
        "Модель элемента заказа": """
        data class OrderItem(
            val productId: String,
            val quantity: Int,
            val price: Double
        )
        """
    }
    
    result = ollama_client.analyze_code(main_code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем упоминание всех контекстных элементов
    assert "PaymentService" in doc, "Отсутствует упоминание PaymentService"
    assert "NotificationService" in doc, "Отсутствует упоминание NotificationService"
    assert "Order" in doc, "Отсутствует упоминание Order"
    assert "OrderItem" in doc, "Отсутствует упоминание OrderItem"
    
    # Проверяем связи между компонентами
    assert "processPayment" in doc, "Отсутствует упоминание метода processPayment"
    assert "notify" in doc, "Отсутствует упоминание метода notify"


def test_documentation_with_empty_context(ollama_client):
    """Тест генерации документации с пустым контекстом"""
    code = """
    class SimpleLogger {
        fun log(message: String) {
            println(message)
        }
    }
    """
    
    # Передаем пустой контекст
    result = ollama_client.analyze_code(code, "kotlin", context={})
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем базовую структуру документации
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc or "@constructor" in doc, "Отсутствуют основные KDoc аннотации"
    
    # Проверяем метрики
    assert "metrics" in result, "Отсутствуют метрики"
    assert result["metrics"]["context_size"] == 0, "Неверный размер пустого контекста"


def test_documentation_with_large_context_handling(ollama_client):
    """Тест обработки очень большого контекста"""
    code = """
    class MetricsCollector {
        fun collect(): Map<String, Int> = mapOf()
    }
    """
    
    # Создаем большой контекст (более 10000 символов)
    large_context = {
        "Большой файл 1": "A" * 5000,
        "Большой файл 2": "B" * 5000,
        "Большой файл 3": "C" * 5000
    }
    
    result = ollama_client.analyze_code(code, "kotlin", context=large_context)
    
    assert "documentation" in result, "Отсутствует документация"
    assert "metrics" in result, "Отсутствуют метрики"
    
    # Проверяем, что размер контекста корректно рассчитан
    context_size = result["metrics"]["context_size"]
    assert context_size > 0, "Размер контекста должен быть больше 0"
    assert context_size >= 5000, "Неверный расчет размера большого контекста"
    
    # Проверяем, что документация все еще генерируется корректно
    doc = result["documentation"]
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc or "@constructor" in doc, "Отсутствуют основные KDoc аннотации"


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
    
    # Предоставляем подробный контекст о работе класса
    context = {
        "Описание работы": """
        Описание работы:
        Создание массива
        Используется конструктор Array с лямбда-выражением для инициализации элементов. 
        В данном случае каждый элемент равен индексу, умноженному на 2.

        Работа с элементами
        get(0) возвращает значение по индексу
        set(2, 10) изменяет элемент в позиции 2

        Пользовательская функция
        Функция calculateSum принимает два параметра, вычисляет их сумму и выводит результат. 
        Объявляется ключевым словом fun.

        Точка входа
        Функция main — точка старта программы. В ней демонстрируются основные операции 
        с массивами и вызов пользовательской функции.
        """,
        
        "Особенности реализации": """
        Особенности реализации:
        Синтаксис массивов
        В Kotlin массивы создаются через конструктор Array, а не через квадратные скобки как в Java.

        Обработка ошибок
        Для работы с индексами массива рекомендуется использовать безопасные методы 
        (например, getOrNull()), чтобы избежать исключений.

        Функциональные возможности
        Лямбда-выражения в конструкторе массива позволяют гибко инициализировать элементы.
        """
    }
    
    result = ollama_client.analyze_code(code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    # Проверяем базовую структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    
    # Проверяем наличие описания методов
    assert "mainCalc" in doc, "Отсутствует документация метода mainCalc"
    assert "calculateSum" in doc, "Отсутствует документация метода calculateSum"
    
    # Проверяем наличие информации из контекста
    assert "Array" in doc, "Отсутствует информация о работе с Array"
    assert "лямбда" in doc.lower(), "Отсутствует информация о лямбда-выражениях"
    assert "getOrNull" in doc, "Отсутствует информация о безопасных методах"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    # Проверяем метрики
    assert "metrics" in result, "Отсутствуют метрики"
    assert "context_size" in result["metrics"], "Отсутствует размер контекста"
    assert result["metrics"]["context_size"] > 0, "Неверный размер контекста"
    
    logging.info("\n=== Документация с контекстом ===\n")
    logging.info(doc)
