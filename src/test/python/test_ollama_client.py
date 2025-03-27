import logging
import sys
from pathlib import Path
from typing import Optional, Dict
import os

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


@pytest.mark.skip
def test_invalid_file_type(ollama_client):
    """Проверка обработки неподдерживаемых типов файлов"""
    with pytest.raises(ValueError, match="Поддерживается только Kotlin"):
        ollama_client.analyze_code("...", "java")


@pytest.mark.skip
def test_missing_code(ollama_client):
    """Тест на пустой код"""
    result = ollama_client.analyze_code("", "kotlin")
    assert "error" in result, "Не обнаружена ошибка пустого кода"
    assert "пустой код" in result['error'], "Неверное сообщение об ошибке"


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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
        tokens_per_second = metrics["total_tokens"] / (
                    metrics["total_duration"] / 1e9)  # наносекунды в секунды
        logging.info(f"- Скорость обработки: {tokens_per_second:.2f} токенов/сек")
        logging.info(
            f"- Среднее время на токен: {(metrics['total_duration'] / 1e9 / metrics['total_tokens']) * 1000:.2f} мс")


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


def test_documentation_with_context(ollama_client, caplog):
    """Тест генерации документации с учетом контекста"""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Начало теста документации с контекстом интерфейсов")
    
    # Основной код для документирования
    main_code = """
    class UserRepository {
        private val userDao: UserDao
        
        fun getUser(id: String): User? {
            return userDao.findById(id)
        }
        
        fun saveUser(user: User) {
            userDao.save(user)
        }
    }
    """

    # Контекст - интерфейс и связанные классы
    context = {
        "Описание интерфейса": """
        UserDao - основной интерфейс для работы с данными пользователей.
        Предоставляет базовые операции CRUD для сущности User.
        """,
        "Интерфейс": """
        interface UserDao {
            // Поиск пользователя по ID
            fun findById(id: String): User?
            // Сохранение пользователя
            fun save(user: User)
            // Удаление пользователя
            fun delete(id: String)
        }
        """,
        "Модель данных": """
        // Модель пользователя с основными полями
        data class User(
            val id: String,
            val name: String,
            val email: String
        )
        """
    }

    logger.info("\nАнализируемый код:")
    logger.info(main_code)
    
    logger.info("\nКонтекст:")
    for section, content in context.items():
        logger.info(f"\n{section}:")
        logger.info(content)

    result = ollama_client.analyze_code(main_code, "kotlin", context=context)

    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    logger.info("\nСгенерированная документация:")
    logger.info(doc)

    # Проверяем структуру KDoc с учетом контекста
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "@property" in doc, "Отсутствует описание свойств"
    assert "@constructor" in doc, "Отсутствует описание конструктора"
    assert "UserDao" in doc, "Отсутствует упоминание интерфейса из контекста"
    assert "User" in doc, "Отсутствует упоминание модели данных"
    assert "findById" in doc, "Отсутствует описание метода findById"
    assert "save" in doc, "Отсутствует описание метода save"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logger.info("\nТест успешно завершен")


def test_documentation_with_implementation_context(ollama_client, caplog):
    """Тест генерации документации с учетом контекста реализации"""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Начало теста документации с контекстом реализации")
    
    # Интерфейс для документирования
    interface_code = """
    interface PaymentProcessor {
        /**
         * Обрабатывает платеж на указанную сумму
         * @param amount сумма платежа
         * @return true если платеж успешен, false в противном случае
         */
        fun processPayment(amount: Double): Boolean
        
        /**
         * Проверяет валидность суммы платежа
         * @param amount сумма для проверки
         * @return true если сумма валидна, false в противном случае
         */
        fun validatePayment(amount: Double): Boolean
    }
    """
    
    logger.info("\nАнализируемый код:")
    logger.info(interface_code)

    # Контекст - реализация интерфейса
    context = {
        "Реализация": """
        class StripePaymentProcessor : PaymentProcessor {
            override fun processPayment(amount: Double): Boolean {
                return if (validatePayment(amount)) {
                    // Обработка платежа через Stripe API
                    stripeClient.charge(amount)
                    true
                } else {
                    false
                }
            }
            
            override fun validatePayment(amount: Double): Boolean {
                return amount > 0 && amount < 1000000
            }
            
            private val stripeClient = StripeClient()
        }
        """
    }
    
    logger.info("\nКонтекст реализации:")
    for section, content in context.items():
        logger.info(f"\n{section}:")
        logger.info(content)

    result = ollama_client.analyze_code(interface_code, "kotlin", context=context)

    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    logger.info("\nСгенерированная документация:")
    logger.info(doc)

    # Проверяем только базовую структуру KDoc и наличие основных элементов
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    
    # Проверяем наличие хотя бы одного из методов
    methods = ["processPayment", "validatePayment"]
    assert any(method in doc for method in methods), "Отсутствует описание методов интерфейса"
    
    # Проверяем наличие хотя бы одного упоминания о платежах
    payment_terms = ["payment", "платеж", "сумма", "amount"]
    assert any(term.lower() in doc.lower() for term in payment_terms), "Отсутствует описание работы с платежами"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logger.info("\nТест успешно завершен")


@pytest.mark.skip
def test_context_size_calculation(ollama_client):
    """Тест корректности расчета размера контекста"""
    # Контекст с фиксированным размером
    context = {
        "Файл1": "A" * 100,  # 100 байт
        "Файл2": "B" * 100  # 100 байт
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


def test_documentation_with_multiple_contexts(ollama_client, caplog):
    """Тест генерации документации с множественным контекстом"""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Начало теста документации с множественным контекстом")
    
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
    
    logger.info("\nАнализируемый код:")
    logger.info(main_code)

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
    
    logger.info("\nКонтексты:")
    for section, content in context.items():
        logger.info(f"\n{section}:")
        logger.info(content)

    result = ollama_client.analyze_code(main_code, "kotlin", context=context)

    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    logger.info("\nСгенерированная документация:")
    logger.info(doc)

    # Проверяем структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "OrderProcessor" in doc, "Отсутствует название основного класса"
    assert "PaymentService" in doc, "Отсутствует упоминание PaymentService"
    assert "NotificationService" in doc, "Отсутствует упоминание NotificationService"
    assert "Order" in doc, "Отсутствует упоминание Order"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logger.info("\nТест успешно завершен")


@pytest.mark.skip
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
        {},  # Пустой словарь
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


@pytest.mark.skip
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


def test_documentation_quality_without_context(ollama_client, caplog):
    """Тест качества документации без контекстной информации"""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Начало теста качества документации без контекста")
    
    code = """
    class Calculator {
        fun calculate(a: Int, b: Int): Int {
            return a + b
        }
        
        fun multiply(a: Int, b: Int): Int {
            return a * b
        }
    }
    """
    
    logger.info("\nАнализируемый код:")
    logger.info(code)

    result = ollama_client.analyze_code(code, "kotlin")

    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    logger.info("\nСгенерированная документация:")
    logger.info(doc)

    # Проверяем базовую структуру KDoc
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    assert "Calculator" in doc, "Отсутствует название класса"
    assert "calculate" in doc, "Отсутствует документация метода calculate"
    assert "multiply" in doc, "Отсутствует документация метода multiply"
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logger.info("\nТест успешно завершен")


def test_documentation_quality_with_context(ollama_client, caplog):
    """Тест качества документации с контекстной информацией"""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Начало теста документации с контекстом лямбда-выражений")
    
    code = """
    class Calculator {
        fun processNumbers(numbers: Array<Int>): List<Int> {
            return numbers
                .filter { it > 0 }                    // Фильтрация положительных чисел
                .map { it * 2 }                       // Умножение каждого числа на 2
                .takeWhile { it < 100 }              // Взять числа меньше 100
        }
        
        fun calculateSum(a: Int, b: Int): Int = a + b  // Лямбда для сложения
    }
    """
    
    logger.info("\nАнализируемый код:")
    logger.info(code)

    context = {
        "Описание работы": """
        Обработка массива с использованием лямбда-выражений:
        1. Фильтрация элементов через лямбда-функцию filter
        2. Трансформация данных через лямбда-функцию map
        3. Ограничение выборки через лямбда-функцию takeWhile
        4. Использование однострочной лямбда-функции для calculateSum
        """,
        "Особенности реализации": """
        Ключевые особенности:
        - Активное использование лямбда-выражений для обработки коллекций
        - Цепочки вызовов с лямбда-функциями (filter, map, takeWhile)
        - Однострочные лямбда-выражения для простых операций
        - Безопасная работа с массивами через функции-расширения
        """
    }
    
    logger.info("\nКонтекст для анализа:")
    for section, content in context.items():
        logger.info(f"\n{section}:")
        logger.info(content)

    result = ollama_client.analyze_code(code, "kotlin", context=context)
    
    assert "documentation" in result, "Отсутствует документация"
    doc = result["documentation"]
    
    logger.info("\nСгенерированная документация:")
    logger.info(doc)

    # Проверяем только базовую структуру KDoc и основные элементы
    assert "/**" in doc and "*/" in doc, "Неверный формат KDoc"
    
    # Проверяем наличие хотя бы одного упоминания о ключевых концепциях
    key_concepts = ["Array", "List", "filter", "map", "takeWhile"]
    assert any(concept in doc for concept in key_concepts), "Отсутствует описание основных концепций"
    
    # Проверяем обязательные секции
    assert "Внешние зависимости:" in doc, "Отсутствует секция внешних зависимостей"
    assert "Взаимодействие:" in doc, "Отсутствует секция взаимодействия"
    
    logger.info("\nТест успешно завершен")


# Добавляем функцию для логирования разделителей
def log_test_separator(test_name: str):
    """Добавляет разделитель между тестами для лучшей читаемости логов"""
    logging.info(f"\n{'=' * 50}")
    logging.info(f"Запуск теста: {test_name}")
    logging.info(f"{'=' * 50}\n")


@pytest.fixture(autouse=True)
def log_test_name(request):
    """Автоматически логирует название каждого теста"""
    log_test_separator(request.node.name)

def run_test_with_logging(test_func, ollama_client, caplog) -> bool:
    """Запускает тест и возвращает результат его выполнения"""
    try:
        test_func(ollama_client, caplog)
        return True
    except AssertionError as e:
        logging.error(f"Тест {test_func.__name__} не прошел: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Ошибка в тесте {test_func.__name__}: {str(e)}")
        return False

def clear_logs():
    """Очистка лог-файлов перед запуском тестов"""
    log_files = [
        "ollama_client.log",
        "stability_test.log"
    ]
    
    for log_file in log_files:
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("")
            logging.info(f"Лог-файл очищен: {log_file}")
        except Exception as e:
            logging.warning(f"Не удалось очистить лог-файл {log_file}: {e}")

def test_stability(ollama_client, caplog):
    """Тест стабильности выполнения основных тестов"""
    # Очищаем логи перед запуском
    clear_logs()
    
    # Настраиваем логирование в файл
    log_file = "stability_test.log"
    
    # Форматтер для файла
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Цветной форматтер для консоли
    class ColoredFormatter(logging.Formatter):
        """Форматтер с цветным выводом для разных уровней логирования"""
        COLORS = {
            'DEBUG': '\033[37m',  # Серый
            'INFO': '\033[32m',   # Зеленый
            'WARNING': '\033[33m', # Желтый
            'ERROR': '\033[31m',   # Красный
            'CRITICAL': '\033[41m' # Красный фон
        }
        RESET = '\033[0m'

        def format(self, record):
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname_colored = f"{color}{record.levelname:<8}{self.RESET}"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(levelname_colored)s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    try:
        # Список тестов для проверки стабильности
        tests_to_check = [
            test_documentation_with_context,
            test_documentation_with_implementation_context,
            test_documentation_with_multiple_contexts,
            test_documentation_quality_with_context,
            test_documentation_quality_without_context
        ]
        
        iterations = 10  # Количество прогонов каждого теста
        results = {test.__name__: {"passed": 0, "failed": 0} for test in tests_to_check}
        
        logger.info(f"\nНачало проверки стабильности тестов ({iterations} итераций)")
        logger.info("=" * 50)
        
        for iteration in range(iterations):
            logger.info(f"\nИтерация {iteration + 1}/{iterations}")
            logger.info("-" * 30)
            
            for test in tests_to_check:
                test_name = test.__name__
                logger.info(f"\nЗапуск теста: {test_name}")
                
                success = run_test_with_logging(test, ollama_client, caplog)
                if success:
                    results[test_name]["passed"] += 1
                    logger.info(f"Тест {test_name} успешно пройден")
                else:
                    results[test_name]["failed"] += 1
                    logger.error(f"Тест {test_name} не пройден")
        
        # Вывод статистики
        logger.info("\nРезультаты проверки стабильности:")
        logger.info("=" * 50)
        
        all_stable = True
        for test_name, stats in results.items():
            total = stats["passed"] + stats["failed"]
            success_rate = (stats["passed"] / total) * 100
            logger.info(f"\nТест: {test_name}")
            logger.info(f"Успешно: {stats['passed']}/{total} ({success_rate:.1f}%)")
            logger.info(f"Неудачно: {stats['failed']}/{total}")
            
            # Проверяем стабильность
            if success_rate < 80:
                all_stable = False
                logger.error(f"Тест {test_name} нестабилен (успешность {success_rate:.1f}%)")
        
        logger.info("\nПроверка стабильности завершена")
        
        assert all_stable, "Обнаружены нестабильные тесты, смотрите лог для деталей"
        
    finally:
        # Удаляем handlers
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)
        file_handler.close()
        
        # Выводим путь к файлу с логами
        abs_path = os.path.abspath(log_file)
        print(f"\nЛоги теста стабильности сохранены в: {abs_path}")
