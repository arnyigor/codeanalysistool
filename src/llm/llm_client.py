import hashlib
import json
import logging
import os
import re
import time
from typing import Dict, Optional

import ollama

LOG_FILE = os.path.abspath("ollama_client.log")

def setup_logging():
    """Настройка логгера для тестов"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Форматтер для файла
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Очистка файла перед запуском
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("")

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

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
    logger.addHandler(console_handler)


# Настройка логгирования перед тестами
setup_logging()


class OllamaClient:
    def __init__(self):
        """
        Инициализация клиента Ollama.
        Использует первую доступную запущенную модель.
        """
        try:
            models = ollama.list()
            available_models = models.get('models', [])
            if not available_models:
                raise RuntimeError("Нет запущенных моделей Ollama")

            # Проверяем наличие обязательных полей
            model_info = available_models[0]

            if 'model' not in model_info:
                raise ValueError("Некорректный формат данных модели: отсутствует поле 'name'")

            self.model = model_info['model']
            # Получаем детальную информацию о модели
            model_details = ollama.show(self.model)

            # Форматируем размер модели
            size_bytes = int(model_info.get('size', 0))
            size_gb = size_bytes / (1024 * 1024 * 1024)

            # Получаем размер контекста
            model_params = model_details.get('model_info', {})
            self.context_length = None

            # Ищем любой ключ, содержащий context_length
            for key in model_params.keys():
                if 'context_length' in key:
                    self.context_length = model_params[key]
                    break

            if self.context_length is None:
                self.context_length = 8192  # Безопасное значение по умолчанию
                logging.warning(
                    "Не удалось определить размер контекста модели, используем значение по умолчанию")

            # Логируем основную информацию о модели
            logging.info("Используется модель Ollama:")
            logging.info(f"- Название: {self.model}")
            logging.info(f"- Размер модели: {size_gb:.2f} GB")
            logging.info(f"- Размер контекста: {self.context_length} токенов")

            # Сохраняем структуру
            self.model_details = model_details

        except KeyError as e:
            logging.error("Отсутствует обязательное поле в данных модели: {}".format(str(e)),
                          stack_info=True)
        except Exception as e:
            logging.error("Ошибка при проверке моделей: {}".format(str(e)), stack_info=True)
            raise RuntimeError(
                "Ошибка при получении списка моделей Ollama. Убедитесь, что:\n"
                "1. Сервис Ollama запущен\n"
                "2. Хотя бы одна модель активна"
            )

    def _estimate_doc_size(self, code: str) -> int:
        """Оценивает размер необходимой документации."""
        lines_count = len(code.splitlines())

        # В среднем на каждые 3 строки кода - 1 строка документации
        # Каждая строка документации ~ 20 токенов
        estimated_tokens = (lines_count // 3) * 20

        # Минимум 200 токенов, максимум 3000
        return max(200, min(estimated_tokens, 3000))

    def _get_model_params(self, code: str, prompt_size: int, file_type: str, context: Optional[Dict[str, str]] = None) -> dict:
        """Формирует параметры запроса к модели."""
        # Базовый системный промпт
        BASE_SYSTEM_PROMPT_SIZE = 500  # ~500 токенов для базового системного промпта
        
        # Оцениваем размеры в токенах (примерно 3 байта на токен)
        code_tokens = len(code.encode()) // 3
        template_tokens = prompt_size // 3
        
        # Оцениваем размер системного промпта с контекстом
        system_tokens = BASE_SYSTEM_PROMPT_SIZE
        context_size = 0
        
        # Обработка контекста
        if context:
            valid_context = {
                title: content.strip() 
                for title, content in context.items() 
                if content is not None and content.strip()
            }
            if valid_context:
                # Добавляем токены на заголовки и форматирование
                context_size = sum(len(str(content).encode()) // 3 for content in valid_context.values())
                context_size += len(valid_context) * 20  # ~20 токенов на заголовки и форматирование
                system_tokens += context_size
                
                # Логируем информацию о контексте
                logging.info("\nКонтекст:")
                for title, content in valid_context.items():
                    preview = content[:50] + "..." if len(content) > 50 else content
                    logging.info(f"- {title}:\n{preview}")
        
        # Общий размер входных данных
        total_input_tokens = code_tokens + template_tokens + system_tokens
        
        # Определяем базовый размер контекста (минимум 2048)
        base_context = max(2048, total_input_tokens * 2)  # x2 для места под ответ
        
        # Добавляем буфер, который растет с размером входных данных
        buffer_size = min(1000, total_input_tokens // 4)  # Адаптивный буфер
        
        # Определяем максимальный безопасный размер контекста (80% от максимума модели)
        max_safe_context = int(self.context_length * 0.8)
        
        # Вычисляем оптимальный размер контекста
        OPTIMAL_CONTEXT = min(base_context + buffer_size, max_safe_context)

        # Оцениваем размер документации
        doc_tokens = self._estimate_doc_size(code)
        if context_size > 0:
            doc_tokens = int(doc_tokens * 1.5)  # Увеличиваем размер для контекстной документации

        # Устанавливаем параметры
        params = {
            "temperature": 0.2,
            "top_p": 0.7,
            "num_tokens": doc_tokens + 200,  # Базовый размер + буфер
            "context_length": OPTIMAL_CONTEXT
        }

        # Логируем параметры
        logging.info("\nПараметры запроса к модели:")
        logging.info(f"- Размер контекста: {OPTIMAL_CONTEXT} токенов")
        logging.info(f"- Базовый размер контекста: {base_context} токенов")
        logging.info(f"- Размер буфера: {buffer_size} токенов")
        logging.info(f"- Максимальный безопасный контекст: {max_safe_context} токенов")
        logging.info(f"- Размер кода: {code_tokens} токенов")
        logging.info(f"- Размер шаблона: {template_tokens} токенов")
        logging.info(f"- Размер системного промпта: {system_tokens} токенов")
        if context_size > 0:
            logging.info(f"- Размер дополнительного контекста: {context_size} токенов")
        logging.info(f"- Общий размер входных данных: {total_input_tokens} токенов")
        logging.info(f"- Оценка документации: {doc_tokens} токенов")
        logging.info(f"- Тип файла: {file_type}")
        logging.info(f"- Температура: {params['temperature']}")
        logging.info(f"- Top-p: {params['top_p']}")
        logging.info(f"- Предсказание токенов: {params['num_tokens']}")

        return params

    def _log_model_response(self, response: Dict, elapsed_time: float, metrics: Dict):
        """Логирует информацию о работе модели после запроса."""
        logging.info("\nИнформация о выполнении запроса:")
        
        # Логируем метрики
        logging.info(f"- Общее время: {elapsed_time:.2f} сек")
        logging.info(f"- Токены промпта: {metrics['prompt_tokens']}")
        logging.info(f"- Токены ответа: {metrics['completion_tokens']}")
        logging.info(f"- Всего токенов: {metrics['tokens']}")
        logging.info(f"- Скорость: {metrics['speed']:.2f} токенов/сек")
        logging.info(f"- Среднее время на токен: {(elapsed_time * 1000) / metrics['tokens']:.2f} мс")
        
        # Информация о памяти
        if 'memory' in response:
            memory = response['memory']
            if isinstance(memory, dict):
                logging.info("\nИспользование памяти:")
                for key, value in memory.items():
                    if isinstance(value, (int, float)):
                        value_mb = value / (1024 * 1024)  # Конвертируем в МБ
                        logging.info(f"- {key}: {value_mb:.2f} МБ")

    def analyze_code(self, code: str, file_type: str, context: Optional[Dict[str, str]] = None) -> Optional[dict]:
        """Анализирует код и генерирует документацию."""
        # Проверяем тип файла
        if file_type != 'kotlin':
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}. Поддерживается только Kotlin.")

        # Проверяем наличие кода
        if not code or not code.strip():
            return self._create_error_response("пустой код")

        logging.info(f"\n{'=' * 50}\nНачало анализа кода\n{'=' * 50}")
        logging.info(f"Тип файла: {file_type}")

        try:
            # Формируем системный промпт
            system_prompt = ""

            # Добавляем контекстную информацию если она есть
            if context:
                valid_context = {
                    title: content.strip() 
                    for title, content in context.items() 
                    if content is not None and content.strip()
                }
                if valid_context:
                    system_prompt += "=== Контекстная информация ===\n"
                    for title, content in valid_context.items():
                        system_prompt += f"\n--- {title} ---\n{content}\n\n"

            # Добавляем основной системный промпт
            system_prompt += """Вы - опытный разработчик, создающий документацию ИСКЛЮЧИТЕЛЬНО на РУССКОМ языке в формате KDoc.  
Ваша задача - добавить **полную и строгую документацию** к классу, используя только следующие аннотации:  
`@property`, `@constructor`, `@param`, `@return`, `@see`

Важные требования:
1. Вернуть ТОЛЬКО документацию класса, БЕЗ КОДА
2. НЕ создавать отдельную документацию для методов
3. Описывать функциональность методов в общем описании класса

При наличии контекстной информации:
1. Учитывайте связи между компонентами
2. Отражайте зависимости в документации
3. Используйте информацию о реализации
4. Добавляйте ссылки на связанные классы
5. Включайте информацию о взаимодействии с другими компонентами
6. Описывайте особенности реализации из контекста"""

            # Создаем промпт с кодом и контекстом
            prompt = self._create_documentation_prompt(code, file_type, context)

            # Получаем адаптивные параметры запроса
            model_params = self._get_model_params(code, len(prompt.encode()), file_type, context)

            logging.info("\nОтправляем запрос к модели...")
            start_time = time.time()

            # Отправляем запрос к модели с системным промптом
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                options=model_params
            )
            elapsed_time = time.time() - start_time

            # Получаем метрики из ответа
            context = response.get('context', [])
            prompt_eval_count = len(context) if isinstance(context, list) else 0
            eval_count = response.get('eval_count', 0)
            
            # Вычисляем токены
            completion_tokens = eval_count - prompt_eval_count if eval_count > prompt_eval_count else 0
            total_tokens = prompt_eval_count + completion_tokens
            
            # Скорость обработки
            tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

            # Формируем метрики
            metrics = {
                "time": round(elapsed_time, 2),
                "tokens": total_tokens,
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": completion_tokens,
                "speed": round(tokens_per_second, 2)
            }

            # Логируем информацию о работе модели
            self._log_model_response(response, elapsed_time, metrics)

            # Получаем документацию из ответа
            documentation = response.get('response', '').strip()

                # Проверяем наличие документации
            if not documentation or not "/**" in documentation:
                documentation = self._create_empty_java_doc(code)
                logging.warning("Модель вернула некорректный ответ, создана базовая документация")

            # Сохраняем результат для тестов
            self._save_test_result(documentation, file_type)

            # Формируем результат
            result = {
                "documentation": documentation,
                    "status": "success",
                "metrics": metrics
            }

            return result

        except Exception as e:
            error_msg = f"Неизвестная ошибка: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)

    def _create_documentation_prompt(self, code: str, file_type: str, context: Optional[Dict[str, str]] = None) -> str:
        """Создает промпт для генерации документации."""
        if file_type != 'kotlin':
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}. Поддерживается только Kotlin.")

        prompt_template = f"""[ПРАВИЛА ДОКУМЕНТИРОВАНИЯ KDOC]
#### ПРАВИЛА ДОКУМЕНТИРОВАНИЯ:
**Документация класса:**
   - Начните с подробного описания назначения класса и его роли в системе.
   - Опишите основные методы класса и их функциональность в общем описании.
   - Используйте `@property` для описания полей с указанием их целей.
   - В `@constructor` укажите все параметры конструктора и их назначение.
   - Добавьте `@see` для связанных классов, интерфейсов или методов.
   - В конце добавьте **секции**:
     - **Внешние зависимости:** список библиотек/классов, напрямую используемых классом.
     - **Взаимодействие:** описание, как класс взаимодействует с другими компонентами.

#### ДОПОЛНИТЕЛЬНЫЕ УСЛОВИЯ:
- Если в коде уже присутствует KDoc, обновите его:
  1. Уберите запрещённые элементы (`@author`, `@version`).
  2. Добавьте недостающие аннотации (например, `@see`).
  3. Дополните разделы "Внешние зависимости" и "Взаимодействие".
  4. Сохраните полезные части старого KDoc, если они корректны.
  5. Исправьте ошибки в описаниях.

- Если KDoc отсутствует — создайте его с нуля по указанным правилам.

#### ПОРЯДОК ВЫВОДА:
```
/**
 * [Краткое описание класса]
 * 
 * [Подробное описание класса, включая описание основных методов]
 * 
 * @property [название] [описание]
 * @property ...
 * 
 * @constructor [описание конструктора]
 * @param [параметр] [описание]
 * 
 * @see [ссылки]
 * 
 * **Внешние зависимости:**
 * - [зависимость 1]
 * - [зависимость 2]
 * 
 * **Взаимодействие:**
 * - [описание взаимодействия]
 */
```

#### ТРЕБОВАНИЯ:
- Документация **ТОЛЬКО** в формате KDoc (/** ... */).
- НЕ использовать `@author`, `@version`, `@since`.
- НЕ добавлять описания после свойств в коде.
- НЕ добавлять отдельную документацию для методов.
- НЕ включать код в ответ, только документацию.
- Внешние зависимости: только **напрямую используемые** классы/библиотеки.
- Взаимодействия: уточнить **как** (например, "вызывает метод X", "использует сервис Y").
- Вернуть **ТОЛЬКО документацию** (без кода, импортов и посторонних текстов).

[КОД]
{code}
[КОНЕЦ КОДА]

Пожалуйста, верните **ТОЛЬКО документацию** в указанном формате, строго соблюдая структуру и требования, БЕЗ КОДА.
Учитывайте контекстную информацию при её наличии."""

        return prompt_template

    def _create_error_response(self, error_message: str) -> Dict:
        """Создает структурированный ответ с ошибкой."""
        return {
            "error": error_message,
            "code": ""
        }

    def _get_cache_path(self, code: str, file_type: str) -> str:
        """Генерирует путь к кэш-файлу на основе хэша кода."""
        import hashlib
        import os

        # Создаем хэш кода
        code_hash = hashlib.md5(code.encode()).hexdigest()

        # Создаем директорию для кэша если её нет
        cache_dir = os.path.join(os.getcwd(), '.cache', 'docs')
        os.makedirs(cache_dir, exist_ok=True)

        # Возвращаем путь к файлу кэша
        return os.path.join(cache_dir, f"{code_hash}_{file_type}.json")

    def _save_to_cache(self, cache_path: str, result: dict):
        """Сохраняет результат в кэш."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"Результат сохранен в кэш: {cache_path}")
        except Exception as e:
            logging.warning(f"Не удалось сохранить кэш: {str(e)}")

    def _load_from_cache(self, cache_path: str) -> Optional[dict]:
        """Загружает результат из кэша."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    logging.info(f"Результат загружен из кэша: {cache_path}")
                    return result
            else:
                logging.info(f"Кэш не найден: {cache_path}")
        except Exception as e:
            logging.warning(f"Не удалось загрузить кэш: {str(e)}")
        return None

    def _save_test_result(self, documentation: str, file_type: str):
        """Сохраняет результат теста в файл."""
        try:
            # Создаем директорию для результатов тестов
            test_results_dir = os.path.join(os.getcwd(), '.cache','test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            # Генерируем уникальное имя файла
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_result_{timestamp}_{file_type}.txt"
            filepath = os.path.join(test_results_dir, filename)

            # Сохраняем документацию
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(documentation)

            logging.info(f"Результат сохранен в файл: {filepath}")

        except Exception as e:
            logging.error(f"Ошибка при сохранении результата теста: {str(e)}")

    def _create_empty_java_doc(self, code: str) -> str:
        """Создает пустую Java документацию для класса."""
        class_name = re.search(r'class\s+(\w+)', code)
        if not class_name:
            return "/** Документация отсутствует */"

        return f"""/**
 * Класс {class_name.group(1)}.
 * 
 * @constructor Создает новый экземпляр класса {class_name.group(1)}.
 *
 * **Внешние зависимости:**
 * - Нет
 *
 * **Взаимодействие:**
 * - Нет
 */"""
