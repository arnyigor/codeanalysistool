import hashlib
import json
import logging
import os
import re
import time
from typing import Dict, Optional

import ollama


class OllamaClient:
    def __init__(self):
        """
        Инициализация клиента Ollama.
        Использует первую доступную запущенную модель.
        """
        # Настраиваем логирование
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

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

    def _get_model_params(self, code: str, template_size: int, file_type: str) -> dict:
        """Формирует параметры запроса к модели."""
        # Используем меньший контекст для документации
        OPTIMAL_CONTEXT = 2048

        # Оцениваем размеры в токенах (примерно 3 байта на токен)
        input_tokens = (len(code.encode()) + template_size) // 3

        # Оцениваем размер документации (примерно 1/3 от размера кода)
        doc_tokens = max(200, min(input_tokens // 3, 1000))

        # Устанавливаем параметры
        options = {
            "temperature": 0.2,  # Более низкая температура для более предсказуемой документации
            "top_p": 0.7,
            "num_predict": doc_tokens + 200,  # Базовый размер + небольшой буфер
            "num_ctx": OPTIMAL_CONTEXT
        }

        # Логируем параметры
        logging.info("\nПараметры запроса к модели:")
        logging.info(f"- Размер контекста: {OPTIMAL_CONTEXT} токенов")
        logging.info(f"- Входной контекст: {input_tokens} токенов")
        logging.info(f"- Оценка документации: {doc_tokens} токенов")
        logging.info(f"- Тип файла: {file_type}")
        logging.info(f"- Температура: {options['temperature']}")
        logging.info(f"- Top-p: {options['top_p']}")
        logging.info(f"- Предсказание токенов: {options['num_predict']}")

        return {"options": options}

    def analyze_code(self, code: str, file_type: str) -> Optional[dict]:
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
            # Системный промпт для модели
            system_prompt = """Вы - опытный разработчик, создающий документацию ИСКЛЮЧИТЕЛЬНО на РУССКОМ языке в формате KDoc.  
Ваша задача - добавить **полную и строгую документацию** к классу, используя только следующие аннотации:  
`@property`, `@constructor`, `@param`, `@return`, `@see`
Вернуть ТОЛЬКО документацию, БЕЗ КОДА"""

            # Создаем промпт с кодом
            prompt = self._create_documentation_prompt(code, file_type)

            # Получаем адаптивные параметры запроса
            model_params = self._get_model_params(code, len(prompt.encode()), file_type)

            logging.info("\nОтправляем запрос к модели...")
            start_time = time.time()

            # Отправляем запрос к модели с системным промптом
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                **model_params
            )

            # Получаем метрики
            elapsed_time = time.time() - start_time
            total_tokens = response.get('total_tokens', 0)
            tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

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
                "metrics": {
                    "time": round(elapsed_time, 2),
                    "tokens": total_tokens,
                    "speed": round(tokens_per_second, 2)
                }
            }

            return result

        except Exception as e:
            error_msg = f"Неизвестная ошибка: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)

    def _create_documentation_prompt(self, code: str, file_type: str) -> str:
        """Создает промпт для генерации документации."""
        if file_type != 'kotlin':
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}. Поддерживается только Kotlin.")

        prompt_template = """[ПРАВИЛА ДОКУМЕНТИРОВАНИЯ KDOC]
#### ПРАВИЛА ДОКУМЕНТИРОВАНИЯ:
**Документация классов:**
   - Начните с подробного описания назначения класса и его роли в системе.
   - Используйте `@property` для описания полей с указанием их целей.
   - В `@constructor` укажите все параметры конструктора и их назначение.
   - Добавьте `@see` для связанных классов, интерфейсов или методов.
   - В конце добавьте **секции**:
     - **Внешние зависимости:** список библиотек/классов, напрямую используемых классом (без родительских классов/интерфейсов).
     - **Взаимодействие:** описание, как класс взаимодействует с другими компонентами (например, "вызывает X.method() для Y").

#### ДОПОЛНИТЕЛЬНЫЕ УСЛОВИЯ:
- Если в коде уже присутствует KDoc, обновите его:
  1. Уберите запрещённые элементы (`@author`, `@version`).
  2. Добавьте недостающие аннотации (например, `@see`).
  3. Дополните разделы "Внешние зависимости" и "Взаимодействие".
  4. Сохраните полезные части старого KDoc, если они корректны.
  5. Исправьте ошибки в описаниях (например, устаревшие методы).

- Если KDoc отсутствует — создайте его с нуля по указанным правилам.


#### ПОРЯДОК ВЫВОДА:
Для класса:
```
/**
 * [Краткое описание класса]
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
- Внешние зависимости: только **напрямую используемые** классы/библиотеки (например, `ILandingsBuilderInteractor`, `Timber`).
- Взаимодействия: уточнить **как** (например, "вызывает метод X", "использует сервис Y").
- Вернуть **ТОЛЬКО документацию** (без кода, импортов и посторонних текстов).

[КОД]
{code}
[КОНЕЦ КОДА]
Пожалуйста, верните **ТОЛЬКО документацию** в указанном формате, строго соблюдая структуру и требования, БЕЗ КОДА."""

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
 */"""
