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
        # Фиксированный размер контекста для файлов до 1000 строк
        OPTIMAL_CONTEXT = 4096

        # Оцениваем размеры в токенах
        input_tokens = (len(code.encode()) + template_size) // 3

        # Оцениваем размер документации
        doc_tokens = self._estimate_doc_size(code)

        # Устанавливаем параметры
        options = {
            "temperature": 0.3,  # Более консервативное значение для документации
            "top_p": 0.7,
            "num_predict": doc_tokens + 500,  # Базовый размер + буфер
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
        if file_type not in ['java', 'kotlin']:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

        # Проверяем наличие кода
        if not code or not code.strip():
            return self._create_error_response("пустой код")

        logging.info(f"\n{'=' * 50}\nНачало анализа кода\n{'=' * 50}")
        logging.info(f"Тип файла: {file_type}")

        code_size = len(code.encode())
        logging.info(f"Размер кода: {code_size} байт")

        try:
            # Создаем промпт с кодом
            prompt = self._create_documentation_prompt(code, file_type)

            # Получаем адаптивные параметры запроса
            model_params = self._get_model_params(code, len(prompt.encode()), file_type)

            # Добавляем системный промпт к параметрам
            if "[КОД]" in prompt:
                system_prompt = prompt.split("[КОД]")[0].strip()
                model_params['system'] = system_prompt

            logging.info("\nОтправляем запрос к модели...")
            start_time = time.time()

            # Отправляем запрос к модели с полным промптом
            response = ollama.generate(
                model=self.model,
                prompt=prompt,  # Отправляем полный промпт с кодом
                **model_params
            )

            # Получаем метрики
            elapsed_time = time.time() - start_time
            total_tokens = response.get('total_tokens', 0)
            tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

            # Очищаем ответ
            clean_response = response.get('response', '').strip()
            logging.info(f"\nОтвет модели:\n{clean_response}")

            # Извлекаем документацию из ответа и очищаем от маркеров языка
            documentation = clean_response
            if "```" in clean_response:
                code_parts = clean_response.split("```")
                for part in code_parts:
                    if part.strip() and "/**" in part:
                        # Убираем маркер языка, если он есть
                        lines = part.strip().split('\n')
                        if lines[0].lower() in ['java', 'kotlin']:
                            documentation = '\n'.join(lines[1:])
                        else:
                            documentation = part.strip()
                        break

            # Если нет документации, создаем базовую
            if not "/**" in documentation:
                documentation = self._create_empty_java_doc(code)
                logging.warning("Модель вернула некорректный ответ, создана базовая документация")

            # Сохраняем результат для тестов
            self._save_test_result(code, documentation, file_type)

            # Формируем результат
            result = {
                "documentation": documentation,  # Только документация
                "code": code,  # Исходный код
                "status": "success",
                "metrics": {
                    "time": round(elapsed_time, 2),
                    "tokens": total_tokens,
                    "speed": round(tokens_per_second, 2)
                },
                "generation_info": {
                    "model": self.model,
                    "total_tokens": total_tokens,
                    "elapsed_time": elapsed_time,
                    "chunks_received": response.get('eval_count', 0),
                    "chunks_total": response.get('eval_duration', 0)
                }
            }

            return result

        except Exception as e:
            error_msg = f"Неизвестная ошибка: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)

    def _create_documentation_prompt(self, code: str, file_type: str) -> str:
        """Создает промпт для генерации документации."""
        # Выбираем правила документирования в зависимости от типа файла
        if file_type == "kotlin":
            doc_rules = """[ПРАВИЛА ДОКУМЕНТИРОВАНИЯ KDOC]
#### ПРАВИЛА ДОКУМЕНТИРОВАНИЯ:
1. **Документация классов:**
   - Начните с краткого описания назначения класса и его роли в системе.
   - Используйте `@property` для описания полей с указанием их целей.
   - В `@constructor` укажите все параметры конструктора и их назначение.
   - Добавьте `@see` для связанных классов, интерфейсов или методов.
   - В разделе `@throws` укажите все возможные исключения (или "Нет исключений").
   - В конце добавьте **секции**:
     - **Внешние зависимости:** список библиотек/классов, напрямую используемых классом (без родительских классов/интерфейсов).
     - **Взаимодействие:** описание, как класс взаимодействует с другими компонентами (например, "вызывает X.method() для Y").

2. **Документация методов:**
   - Описание должно отражать **логику** и **цель** метода.
   - Для каждого параметра в `@param` укажите тип, назначение и допустимые значения (если нужно).
   - В `@return` уточните возвращаемое значение (или "Нет возврата").
   - В `@throws` укажите исключения с условиями их возникновения.
   - В `@see` ссылайтесь на связанные методы или классы.
   - Добавьте описание **взаимодействий** с другими компонентами (например, "использует landingsBuilderInteractor.submit()").

#### ДОПОЛНИТЕЛЬНЫЕ УСЛОВИЯ:
- Если в коде уже присутствует KDoc, обновите его:
  1. Уберите запрещённые элементы (`@author`, `@version`).
  2. Добавьте недостающие аннотации (например, `@see`, `@throws`).
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
 * @throws [исключения] (или "Нет исключений")
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

Для метода:
```
/**
 * [Описание метода]
 * 
 * @param [параметр] [описание]
 * @return [описание возврата]
 * @throws [исключения]
 * 
 * **Взаимодействие:**
 * - [описание взаимодействия с компонентами]
 * 
 * @see [связанные методы/классы]
 */
```"""
        else:  # java
            doc_rules = """[ПРАВИЛА ДОКУМЕНТИРОВАНИЯ JAVADOC]
#### ПРАВИЛА ДОКУМЕНТИРОВАНИЯ:
1. **Документация классов:**
   - Начните с краткого описания назначения класса и его роли в системе.
   - Используйте `@property` для описания полей с указанием их целей.
   - В `@constructor` укажите все параметры конструктора и их назначение.
   - Добавьте `@see` для связанных классов, интерфейсов или методов.
   - В разделе `@throws` укажите все возможные исключения (или "Нет исключений").
   - В конце добавьте **секции**:
     - **Внешние зависимости:** список библиотек/классов, напрямую используемых классом (без родительских классов/интерфейсов).
     - **Взаимодействие:** описание, как класс взаимодействует с другими компонентами (например, "вызывает X.method() для Y").

2. **Документация методов:**
   - Описание должно отражать **логику** и **цель** метода.
   - Для каждого параметра в `@param` укажите тип, назначение и допустимые значения (если нужно).
   - В `@return` уточните возвращаемое значение (или "Нет возврата").
   - В `@throws` укажите исключения с условиями их возникновения.
   - В `@see` ссылайтесь на связанные методы или классы.
   - Добавьте описание **взаимодействий** с другими компонентами (например, "использует landingsBuilderInteractor.submit()").

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
 * @throws [исключения] (или "Нет исключений")
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

Для метода:
```
/**
 * [Описание метода]
 * 
 * @param [параметр] [описание]
 * @return [описание возврата]
 * @throws [исключения]
 * 
 * **Взаимодействие:**
 * - [описание взаимодействия с компонентами]
 * 
 * @see [связанные методы/классы]
 */
```"""

        prompt_template = f"""Вы - опытный разработчик, создающий документацию ИСКЛЮЧИТЕЛЬНО на РУССКОМ языке в формате KDoc.  
Ваша задача - добавить **полную и строгую документацию** к классам и методам, используя только следующие аннотации:  
`@property`, `@constructor`, `@param`, `@return`, `@throws`, `@see`.

{doc_rules}

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

    def _save_test_result(self, code: str, documentation: str, file_type: str) -> None:
        """Сохраняет результат для тестов."""
        try:
            # Создаем директорию для результатов в папке .cache
            results_dir = os.path.join(os.getcwd(), '.cache', 'test_results')
            os.makedirs(results_dir, exist_ok=True)

            # Генерируем имя файла на основе хеша кода
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]

            # Определяем расширение файла
            extension = 'kt' if file_type == 'kotlin' else file_type
            result_file = os.path.join(results_dir, f'doc_{code_hash}.{extension}')

            # Очищаем документацию от маркеров кода
            clean_doc = documentation.strip()

            # Убираем маркеры кода в начале и конце
            if '```' in clean_doc:
                parts = clean_doc.split('```')
                for part in parts:
                    if '/**' in part:
                        clean_doc = part.strip()
                        break

            # Убираем маркер языка, если он остался в начале
            lines = clean_doc.split('\n')
            if lines[0].lower() in ['java', 'kotlin']:
                clean_doc = '\n'.join(lines[1:])

            # Сохраняем только документацию
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(clean_doc)

            logging.info(f"\nРезультат сохранен в файл: {result_file}")

        except Exception as e:
            logging.error(f"Ошибка при сохранении результата: {str(e)}")
            raise

    def _create_empty_java_doc(self, code: str) -> str:
        """Создает пустую Java документацию для класса."""
        class_name = re.search(r'class\s+(\w+)', code)
        if not class_name:
            return "/** Документация отсутствует */"

        return f"""/**
 * Класс {class_name.group(1)}.
 */"""
