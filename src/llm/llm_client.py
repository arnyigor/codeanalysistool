import json
import logging
import os
import time
from typing import Dict, Optional

import ollama


class OllamaClient:
    def __init__(self):
        """
        Инициализация клиента Ollama.
        Использует первую доступную запущенную модель.
        """
        # Укажите явный путь к log файлу
        # log_file_path = os.path.abspath("ollama.log")
        # logging.basicConfig(
        #     filename=log_file_path,
        #     level=logging.DEBUG,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     encoding='utf-8'
        # )
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

            # Получаем важные параметры модели
            model_params = model_details.modelinfo
            context_length = model_params.get('llama.context_length', 'Неизвестно')
            param_count = model_params.get('general.parameter_count', 0)
            param_count_b = param_count / 1_000_000_000 if param_count else 'Неизвестно'

            # Логируем расширенную информацию о модели
            logging.info(
                f"Используется модель Ollama:\n"
                f"- Название: {self.model}\n"
                f"- Размер модели: {size_gb:.2f} GB\n"
                f"- Количество параметров: {param_count_b:.1f}B\n"
                f"- Размер контекста: {context_length} токенов\n"
                f"- Архитектура: {model_params.get('general.architecture', 'Неизвестно')}\n"
                f"- Квантизация: {model_params.get('general.quantization', 'Неизвестно')}"
            )

            # Сохраняем детали модели
            self.model_details = model_details
        except KeyError as e:
            logging.error(f"Отсутствует обязательное поле в данных модели: {str(e)}")
        except Exception as e:
            logging.error(f"Ошибка при проверке моделей: {str(e)}")
            raise RuntimeError(
                "Ошибка при получении списка моделей Ollama. Убедитесь, что:\n"
                "1. Сервис Ollama запущен\n"
                "2. Хотя бы одна модель активна"
            )

    def analyze_code(self, code: str, file_type: str) -> Optional[dict]:
        """
        Анализирует код и генерирует документацию.

        Args:
            code (str): Исходный код
            file_type (str): Тип файла ('kotlin', 'java', 'python')

        Returns:
            Dict: Результат анализа (документация или ошибка)
        """
        logging.info(f"Начинаем анализ файла типа {file_type}")
        logging.info(f"Размер кода: {len(code.encode())} байт")

        try:
            # Проверка на пустой код
            if not code.strip():
                return self._create_error_response("Пустой исходный код")

            # Создаем промпт
            prompt = self._create_documentation_prompt(code, file_type)
            logging.info(f"Создан промпт размером {len(prompt.encode())} байт")

            # Параметры генерации
            model_params = {
                "temperature": 0.0,
                "top_p": 0.1,
                "max_tokens": 8192,
                "stop": ["</code>"],
                "repeat_penalty": 1.2,
                "context_length": 32768
            }

            # Логирование параметров
            logging.info(f"Параметры запроса к модели: {json.dumps(model_params, indent=2)}")

            # Генерация ответа
            start_time = time.time()  # Начало таймера
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options=model_params
            )

            full_response = []
            chunks_received = 0
            total_tokens = 0
            logging.info("Отправляем запрос к модели...")

            try:
                for chunk in response:
                    chunks_received += 1
                    chunk_text = getattr(chunk, 'response', '') or getattr(chunk, 'text', '')

                    if chunk_text:
                        full_response.append(chunk_text)
                        tokens = len(chunk_text.split())  # Считаем токены в чанке
                        total_tokens += tokens
                        # logging.debug(f"Чанк #{chunks_received}: {chunk_text[:100]}... ({tokens} токенов)")

                    # Проверка завершения потока
                    if getattr(chunk, 'done', False):
                        done_reason = getattr(chunk, 'done_reason', 'N/A')
                        logging.info(f"Поток завершен: {done_reason}")
                        break

                # Объединение ответа
                full_response_str = ''.join(full_response)
                elapsed_time = time.time() - start_time  # Время выполнения

                # Извлекаем метрики из последнего чанка (если доступно)
                if chunks_received > 0:
                    last_chunk = chunk  # Предполагаем, что последний чанк содержит метаданные
                    time_taken = getattr(last_chunk, 'time_taken', elapsed_time)
                    tokens_per_second = getattr(last_chunk, 'tokens_per_second', total_tokens / elapsed_time)
                else:
                    time_taken = elapsed_time
                    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

            except Exception as e:
                logging.error(f"Ошибка при получении ответа: {str(e)}", exc_info=True)
                return self._create_error_response(f"Ошибка генерации: {str(e)}")

            # Проверка на пустой ответ
            if not full_response_str.strip():
                logging.error("Пустой ответ от модели")
                return self._create_error_response("Пустой ответ от модели")

            # Логирование метрик
            logging.info(
                f"Готово:\n"
                f"- Время выполнения: {elapsed_time:.1f} секунд\n"
                f"- Объем контекста: {model_params['context_length']} токенов\n"
                f"- Обработано токенов: {total_tokens}\n"
                f"- Скорость генерации: {tokens_per_second:.1f} ток/с\n"
                f"- Чанков: {chunks_received}\n"
                f"- Размер ответа: {len(full_response_str.encode())} байт\n"
                f"- Первые 100 символов: {full_response_str[:100]}"
            )

            return {
                "documentation": full_response_str,
                "status": "success",
                "metrics": {
                    "time": elapsed_time,
                    "tokens": total_tokens,
                    "speed": tokens_per_second
                }
            }

        except Exception as e:
            error_msg = f"Неизвестная ошибка: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)

    def _create_documentation_prompt(self, code: str, file_type: str) -> str:
        # Проверка поддерживаемых типов
        if file_type not in ['java', 'kotlin']:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

        """Создает промпт для генерации документации."""
        doc_format = "KDoc" if file_type == "kotlin" else "JavaDoc"

        prompt_template = f"""[СИСТЕМНЫЕ ТРЕБОВАНИЯ]
Вы - опытный разработчик, создающий документацию в формате {doc_format} на русском языке.

[ПРАВИЛА ДОКУМЕНТИРОВАНИЯ]
1. Создать подробную документацию для каждого класса:
   - Описание назначения класса
   - Основные возможности
   - Примеры использования
   - Зависимости и требования

2. Документировать все методы:
   - Подробное описание функциональности
   - Все параметры с типами и описанием
   - Возвращаемые значения
   - Исключения и условия их возникновения

3. Документировать важные поля:
   - Назначение поля
   - Допустимые значения
   - Влияние на работу класса

4. Особые требования:
   - Вся документация на русском языке
   - Использовать стандартные теги {doc_format}
   - НЕ ИЗМЕНЯТЬ оригинальный код
   - Добавлять ТОЛЬКО документацию
   - Документация должна быть в формате /** ... */
   - Каждый документируемый элемент должен иметь описание

[ФОРМАТ ОТВЕТА]
1. Верните ТОЛЬКО код с добавленной документацией
2. НЕ добавляйте никаких пояснений до или после кода
3. НЕ изменяйте оригинальный код
4. Добавляйте документацию ТОЛЬКО в формате /** ... */

[КОД ДЛЯ ДОКУМЕНТИРОВАНИЯ]
```{file_type}
{code}
```"""
        return prompt_template

    def _create_error_response(self, error_message: str) -> Dict:
        """Создает структурированный ответ с ошибкой."""
        return {
            "error": error_message,
            "code": ""
        }
