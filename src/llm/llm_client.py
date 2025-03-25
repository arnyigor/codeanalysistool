import json
import logging
import time
from typing import Dict, Optional

import ollama


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
            
            # Логируем структуру для отладки
            # logging.debug("Структура model_details:")
            # logging.debug(json.dumps(model_details, indent=2, ensure_ascii=False))

            # Форматируем размер модели
            size_bytes = int(model_info.get('size', 0))
            size_gb = size_bytes / (1024 * 1024 * 1024)

            # Получаем параметры модели из структуры model_info
            model_params = model_details.get('model_info', {})
            
            # Получаем параметры
            param_count = model_params.get('general.parameter_count', 0)
            param_count_b = round(param_count / 1_000_000_000, 2) if param_count else 'Неизвестно'
            
            context_length = model_params.get('qwen2.context_length', 
                           model_params.get('general.context_length', 'Неизвестно'))
            
            architecture = model_params.get('general.architecture', 'Неизвестно')
            quantization = model_params.get('general.quantization_level', 'Неизвестно')

            # Логируем информацию о модели
            logging.info("Используется модель Ollama:")
            logging.info(f"- Название: {self.model}")
            logging.info(f"- Размер модели: {size_gb:.2f} GB")
            logging.info(f"- Количество параметров: {param_count_b}B")
            logging.info(f"- Размер контекста: {context_length} токенов")
            logging.info(f"- Архитектура: {architecture}")
            logging.info(f"- Квантизация: {quantization}")

            # Сохраняем структуру
            self.model_details = model_details

        except KeyError as e:
            logging.error("Отсутствует обязательное поле в данных модели: {}".format(str(e)), stack_info=True)
        except Exception as e:
            logging.error("Ошибка при проверке моделей: {}".format(str(e)), stack_info=True)
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

        Raises:
            ValueError: Если тип файла не поддерживается
        """
        logging.info(f"Начинаем анализ файла типа {file_type}")
        logging.info(f"Размер кода: {len(code.encode())} байт")

        try:
            # Проверка на пустой код
            if not code.strip():
                return self._create_error_response("пустой код")

            # Создаем промпт
            prompt = self._create_documentation_prompt(code, file_type)
            prompt_size = len(prompt.encode())
            logging.info(f"Создан промпт размером {prompt_size} байт")

            # Проверка размера контекста
            max_context_size = 16384  # Лимит 16KB как в тесте
            if prompt_size > max_context_size:
                logging.error(f"Превышен максимальный размер контекста: {prompt_size} байт > {max_context_size} байт")
                return {
                    "error": "превышен лимит контекста",
                    "code": code
                }

            # Параметры генерации
            model_params = {
                "temperature": 0.1,
                "top_p": 0.1,
                "num_predict": 2048,
                "stop": ["[КОНЕЦ ОТВЕТА]"],
                "repeat_penalty": 1.1,
                "num_ctx": 8192,
                "system": "Вы - опытный разработчик, создающий документацию на русском языке."
            }

            # Логирование параметров
            logging.info(f"Параметры запроса к модели: {json.dumps(model_params, indent=2, ensure_ascii=False)}")

            # Генерация ответа
            start_time = time.time()
            
            logging.info("Отправляем запрос к модели...")
            
            try:
                # Получаем ответ от модели
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=True,
                    options=model_params
                )

                full_response = []
                chunks_received = 0
                total_tokens = 0
                generation_info = {}
                
                # Универсальная обработка ответа
                for chunk in response:
                    chunks_received += 1
                    
                    # Подробное логирование первого чанка
                    if chunks_received == 1:
                        logging.info(f"Первый чанк: {chunk}")
                        if isinstance(chunk, dict):
                            logging.info(f"Ключи в первом чанке: {chunk.keys()}")
                            # Сохраняем информацию о генерации
                            generation_info = {
                                'model': chunk.get('model', ''),
                                'created_at': chunk.get('created_at', ''),
                                'chunks_received': chunks_received
                            }
                    
                    chunk_text = None
                    
                    # Пробуем разные варианты получения текста
                    if isinstance(chunk, dict):
                        # Проверяем все возможные ключи
                        for key in ['response', 'content', 'text', 'output', 'generated_text']:
                            if key in chunk:
                                chunk_text = chunk[key]
                                break
                        # Проверяем на ошибки
                        if 'error' in chunk:
                            error_msg = f"Ошибка от модели: {chunk['error']}"
                            logging.error(error_msg)
                            return self._create_error_response(error_msg)
                    elif hasattr(chunk, 'response'):
                        chunk_text = chunk.response
                    elif hasattr(chunk, 'text'):
                        chunk_text = chunk.text
                    elif isinstance(chunk, str):
                        chunk_text = chunk
                        
                    if chunk_text:
                        full_response.append(chunk_text)

                # Объединяем ответ
                full_response_str = ''.join(full_response).strip()
                elapsed_time = time.time() - start_time

                # Подробное логирование при пустом ответе
                if not full_response_str:
                    logging.error("Пустой ответ от модели")
                    logging.error("Отправленный промпт:")
                    logging.error(prompt[:500] + "...")
                    logging.error("Получено чанков: {}".format(chunks_received))
                    # Проверяем последний чанк
                    if chunks_received > 0:
                        logging.error("Последний полученный чанк:")
                        logging.error(str(chunk))
                        if 'done_reason' in chunk:
                            logging.error(f"Причина завершения: {chunk['done_reason']}")
                    return self._create_error_response("Пустой ответ от модели")

                # Логируем полный ответ для отладки
                # logging.debug("Полный ответ от модели:")
                # logging.debug(full_response_str)

                # Очищаем ответ от возможных маркеров
                clean_response = full_response_str
                if "[КОНЕЦ КОДА]" in clean_response:
                    clean_response = clean_response.split("[КОНЕЦ КОДА]")[0]
                if "[КОД ДЛЯ ДОКУМЕНТИРОВАНИЯ]" in clean_response:
                    clean_response = clean_response.split("[КОД ДЛЯ ДОКУМЕНТИРОВАНИЯ]")[1]
                
                clean_response = clean_response.strip()

                # Проверяем наличие документации
                if "/**" not in clean_response:
                    logging.error("Ответ не содержит документацию")
                    logging.error("Полученный ответ:")
                    logging.error(clean_response[:500] + "...")
                    return self._create_error_response("Ответ не содержит документацию")

                # Подсчитываем токены если не получили из модели
                if total_tokens == 0:
                    total_tokens = len(full_response_str.split())

                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

                # Логирование метрик
                metrics_str = (
                    "\nГотово:"
                    f"\n- Время выполнения: {elapsed_time:.1f} секунд"
                    f"\n- Объем контекста: {model_params['num_ctx']} токенов"
                    f"\n- Обработано токенов: {total_tokens}"
                    f"\n- Скорость генерации: {tokens_per_second:.1f} ток/с"
                    f"\n- Чанков получено: {chunks_received}"
                    f"\n- Размер ответа: {len(clean_response.encode('utf-8'))} байт"
                )
                logging.info(metrics_str)
                
                if clean_response:
                    preview = clean_response[:100] + "..." if len(clean_response) > 100 else clean_response
                    logging.info(f"Начало ответа:\n{preview}")

                # Обновляем информацию о генерации
                generation_info['chunks_received'] = chunks_received
                generation_info['total_tokens'] = total_tokens
                generation_info['elapsed_time'] = elapsed_time

                # Удаляем маркеры кода из ответа
                if clean_response.startswith("```"):
                    clean_response = clean_response[clean_response.find("\n")+1:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]

                return {
                    "documentation": clean_response,
                    "code": code + "\n" + clean_response,  # Добавляем оригинальный код
                    "status": "success",
                    "metrics": {
                        "time": elapsed_time,
                        "tokens": total_tokens,
                        "speed": tokens_per_second,
                        "chunks": chunks_received
                    },
                    "generation_info": generation_info
                }

            except Exception as e:
                error_msg = "Ошибка при генерации: {}".format(str(e))
                logging.error(error_msg, exc_info=True)
                return self._create_error_response(error_msg)

        except ValueError as e:
            # Пробрасываем ValueError наверх
            raise
        except Exception as e:
            error_msg = f"Неизвестная ошибка: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)

    def _create_documentation_prompt(self, code: str, file_type: str) -> str:
        """Создает промпт для генерации документации."""
        # Проверка поддерживаемых типов
        if file_type not in ['java', 'kotlin']:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

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
{code}

[КОНЕЦ КОДА]
Пожалуйста, верните документированный код без дополнительных пояснений."""
        return prompt_template

    def _create_error_response(self, error_message: str) -> Dict:
        """Создает структурированный ответ с ошибкой."""
        return {
            "error": error_message,
            "code": ""
        }
