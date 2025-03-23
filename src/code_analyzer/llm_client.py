import json
import logging
import re
import time
from typing import Dict, List

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
                
            # Берем первую доступную модель
            model_info = available_models[0]
            self.model = model_info['name']
            
            # Получаем детальную информацию о модели
            model_details = ollama.show(self.model)
            
            # Форматируем размер модели
            size_bytes = int(model_info.get('size', 0))
            size_gb = size_bytes / (1024 * 1024 * 1024)
            
            # Получаем важные параметры модели
            model_params = model_details.get('model_info', {})
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
            
        except Exception as e:
            logging.error(f"Ошибка при проверке моделей: {str(e)}")
            raise RuntimeError(
                "Ошибка при получении списка моделей Ollama. Убедитесь, что:\n"
                "1. Сервис Ollama запущен\n"
                "2. Хотя бы одна модель активна"
            )

    def analyze_code(self, code: str, file_type: str) -> Dict:
        """
        Анализирует код и генерирует документацию.
        
        Args:
            code (str): Исходный код
            file_type (str): Тип файла ('kotlin' или 'java')
            
        Returns:
            Dict: Результат анализа
        """
        logging.info(f"Начинаем анализ файла типа {file_type}")
        logging.info(f"Размер кода: {len(code.encode())} байт")
        
        try:
            # Создаем промпт
            prompt = self._create_documentation_prompt(code, file_type)
            logging.info(f"Создан промпт размером {len(prompt.encode())} байт")
            
            # Логируем параметры запроса к модели
            model_params = {
                "temperature": 0.1,
                "top_p": 0.1,
                "num_predict": 16384,
                "stop": ["```"],
                "repeat_penalty": 1.1,
                "num_ctx": 16384
            }
            logging.info(f"Параметры запроса к модели: {json.dumps(model_params, indent=2)}")
            
            # Собираем ответ из чанков
            full_response = ""
            chunks_received = 0
            start_time = time.time()
            
            logging.info("Отправляем запрос к модели...")
            
            # Правильная работа с моделью
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options=model_params
            )
            
            for chunk in response:
                chunks_received += 1
                if 'response' in chunk:
                    chunk_text = chunk['response']
                    chunk_size = len(chunk_text.encode())
                    full_response += chunk_text
                    
                    # Подробное логирование чанков
                    logging.debug(f"Чанк #{chunks_received}: {chunk_size} байт")
                    if chunks_received % 5 == 0:
                        elapsed = time.time() - start_time
                        logging.info(
                            f"Прогресс: получено {chunks_received} чанков, "
                            f"текущий размер: {len(full_response.encode())} байт, "
                            f"время: {elapsed:.1f}с"
                        )
            
            elapsed_total = time.time() - start_time
            
            # Проверяем результат
            if not full_response:
                logging.error("Получен пустой ответ от модели")
                logging.error("Отправленный промпт:")
                logging.error(prompt)
                return self._create_error_response("Пустой ответ от модели")
            
            # Логируем успешный результат
            logging.info(
                f"Получен полный ответ от модели:\n"
                f"- Всего чанков: {chunks_received}\n"
                f"- Размер ответа: {len(full_response.encode())} байт\n"
                f"- Общее время: {elapsed_total:.1f}с\n"
                f"- Первые 100 символов ответа: {full_response[:100]}"
            )
            
            # Очищаем от маркеров кода
            documented_code = re.sub(r'^```\w*\n?', '', full_response)
            documented_code = re.sub(r'\n?```$', '', documented_code)
            
            # Проверяем наличие документации
            doc_patterns = [
                r'/\*\*[\s\S]*?\*/',  # Многострочные комментарии
                r'///.*$',            # Однострочные KDoc комментарии
                r'//.*@.*$'           # Строки с тегами документации
            ]
            
            has_documentation = False
            for pattern in doc_patterns:
                matches = re.finditer(pattern, documented_code, re.MULTILINE)
                doc_count = sum(1 for _ in matches)
                if doc_count > 0:
                    has_documentation = True
                    logging.info(f"Найдено {doc_count} блоков документации типа {pattern}")
            
            if not has_documentation:
                logging.error("В ответе отсутствует документация")
                # Логируем часть ответа для отладки
                logging.error("Первые 500 символов ответа:")
                logging.error(documented_code[:500])
                return self._create_error_response("В ответе отсутствует документация")
            
            # Проверяем сохранение оригинального кода
            original_code_lines = [
                line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('//')
            ]
            documented_code_lines = [
                line.strip() for line in documented_code.split('\n') 
                if line.strip() and not line.strip().startswith(('/', '*'))
            ]
            
            # Проверяем, что все строки оригинального кода присутствуют
            missing_lines = []
            for line in original_code_lines:
                found = False
                for doc_line in documented_code_lines:
                    if line in doc_line:
                        found = True
                        break
                if not found:
                    missing_lines.append(line)
            
            if missing_lines:
                logging.error(f"Модель пропустила {len(missing_lines)} строк кода")
                logging.error("Первые 3 пропущенные строки:")
                for line in missing_lines[:3]:
                    logging.error(f"- {line}")
                return self._create_error_response("Модель изменила оригинальный код")
            
            result_size = len(documented_code.encode())
            logging.info(
                f"Документация успешно сгенерирована:\n"
                f"- Размер результата: {result_size} байт\n"
                f"- Строк кода: {len(documented_code_lines)}\n"
                f"- Увеличение размера: {(result_size / len(code.encode()) * 100):.1f}%"
            )
            
            return {
                'code': documented_code,
                'model_info': self.model_details,
                'generation_info': {
                    'chunks_received': chunks_received,
                    'response_size': len(full_response),
                    'processing_time': elapsed_total,
                    'result_size': result_size
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Ошибка при генерации документации: {error_msg}")
            logging.error("Traceback:", exc_info=True)
            return self._create_error_response(f"Ошибка при генерации документации: {error_msg}")

    def _create_documentation_prompt(self, code: str, file_type: str) -> str:
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
   - Примеры использования для сложных методов

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
