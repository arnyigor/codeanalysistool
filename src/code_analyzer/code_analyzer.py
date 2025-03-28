import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.llm.llm_client import OllamaClient


class CodeAnalyzer:
    """Анализатор кода для документирования файлов и проектов."""
    
    def __init__(self, model_client: Optional[OllamaClient] = None, cache_dir: str = '.cache'):
        """
        Инициализирует анализатор кода.
        
        Args:
            model_client: Клиент для взаимодействия с языковой моделью.
                          Если не указан, будет создан новый экземпляр.
            cache_dir: Директория для кэширования результатов анализа.
        """
        self.client = model_client or OllamaClient()
        self.supported_extensions = {
            '.kt': 'kotlin',
            '.java': 'java',
            # Дополнительные типы файлов могут быть добавлены здесь
        }
        self.cache_dir = cache_dir
        self.analysis_results = {}
        
        # Создаем директории для кэша если их нет
        os.makedirs(os.path.join(cache_dir, 'docs'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'test_results'), exist_ok=True)
        
        logging.info("Инициализирован анализатор кода")
    
    def analyze_path(self, path: str, recursive: bool = True, 
                     output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """
        Анализирует все поддерживаемые файлы по указанному пути.
        
        Args:
            path: Путь к файлу или директории для анализа
            recursive: Анализировать ли вложенные директории
            output_dir: Директория для сохранения результатов.
                        Если не указана, результаты не сохраняются.
        
        Returns:
            Словарь с результатами анализа, где ключи - пути к файлам
        """
        path_obj = Path(path)
        results = {}
        
        if path_obj.is_file():
            # Обрабатываем один файл
            result = self.analyze_file(str(path_obj))
            if result:
                results[str(path_obj)] = result
                if output_dir:
                    self._save_result(str(path_obj), result, output_dir)
        
        elif path_obj.is_dir():
            # Обрабатываем директорию
            for file_path in self._find_files(path_obj, recursive):
                logging.info(f"Анализ файла: {file_path}")
                result = self.analyze_file(str(file_path))
                if result:
                    results[str(file_path)] = result
                    if output_dir:
                        self._save_result(str(file_path), result, output_dir)
        else:
            logging.error(f"Путь не существует: {path}")
        
        # Сохраняем результаты для метода generate_documentation
        self.analysis_results.update(results)
        
        return results
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Анализирует директорию и подготавливает результаты для документации.
        Метод используется в main.py.
        
        Args:
            directory_path: Путь к директории с исходным кодом
            
        Returns:
            Словарь с результатами анализа
        """
        logging.info(f"Начинаем анализ директории: {directory_path}")
        
        # Используем метод analyze_path для анализа директории
        results = self.analyze_path(directory_path, recursive=True)
        
        # Выводим статистику
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        error_count = len(results) - success_count
        
        logging.info(f"Анализ директории завершен: {directory_path}")
        logging.info(f"Всего файлов: {len(results)}")
        logging.info(f"Успешно: {success_count}")
        logging.info(f"С ошибками: {error_count}")
        
        # Возвращаем результаты
        return results
    
    def generate_documentation(self, output_file: str) -> None:
        """
        Генерирует документацию на основе результатов анализа.
        Метод используется в main.py.
        
        Args:
            output_file: Путь к файлу для сохранения документации
        """
        if not self.analysis_results:
            logging.warning("Нет результатов анализа для генерации документации")
            return
        
        logging.info(f"Генерация документации в файл: {output_file}")
        
        # Создаем директорию для выходного файла, если её нет
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Формируем содержимое документации
        output_content = "# Документация проекта\n\n"
        
        # Добавляем информацию о проанализированных файлах
        output_content += "## Проанализированные файлы\n\n"
        
        for file_path, result in self.analysis_results.items():
            file_name = os.path.basename(file_path)
            status = result.get('status', 'error')
            status_icon = "✅" if status == 'success' else "❌"
            
            output_content += f"- {status_icon} {file_name}\n"
        
        output_content += "\n## Документация классов\n\n"
        
        # Добавляем документацию для каждого файла
        for file_path, result in self.analysis_results.items():
            if result.get('status') == 'success' and 'documentation' in result:
                file_name = os.path.basename(file_path)
                
                output_content += f"### {file_name}\n\n"
                output_content += "```kotlin\n"
                output_content += result['documentation']
                output_content += "\n```\n\n"
        
        # Сохраняем документацию в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        logging.info(f"Документация успешно сгенерирована: {output_file}")
    
    def analyze_file(self, file_path: str, context_files: List[str] = None) -> Optional[Dict]:
        """
        Анализирует отдельный файл и генерирует документацию.
        
        Args:
            file_path: Путь к файлу для анализа
            context_files: Список путей к файлам, содержащим контекстную информацию
        
        Returns:
            Словарь с результатами анализа или None, если файл не поддерживается
        """
        path_obj = Path(file_path)
        
        # Проверяем поддержку типа файла
        file_type = self._get_file_type(path_obj)
        if not file_type:
            logging.warning(f"Неподдерживаемый тип файла: {file_path}")
            return None
        
        # Проверяем кэш
        cache_path = self._get_cache_path(file_path, file_type)
        cached_result = self._load_from_cache(cache_path)
        
        if cached_result:
            logging.info(f"Результат загружен из кэша: {file_path}")
            return cached_result
        
        try:
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Если код пустой, пропускаем файл
            if not code.strip():
                logging.warning(f"Пустой файл: {file_path}")
                return None
            
            # Получаем контекст, если указаны контекстные файлы
            context = self._get_context(context_files) if context_files else None
            
            # Анализируем код
            result = self.client.analyze_code(code, file_type, context)
            
            # Обогащаем результат метаданными
            if result and 'documentation' in result:
                result['file_path'] = file_path
                result['file_type'] = file_type
                result['file_name'] = path_obj.name
            
            # Сохраняем результат в кэш
            if result and 'documentation' in result:
                self._save_to_cache(cache_path, result)
            
            return result
        
        except Exception as e:
            logging.error(f"Ошибка при анализе файла {file_path}: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "file_path": file_path,
                "file_type": file_type,
                "file_name": path_obj.name,
                "status": "error"
            }
    
    def _get_file_type(self, path: Path) -> Optional[str]:
        """
        Определяет тип файла по его расширению.
        
        Args:
            path: Путь к файлу
        
        Returns:
            Строковое представление типа файла или None, если тип не поддерживается
        """
        extension = path.suffix.lower()
        return self.supported_extensions.get(extension)
    
    def _find_files(self, path: Path, recursive: bool) -> List[Path]:
        """
        Находит все поддерживаемые файлы в указанной директории.
        
        Args:
            path: Путь к директории
            recursive: Искать ли во вложенных директориях
        
        Returns:
            Список путей к найденным файлам
        """
        result = []
        
        if recursive:
            # Рекурсивный поиск
            for ext in self.supported_extensions:
                result.extend(path.glob(f"**/*{ext}"))
        else:
            # Только в текущей директории
            for ext in self.supported_extensions:
                result.extend(path.glob(f"*{ext}"))
        
        return result
    
    def _get_context(self, context_files: List[str]) -> Dict[str, str]:
        """
        Читает контекстные файлы и формирует словарь контекста.
        
        Args:
            context_files: Список путей к контекстным файлам
        
        Returns:
            Словарь контекста, где ключи - имена файлов, а значения - их содержимое
        """
        context = {}
        
        for file_path in context_files:
            try:
                path_obj = Path(file_path)
                if path_obj.exists() and path_obj.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        context[path_obj.name] = f.read()
                else:
                    logging.warning(f"Контекстный файл не найден: {file_path}")
            except Exception as e:
                logging.error(f"Ошибка при чтении контекстного файла {file_path}: {str(e)}")
        
        return context
    
    def _save_result(self, file_path: str, result: Dict, output_dir: str) -> None:
        """
        Сохраняет результат анализа в файл.
        
        Args:
            file_path: Путь к исходному файлу
            result: Результат анализа
            output_dir: Директория для сохранения
        """
        try:
            # Создаем директорию для результатов
            os.makedirs(output_dir, exist_ok=True)
            
            # Формируем имя выходного файла
            path_obj = Path(file_path)
            output_file = Path(output_dir) / f"{path_obj.stem}_doc{path_obj.suffix}"
            
            # Сохраняем документацию
            with open(output_file, 'w', encoding='utf-8') as f:
                if 'documentation' in result:
                    f.write(result['documentation'])
                else:
                    f.write(f"// Ошибка генерации документации: {result.get('error', 'Неизвестная ошибка')}")
            
            logging.info(f"Результат сохранен в файл: {output_file}")
        
        except Exception as e:
            logging.error(f"Ошибка при сохранении результата для {file_path}: {str(e)}")
    
    def _get_cache_path(self, file_path: str, file_type: str) -> str:
        """
        Генерирует путь к кэш-файлу на основе хэша пути к файлу.
        
        Args:
            file_path: Путь к исходному файлу
            file_type: Тип файла
            
        Returns:
            Путь к файлу кэша
        """
        import hashlib
        
        # Создаем хэш пути к файлу
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        
        # Возвращаем путь к файлу кэша
        return os.path.join(self.cache_dir, 'docs', f"{file_hash}_{file_type}.json")
    
    def _save_to_cache(self, cache_path: str, result: dict) -> None:
        """
        Сохраняет результат в кэш.
        
        Args:
            cache_path: Путь к файлу кэша
            result: Результат анализа
        """
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"Результат сохранен в кэш: {cache_path}")
        except Exception as e:
            logging.warning(f"Не удалось сохранить кэш: {str(e)}")
    
    def _load_from_cache(self, cache_path: str) -> Optional[dict]:
        """
        Загружает результат из кэша.
        
        Args:
            cache_path: Путь к файлу кэша
            
        Returns:
            Результат анализа или None, если кэш не найден
        """
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    logging.info(f"Результат загружен из кэша: {cache_path}")
                    return result
        except Exception as e:
            logging.warning(f"Не удалось загрузить кэш: {str(e)}")
        
        return None 