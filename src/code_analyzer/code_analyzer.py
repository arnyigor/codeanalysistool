import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.llm.llm_client import OllamaClient


LOG_FILE = os.path.abspath("code_analyzer.log")

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

# Настройка логирования перед тестами
setup_logging()

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
        
        # Группируем файлы по директориям
        files_by_dirs = {}
        for file_path, result in self.analysis_results.items():
            abs_path = os.path.abspath(file_path)
            dir_path = os.path.dirname(abs_path)
            
            if dir_path not in files_by_dirs:
                files_by_dirs[dir_path] = []
            
            file_name = os.path.basename(file_path)
            status = result.get('status', 'error')
            status_icon = "✅" if status == 'success' else "❌"
            
            files_by_dirs[dir_path].append((file_name, status_icon, file_path, result))
        
        # Выводим файлы сгруппированные по директориям
        for dir_path, files in files_by_dirs.items():
            # Получаем относительный путь директории для отображения
            try:
                rel_dir = os.path.relpath(dir_path, os.getcwd())
            except ValueError:
                rel_dir = dir_path
            
            output_content += f"### Директория: {rel_dir}\n\n"
            
            for file_name, status_icon, full_path, _ in files:
                output_content += f"- {status_icon} {file_name}\n"
            
            output_content += "\n"
        
        output_content += "## Документация классов\n\n"
        
        # Добавляем документацию для каждого файла, сгруппированную по директориям
        for dir_path, files in files_by_dirs.items():
            try:
                rel_dir = os.path.relpath(dir_path, os.getcwd())
            except ValueError:
                rel_dir = dir_path
            
            output_content += f"### Директория: {rel_dir}\n\n"
            
            for file_name, _, full_path, result in files:
                if result.get('status') == 'success' and 'documentation' in result:
                    output_content += f"#### {file_name}\n\n"
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
    
    def _validate_documentation(self, documentation: str) -> str:
        """
        Валидирует документацию, проверяя отсутствие исходного кода.
        
        Args:
            documentation: Документация, полученная от модели
            
        Returns:
            Валидированная документация
        """
        if not documentation:
            logging.error("Получена пустая документация")
            return "/** Ошибка: документация отсутствует */"
            
        logging.info(f"Начало валидации документации, размер: {len(documentation)} символов")
        logging.debug(f"Исходная документация: \n{documentation[:200]}...")
        
        # Удаляем тройные обратные кавычки в начале и конце документации
        original_doc = documentation
        documentation = documentation.strip()
        
        # Проверяем наличие тройных кавычек в начале документации
        starts_with_backticks = documentation.startswith("```")
        if starts_with_backticks:
            logging.info("Документация начинается с тройных кавычек, удаляем")
            
            # Находим первую строку после кавычек
            first_newline = documentation.find("\n")
            if first_newline > 0:
                documentation = documentation[first_newline+1:]
                logging.info(f"Удалены начальные тройные кавычки, новый размер: {len(documentation)}")
        
        # Проверяем наличие тройных кавычек в конце документации
        ends_with_backticks = documentation.strip().endswith("```")
        if ends_with_backticks:
            logging.info("Документация заканчивается тройными кавычками, удаляем")
            
            # Находим последние кавычки
            last_backticks_pos = documentation.rfind("```")
            if last_backticks_pos > 0:
                # Находим начало строки с последними кавычками
                last_newline = documentation.rfind("\n", 0, last_backticks_pos)
                if last_newline > 0:
                    documentation = documentation[:last_newline].strip()
                    logging.info(f"Удалены конечные тройные кавычки, новый размер: {len(documentation)}")
        
        # Проверяем, содержатся ли тройные кавычки где-то еще в документации
        if "```" in documentation:
            logging.info("Обнаружены тройные кавычки внутри документации")
            
            # Удаляем все оставшиеся блоки с кавычками
            while "```" in documentation:
                start_pos = documentation.find("```")
                if start_pos == -1:
                    break
                
                # Ищем закрывающие кавычки
                end_pos = documentation.find("```", start_pos + 3)
                if end_pos == -1:
                    # Если нет закрывающих, удаляем только открывающие
                    documentation = documentation[:start_pos] + documentation[start_pos + 3:]
                    logging.info("Удалены незакрытые тройные кавычки")
                else:
                    # Удаляем кавычки, но сохраняем содержимое между ними
                    content_between = documentation[start_pos + 3:end_pos].strip()
                    documentation = documentation[:start_pos] + content_between + documentation[end_pos + 3:]
                    logging.info(f"Удален блок тройных кавычек, сохранено содержимое ({len(content_between)} символов)")
        
        # Проверяем наличие KDoc комментариев
        has_kdoc_start = "/**" in documentation
        has_kdoc_end = "*/" in documentation
        logging.info(f"Проверка KDoc маркеров: /** найден: {has_kdoc_start}, */ найден: {has_kdoc_end}")
        
        if not documentation.strip().startswith("/**"):
            logging.warning("Документация не начинается с /** - ищем начало KDoc")
            
            # Пытаемся найти начало KDoc
            kdoc_start = documentation.find("/**")
            if kdoc_start >= 0:
                # Удаляем всё до начала KDoc
                pre_content = documentation[:kdoc_start].strip()
                if pre_content:
                    logging.info(f"Текст перед /** (будет удален): '{pre_content[:50]}...'")
                documentation = documentation[kdoc_start:]
                logging.info(f"Начало /** найдено на позиции {kdoc_start}, новый размер: {len(documentation)}")
            else:
                # Если нет KDoc, оборачиваем всю документацию в KDoc
                logging.error("Не найден открывающий KDoc маркер /**")
                
                # Проверяем, нет ли в документации кода
                looks_like_code = any(keyword in documentation for keyword in ["class ", "interface ", "fun ", "val ", "var ", "package "])
                if looks_like_code:
                    logging.error(f"Документация похожа на код, возможно модель сгенерировала код вместо документации. Примерно: '{documentation[:100]}...'")
                    return "/** Ошибка: вместо документации сгенерирован код */"
                
                # Оборачиваем в KDoc
                cleaned_doc = documentation.strip()
                documentation = f"/**\n * {cleaned_doc.replace('\n', '\n * ')}\n */"
                logging.info(f"Документация обернута в KDoc формат, новый размер: {len(documentation)}")
        
        # Проверяем, что документация заканчивается закрывающим комментарием
        if not documentation.strip().endswith("*/"):
            logging.warning("Документация не заканчивается на */ - ищем конец KDoc")
            
            # Пытаемся найти конец KDoc
            kdoc_end = documentation.rfind("*/")
            if kdoc_end >= 0:
                # Проверяем, есть ли код после закрывающего KDoc
                post_content = documentation[kdoc_end+2:].strip()
                if post_content:
                    logging.info(f"Текст после */ (будет удален): '{post_content[:50]}...'")
                
                # Удаляем всё после конца KDoc
                documentation = documentation[:kdoc_end+2]
                logging.info(f"Конец */ найден на позиции {kdoc_end}, новый размер: {len(documentation)}")
            else:
                # Добавляем закрывающий тег, если его нет
                logging.error("Не найден закрывающий KDoc маркер */")
                documentation = documentation.strip() + "\n */"
                logging.info(f"Добавлен закрывающий маркер */, новый размер: {len(documentation)}")
        
        # Удаляем строки с "[Краткое описание]"
        lines = documentation.split('\n')
        filtered_lines = []
        removed_count = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Проверяем различные варианты [Краткое описание]
            if ("[краткое описание" in line_lower or 
                "* [краткое описание" in line_lower or 
                "[brief description" in line_lower or
                "* [brief description" in line_lower):
                removed_count += 1
                logging.info(f"Удалена строка #{i+1}: '{line.strip()}'")
                continue
            filtered_lines.append(line)
        
        if removed_count > 0:
            documentation = '\n'.join(filtered_lines)
            logging.info(f"Удалено {removed_count} строк с '[Краткое описание]', новый размер: {len(documentation)}")
        
        # Финальная проверка
        if not (documentation.strip().startswith("/**") and documentation.strip().endswith("*/")):
            logging.error(f"После валидации документация не соответствует KDoc формату: {documentation[:50]}...{documentation[-50:]}")
            if not has_kdoc_start and not has_kdoc_end:
                # Если мы не нашли ни начало, ни конец KDoc в оригинальной документации
                logging.error("В документации полностью отсутствуют KDoc маркеры")
                # Пытаемся извлечь текст документации
                cleaned_text = original_doc.replace("```", "").strip()
                if cleaned_text:
                    # Если есть полезный текст после удаления кавычек, оборачиваем его в KDoc
                    logging.info(f"Пытаемся восстановить документацию из текста: '{cleaned_text[:100]}...'")
                    return f"/**\n * {cleaned_text.replace('\n', '\n * ')}\n */"
                else:
                    return "/** Ошибка: не найден KDoc комментарий */"
        
        logging.info(f"Валидация документации завершена успешно, финальный размер: {len(documentation)}")
        return documentation
    
    def _extract_package_and_imports(self, file_content: str) -> tuple:
        """
        Извлекает информацию о пакете и импортах из Kotlin/Java файла.
        
        Args:
            file_content: Содержимое исходного файла
            
        Returns:
            Кортеж (package_statement, imports, class_code)
        """
        lines = file_content.split('\n')
        package_statement = ""
        imports = []
        code_start_index = 0
        
        # Находим объявление пакета
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("package "):
                package_statement = line
                code_start_index = i + 1
                break
        
        # Находим импорты
        for i in range(code_start_index, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("import "):
                imports.append(lines[i])
                code_start_index = i + 1
            elif stripped and not stripped.startswith("//"):
                # Если строка не пустая и не комментарий, значит импорты закончились
                break
        
        # Пропускаем пустые строки и комментарии после импортов
        while code_start_index < len(lines) and (not lines[code_start_index].strip() or lines[code_start_index].strip().startswith("//")):
            code_start_index += 1
        
        # Получаем код класса без пакета и импортов
        class_code = "\n".join(lines[code_start_index:])
        
        return package_statement, imports, class_code
    
    def _remove_existing_kdoc(self, class_code: str) -> str:
        """
        Удаляет существующую KDoc документацию из кода класса.
        
        Args:
            class_code: Код класса
            
        Returns:
            Код класса без KDoc документации
        """
        # Проверяем наличие KDoc комментария в начале
        if not class_code.strip().startswith("/**"):
            return class_code
        
        # Находим конец KDoc комментария
        kdoc_end = class_code.find("*/")
        if kdoc_end < 0:
            return class_code
        
        # Удаляем KDoc и пробельные символы после него
        code_without_kdoc = class_code[kdoc_end + 2:]
        return code_without_kdoc.lstrip()
    
    def _save_result(self, file_path: str, result: Dict, output_dir: str) -> None:
        """
        Сохраняет результат анализа в файл, сохраняя структуру папок.
        
        Логика обработки путей:
        1. На вход подается путь до файла или папки (базовая папка для документации)
        2. Скрипт запускается из директории (базовая папка скрипта)
        3. Итоговый путь = output_dir + относительный_путь_от_базового_пути
        
        Args:
            file_path: Путь к исходному файлу
            result: Результат анализа
            output_dir: Директория для сохранения
        """
        try:
            # Получаем абсолютные пути
            abs_file_path = os.path.abspath(file_path)
            abs_output_dir = os.path.abspath(output_dir)
            
            logging.info(f"Сохранение результата анализа. Исходный файл: {abs_file_path}")
            logging.info(f"Директория для сохранения: {abs_output_dir}")
            
            # Получаем исходную директорию и имя файла
            base_input_dir = os.path.dirname(abs_file_path)
            file_name = os.path.basename(abs_file_path)
            
            logging.info(f"Базовая входная директория: {base_input_dir}")
            logging.info(f"Имя файла: {file_name}")
            
            # Простой способ сохранения структуры папок:
            # Просто сохраняем весь путь после буквы диска
            drive, path_part = os.path.splitdrive(abs_file_path)
            # Убираем начальный слеш, чтобы не создавать корневую директорию
            rel_path = path_part.lstrip(os.path.sep)
            rel_dir = os.path.dirname(rel_path)
            
            logging.info(f"Относительный путь (без диска): {rel_path}")
            logging.info(f"Относительная директория: {rel_dir}")
            
            # Создаем структуру папок в выходной директории
            output_sub_dir = os.path.join(abs_output_dir, rel_dir)
            os.makedirs(output_sub_dir, exist_ok=True)
            
            # Формируем полный путь к выходному файлу
            output_file = os.path.join(output_sub_dir, file_name)
            logging.info(f"Итоговый путь для сохранения: {output_file}")
            
            # Читаем исходный файл
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                logging.debug(f"Прочитан исходный файл, размер: {len(original_content)} байт")
            
            # Извлекаем компоненты
            package_statement, imports, class_code = self._extract_package_and_imports(original_content)
            logging.debug(f"Извлечены секции - пакет: {bool(package_statement)}, импортов: {len(imports)}")
            
            # Удаляем существующую документацию
            clean_class_code = self._remove_existing_kdoc(class_code)
            
            # Формируем новый контент
            new_content = []
            
            # Добавляем пакет
            if package_statement:
                new_content.append(package_statement)
                new_content.append("")
                logging.debug("Добавлен пакет в выходной файл")
            
            # Добавляем импорты
            if imports:
                new_content.extend(imports)
                new_content.append("")
                logging.debug(f"Добавлены импорты: {len(imports)} строк")
            
            # Добавляем документацию
            if 'documentation' in result and result['documentation']:
                # Валидируем документацию с помощью метода _validate_documentation
                doc = self._validate_documentation(result['documentation'])
                
                # Добавляем документацию в выходной файл
                new_content.append(doc)
                new_content.append("")
                logging.info(f"Добавлена документация, размер: {len(doc)} символов")
            else:
                logging.warning("Документация отсутствует в результате")
            
            # Добавляем код
            new_content.append(clean_class_code.strip())
            logging.debug(f"Добавлен код класса, размер: {len(clean_class_code)} символов")
            
            # Сохраняем файл
            content_to_save = '\n'.join(new_content)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content_to_save)
            
            logging.info(f"Результат успешно сохранен в файл: {output_file}")
            logging.info(f"Размер сохраненного файла: {len(content_to_save)} символов")
            
        except Exception as e:
            logging.error(f"Ошибка при сохранении файла {file_path}: {str(e)}", exc_info=True)
    
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