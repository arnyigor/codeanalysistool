import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from src.code_analyzer.code_analyzer import CodeAnalyzer
from src.llm.llm_client import OllamaClient


def setup_logging(log_level: str = "INFO") -> None:
    """
    Настраивает логирование для приложения.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Создаем директорию для логов, если её нет
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Формируем имя файла лога с текущей датой и временем
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"code_analyzer_{timestamp}.log"
    
    # Настраиваем логирование
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Логирование настроено. Файл лога: {log_file}")


def parse_args() -> argparse.Namespace:
    """
    Обрабатывает аргументы командной строки.
    
    Returns:
        Объект с параметрами командной строки
    """
    parser = argparse.ArgumentParser(
        description="Инструмент для анализа кода и генерации документации",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'path',
        help="Путь к файлу или директории для анализа"
    )
    
    parser.add_argument(
        '-o', '--output',
        help="Директория для сохранения результатов",
        default="output"
    )
    
    parser.add_argument(
        '-r', '--recursive',
        help="Рекурсивно обрабатывать вложенные директории",
        action='store_true',
        default=True
    )
    
    parser.add_argument(
        '-c', '--context',
        help="Пути к файлам контекста (через запятую)",
        default=""
    )
    
    parser.add_argument(
        '--log-level',
        help="Уровень логирования",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    
    return parser.parse_args()


def process_context_files(context_arg: str) -> List[str]:
    """
    Обрабатывает аргумент контекстных файлов.
    
    Args:
        context_arg: Строка с путями к файлам контекста через запятую
        
    Returns:
        Список путей к файлам контекста
    """
    if not context_arg:
        return []
    
    # Разделяем строку на отдельные пути
    paths = [path.strip() for path in context_arg.split(',')]
    
    # Проверяем существование файлов
    valid_paths = []
    for path in paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            logging.warning(f"Контекстный файл не найден: {path}")
    
    return valid_paths


def validate_input_path(path: str) -> Tuple[bool, str]:
    """
    Проверяет существование входного пути.
    
    Args:
        path: Путь к файлу или директории
        
    Returns:
        Кортеж (результат проверки, сообщение об ошибке)
    """
    if not os.path.exists(path):
        return False, f"Указанный путь не существует: {path}"
    
    if os.path.isfile(path):
        extension = os.path.splitext(path)[1].lower()
        supported_extensions = ['.kt', '.java']  # Должно соответствовать списку в CodeAnalyzer
        
        if extension not in supported_extensions:
            return False, f"Неподдерживаемый тип файла: {extension}. Поддерживаемые типы: {', '.join(supported_extensions)}"
    
    return True, ""


def analyze_code(args: argparse.Namespace) -> None:
    """
    Выполняет анализ кода согласно переданным аргументам.
    
    Args:
        args: Аргументы командной строки
    """
    # Проверяем входной путь
    is_valid, error_message = validate_input_path(args.path)
    if not is_valid:
        logging.error(error_message)
        return
    
    # Обрабатываем контекстные файлы
    context_files = process_context_files(args.context)
    if args.context and not context_files:
        logging.warning("Ни один из контекстных файлов не найден")
    
    try:
        # Создаем клиент Ollama
        client = OllamaClient()
        
        # Создаем анализатор кода
        analyzer = CodeAnalyzer(client)
        
        # Запускаем анализ
        if os.path.isfile(args.path):
            # Обрабатываем одиночный файл
            logging.info(f"Начинаем анализ файла: {args.path}")
            result = analyzer.analyze_file(args.path, context_files)
            
            if result:
                # Сохраняем результат
                analyzer._save_result(args.path, result, args.output)
                logging.info(f"Анализ файла завершен: {args.path}")
            else:
                logging.error(f"Не удалось проанализировать файл: {args.path}")
        
        else:
            # Обрабатываем директорию
            logging.info(f"Начинаем анализ директории: {args.path}")
            results = analyzer.analyze_path(
                args.path, 
                recursive=args.recursive,
                output_dir=args.output
            )
            
            # Выводим статистику
            success_count = sum(1 for r in results.values() if r.get('status') == 'success')
            error_count = len(results) - success_count
            
            logging.info(f"Анализ директории завершен: {args.path}")
            logging.info(f"Всего файлов: {len(results)}")
            logging.info(f"Успешно: {success_count}")
            logging.info(f"С ошибками: {error_count}")
    
    except Exception as e:
        logging.error(f"Произошла ошибка при анализе: {str(e)}", exc_info=True)


def main():
    """Основная функция приложения."""
    # Обрабатываем аргументы командной строки
    args = parse_args()
    
    # Настраиваем логирование
    setup_logging(args.log_level)
    
    # Запускаем анализ
    logging.info("Запуск анализатора кода")
    analyze_code(args)
    logging.info("Анализ кода завершен")


if __name__ == "__main__":
    main() 