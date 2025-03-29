import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

from src.code_analyzer.code_analyzer import CodeAnalyzer
from src.llm.llm_client import OllamaClient

LOG_FILE = os.path.abspath("file_processor.log")

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

setup_logging()


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


def process_files(path: str, output_dir: str = "output", recursive: bool = True,
                  context_files: List[str] = None, executor=None) -> None:
    """
    Обрабатывает файлы и генерирует документацию.
    
    Args:
        path: Путь к файлу или директории
        output_dir: Директория для сохранения результатов
        recursive: Рекурсивно обрабатывать директории
        context_files: Список путей к контекстным файлам
        executor: Объект CodeAnalyzer для анализа
    """
    # Создаем экземпляр анализатора, если не передан
    if not executor:
        from src.code_analyzer.code_analyzer import CodeAnalyzer
        executor = CodeAnalyzer()
        logging.info(f"Создан новый экземпляр CodeAnalyzer")
    else:
        logging.info(f"Используется существующий экземпляр CodeAnalyzer")

    # Преобразуем пути в абсолютные для лучшего логирования
    abs_path = os.path.abspath(path)
    abs_output_dir = os.path.abspath(output_dir)
    logging.info(f"Абсолютный путь к целевому файлу/директории: {abs_path}")
    logging.info(f"Абсолютный путь к выходной директории: {abs_output_dir}")

    # Создаем выходную директорию
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir, exist_ok=True)
        logging.info(f"Создана выходная директория: {abs_output_dir}")
    else:
        logging.info(f"Выходная директория уже существует: {abs_output_dir}")

    # Обрабатываем контекстные файлы, если они указаны
    if context_files:
        logging.info(f"Предоставлены контекстные файлы: {', '.join(context_files)}")
        # Проверяем существование файлов
        for ctx_file in context_files:
            if not os.path.exists(ctx_file):
                logging.warning(f"Контекстный файл не найден: {ctx_file}")
            else:
                logging.info(f"Контекстный файл подтвержден: {ctx_file}")

    # Обрабатываем файлы
    path_obj = Path(abs_path)
    if path_obj.is_file():
        # Обрабатываем отдельный файл
        logging.info(f"Целевой путь является файлом: {abs_path}")

        # Проверяем расширение файла
        file_ext = path_obj.suffix.lower()
        logging.info(f"Расширение файла: {file_ext}")

        # Запускаем анализ
        logging.info(f"Запуск анализа для файла: {abs_path}")
        result = executor.analyze_path(abs_path, output_dir=abs_output_dir)

        if result:
            logging.info(f"Анализ файла успешно завершен: {abs_path}")
            # Выводим статистику
            success_count = sum(1 for r in result.values() if r.get('status') == 'success')
            error_count = len(result) - success_count
            logging.info(f"Статус анализа: успешно={success_count}, ошибки={error_count}")
        else:
            logging.error(f"Анализ файла не вернул результатов: {abs_path}")

    elif path_obj.is_dir():
        # Обрабатываем директорию
        logging.info(f"Целевой путь является директорией: {abs_path}")
        logging.info(f"Рекурсивный обход: {'включен' if recursive else 'выключен'}")

        # Запускаем анализ директории
        logging.info(f"Запуск анализа для директории: {abs_path}")
        result = executor.analyze_path(abs_path, recursive=recursive, output_dir=abs_output_dir)

        if result:
            logging.info(f"Анализ директории успешно завершен: {abs_path}")
            # Выводим статистику
            success_count = sum(1 for r in result.values() if r.get('status') == 'success')
            error_count = len(result) - success_count
            logging.info(f"Всего файлов: {len(result)}")
            logging.info(f"Успешно обработано: {success_count}")
            logging.info(f"С ошибками: {error_count}")

            # Список обработанных файлов
            processed_files = list(result.keys())
            if processed_files:
                logging.debug(f"Обработаны файлы: {', '.join(processed_files[:5])}" +
                              (f" и еще {len(processed_files) - 5} файлов" if len(
                                  processed_files) > 5 else ""))
        else:
            logging.error(f"Анализ директории не вернул результатов: {abs_path}")

    else:
        logging.error(f"Указанный путь не существует: {abs_path}")
        return

    logging.info(f"Обработка завершена. Результаты сохранены в {abs_output_dir}")

    # Проверяем, что в выходной директории есть файлы
    output_files = list(Path(abs_output_dir).rglob("*.*"))
    if output_files:
        logging.info(f"В выходной директории создано {len(output_files)} файлов")
        logging.debug(f"Примеры созданных файлов: {', '.join([str(f) for f in output_files[:3]])}" +
                      (f" и еще {len(output_files) - 3} файлов" if len(output_files) > 3 else ""))
    else:
        logging.warning(f"В выходной директории не обнаружено созданных файлов!")


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
        process_files(args.path, args.output, args.recursive, context_files, analyzer)

    except Exception as e:
        logging.error(f"Произошла ошибка при анализе: {str(e)}", exc_info=True)


def main():
    """Основная функция приложения."""
    # Обрабатываем аргументы командной строки
    args = parse_args()

    # Настраиваем логирование
    setup_logging()

    # Запускаем анализ
    logging.info("Запуск анализатора кода")
    analyze_code(args)
    logging.info("Анализ кода завершен")


if __name__ == "__main__":
    main()
