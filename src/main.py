import argparse
import logging
import sys

import pytest

from src.code_analyzer.code_analyzer import CodeAnalyzer


def setup_logging(verbose: bool, clear_logs: bool = False):
    """
    Настройка логирования
    
    Args:
        verbose (bool): Включить подробное логирование
        clear_logs (bool): Очистить лог-файлы перед запуском
    """
    # Настраиваем уровень логирования
    level = logging.DEBUG if verbose else logging.INFO

    # Отключаем лишние логи от библиотек
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Очищаем лог если требуется
    if clear_logs:
        try:
            with open('code_analysis.log', 'w', encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            print(f"Ошибка при очистке лога: {str(e)}")

    # Настраиваем формат логирования
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Настраиваем вывод в файл
    file_handler = logging.FileHandler('code_analysis.log', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Настраиваем вывод в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Конфигурируем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def run_tests(verbose: bool, clear_logs: bool):
    """Запуск тестов с настройками логирования"""
    # Формируем аргументы для pytest
    pytest_args = [
        # "-v",  # Подробный вывод
        # "--tb=long",  # Полные трейсбеки
        # "--log-cli-level", "DEBUG" if verbose else "INFO",
        # "--log-file", "test_results.log",
        # "--log-file-level", "DEBUG",
        # "--cache-clear"  # Очистка кэша pytest
    ]

    # Передаем управление pytest
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description='Анализатор кода с использованием LLM')
    parser.add_argument('input_dir', nargs='?', help='Директория с исходным кодом (необязательно при запуске тестов)')
    parser.add_argument('--output', default='docs/analysis.md', help='Путь к файлу с результатами')
    parser.add_argument('--cache-dir', default='.cache', help='Директория для кэширования')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    parser.add_argument('--clear-logs', action='store_true', help='Очистить лог-файлы перед запуском')
    parser.add_argument('--test', action='store_true', help='Запустить тесты вместо анализа')

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.verbose, args.clear_logs)

    if args.test:
        # Запуск тестов
        run_tests(args.verbose, args.clear_logs)
    else:
        # Обычный режим анализа
        if not args.input_dir:
            logging.error("Не указана директория с исходным кодом")
            sys.exit(1)

        try:
            analyzer = CodeAnalyzer(cache_dir=args.cache_dir)
            results = analyzer.analyze_directory(args.input_dir)
            analyzer.generate_documentation(args.output)
            logging.info(f"Анализ завершен. Результаты сохранены в {args.output}")
        except Exception as e:
            logging.error(f"Ошибка при выполнении: {str(e)}", exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
