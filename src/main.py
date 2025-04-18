import argparse
import logging
import os
import shutil
import sys

import pytest

from src.code_analyzer.code_analyzer import CodeAnalyzer


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


def clean_cache(cache_dir: str):
    """
    Очистка директории кэша
    
    Args:
        cache_dir (str): Путь к директории кэша
    """
    try:
        if os.path.exists(cache_dir):
            # Удаляем все содержимое директории
            shutil.rmtree(cache_dir)
            logging.info(f"Кэш очищен: {cache_dir}")

            # Создаем пустую директорию заново
            os.makedirs(cache_dir)
            os.makedirs(os.path.join(cache_dir, 'docs'), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, 'test_results'), exist_ok=True)
            logging.info("Структура кэша восстановлена")
        else:
            logging.info(f"Директория кэша не существует: {cache_dir}")
            # Создаем структуру директорий
            os.makedirs(cache_dir)
            os.makedirs(os.path.join(cache_dir, 'docs'), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, 'test_results'), exist_ok=True)
            logging.info("Структура кэша создана")
    except Exception as e:
        logging.error(f"Ошибка при очистке кэша: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Анализатор кода с использованием LLM')
    parser.add_argument('input_dir', nargs='?',
                        help='Директория с исходным кодом (необязательно при запуске тестов)')
    parser.add_argument('--output', default='docs/analysis.md', help='Путь к файлу с результатами')
    parser.add_argument('--cache-dir', default='.cache', help='Директория для кэширования')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    parser.add_argument('--clear-logs', action='store_true',
                        help='Очистить лог-файлы перед запуском')
    parser.add_argument('--test', action='store_true', help='Запустить тесты вместо анализа')
    parser.add_argument('--clean-cache', action='store_true', help='Очистить кэш перед запуском')

    args = parser.parse_args()

    # Очистка кэша если требуется
    if args.clean_cache:
        logging.info("Очистка кэша...")
        clean_cache(args.cache_dir)

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
            analyzer.analyze_directory(args.input_dir)
            analyzer.generate_documentation(args.output)
            logging.info(f"Анализ завершен. Результаты сохранены в {args.output}")
        except Exception as e:
            logging.error(f"Ошибка при выполнении: {str(e)}", exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
