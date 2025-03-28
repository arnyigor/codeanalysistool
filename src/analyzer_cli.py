#!/usr/bin/env python3
"""
Интерфейс командной строки для запуска анализатора кода.
"""

import argparse
import logging
import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.code_analyzer.file_processor import main as file_processor_main
from src.main import main as main_processor

if __name__ == "__main__":
    # Оригинальные аргументы
    original_args = sys.argv.copy()
    
    # Проверяем режим работы
    if len(sys.argv) > 1:
        if sys.argv[1] == "--legacy":
            # Явно указан legacy режим
            sys.argv.pop(1)
            print("Запуск в legacy-режиме (обработка отдельных файлов и папок)")
            file_processor_main()
        elif sys.argv[1] == "--test":
            # Режим тестирования (относится к стандартному режиму)
            print("Запуск тестов")
            main_processor()
        elif sys.argv[1].startswith("-"):
            # Если первый аргумент - опция для стандартного режима
            print("Запуск в стандартном режиме (анализ проекта)")
            main_processor()
        else:
            # Первый аргумент - путь
            path_arg = sys.argv[1]
            
            # Проверяем, существует ли путь
            if os.path.exists(path_arg):
                # Определяем режим по типу пути
                if "--legacy" in original_args:
                    # Если был указан --legacy, используем legacy режим
                    print("Запуск в legacy-режиме (обработка отдельных файлов и папок)")
                    file_processor_main()
                else:
                    # Автоматически выбираем режим на основе типа пути
                    if os.path.isfile(path_arg):
                        # Для отдельных файлов используем legacy режим
                        print(f"Обнаружен файл. Запуск в режиме обработки отдельных файлов для: {path_arg}")
                        file_processor_main()
                    else:
                        # Для директорий используем стандартный режим
                        print(f"Обнаружена директория. Запуск в режиме анализа проекта для: {path_arg}")
                        main_processor()
            else:
                print(f"Ошибка: путь не существует: {path_arg}")
                sys.exit(1)
    else:
        # Без аргументов запускаем стандартный режим
        print("Запуск в стандартном режиме (анализ проекта)")
        main_processor() 