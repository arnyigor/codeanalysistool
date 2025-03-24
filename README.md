
# Android Code Analysis Tool
*Преварительное описание*

## О проекте
Инструмент для автоматического создания README.md с описанием Android-проекта на Java/Kotlin. Анализирует код в локальной файловой системе с помощью модели LLM из Ollama, сохраняя все данные локально.

## Требования
- Python 3.10+
- Ollama (https://ollama.ai/)
- Локальная модель LLM (например, `codellama:7b`)

## Активация окружения
```bash
source myenv/bin/activate
```

## Тесты
```bash
pytest src/test/ -v --log-cli-level=DEBUG
```
Флаг `-v` — подробный вывод.
Флаг `--log-cli-level=DEBUG` — отображение логов из тестов.

## Установка
```bash
# Установите зависимости
pip install -r requirements.txt

# Загрузите модель Ollama (если еще не установлена)
ollama pull codellama
```

## Использование
```bash
# Запустите анализатор
python src/main.py src/test/resources --output docs/analysis.md --cache-dir .cache --clear-logs
```
```bash
# Запустите Тесты
python src/main.py src/test/resources --verbose --clear-logs --test

```

## Структура проекта
```
project/
├── src/          # Основной код
│   ├── file_processor/   # Обработка файлов
│   ├── llm/       # Взаимодействие с Ollama
│   └── documentation/    # Генерация документации
├── logs/         # Лог-файлы
└── README.md     # Этот файл
```

## Функциональность
- 🔍 Рекурсивный анализ Java/Kotlin/XML файлов
- 📝 Генерация структурированного README.md:
  - Общее описание проекта
  - Описание классов/методов
  - Схемы взаимодействий компонентов
  - Оглавление и навигация
- 🚨 Логирование всех операций:
  - `logs/processing.log` - обработка файлов
  - `logs/analysis.log` - анализ кода
  - `logs/relationships.log` - взаимосвязи компонентов

## Пример выходных данных
```markdown
# MyAndroidApp Анализ

## Структура проекта
- `src/main/java/com/example/`
  - MainActivity.kt → Основной экран приложения
  - DataFetcher.kt → Обработка API-запросов

## Классы
### MainActivity
**Цель**: Отображение данных из DataFetcher
**Ключевые методы**:
- `onCreate()`: Инициализация UI
- `loadData()`: Запуск асинхронного запроса

## Взаимодействия
![Класс-диаграмма](./diagram.png)
```

## Настройка
Модифицируйте `src/documentation/md_generator.py` для изменения:
- Формата вывода
- Уровня детализации
- Стиля описаний

## Поддержка
Для вопросов обращайтесь к:
- **Андрей Разработчик** - andrey@dev.com

## Лицензия
MIT License
```

Дополнительные возможности:
- Инкрементальная обработка (через `--incremental` флаг)
- Настройка модели через `OLLAMA_MODEL` переменную окружения
- Логирование в JSON формате (включить через `--json-logs`)

Хотите добавить какие-то конкретные детали или уточнить какие-то разделы?