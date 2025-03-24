import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.llm.llm_client import OllamaClient
from .doc_analyzer import DocAnalyzer

class CodeAnalyzer:
    """
    Анализатор кода с использованием LLM для генерации документации
    и последующим анализом сгенерированной документации.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Инициализация анализатора кода.
        
        Args:
            cache_dir (str): Директория для кэширования результатов
        """
        self.cache_dir = cache_dir
        
        # Создаем директорию для кэша если её нет
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.analyzed_files = {}
        self.relationships = {}
        
        # Инициализация клиентов
        self.llm_client = OllamaClient()
        self.doc_analyzer = DocAnalyzer()
        
        logging.info("Инициализация CodeAnalyzer завершена")

    def analyze_directory(self, directory: str) -> List[Dict]:
        """
        Анализирует директорию с кодом.
        
        Args:
            directory (str): Путь к директории с кодом
            
        Returns:
            List[Dict]: Список результатов анализа для каждого файла
        """
        results = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.kt', '.java')):
                    file_path = os.path.join(root, file)
                    try:
                        result = self.process_file(file_path)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
                        
        return results

    def process_file(self, file_path: str) -> Optional[Dict]:
        """
        Обрабатывает один файл.
        
        Args:
            file_path (str): Путь к файлу
            
        Returns:
            Optional[Dict]: Результат анализа файла или None в случае ошибки
        """
        file_type = 'kotlin' if file_path.endswith('.kt') else 'java'
        cache_file = os.path.join(self.cache_dir, os.path.basename(file_path) + '.json')
        
        # Проверяем кэш
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Ошибка чтения кэша для {file_path}: {str(e)}")
        
        try:
            # Читаем файл
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            logging.info(f"Начало обработки файла: {file_path}")
            
            # Шаг 1: Генерация документированной версии кода
            doc_result = self.llm_client.analyze_code(code, file_type)
            if doc_result.get('error'):
                logging.error(f"Ошибка при генерации документации: {doc_result['error']}")
                return None
                
            documented_code = doc_result['documented_code']
            
            # Шаг 2: Анализ документированного кода
            analysis_result = self.doc_analyzer.analyze_file(documented_code, file_type)
            if analysis_result.get('error'):
                logging.error(f"Ошибка при анализе документации: {analysis_result['error']}")
                return None
                
            # Добавляем информацию о модели
            result = {
                'file_path': file_path,
                'analysis': analysis_result,
                'model_info': doc_result['model_info']
            }
            
            # Сохраняем в кэш
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logging.info(f"Результаты сохранены в кэш: {cache_file}")
            except Exception as e:
                logging.warning(f"Ошибка сохранения в кэш для {file_path}: {str(e)}")
                
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
            return None

    def analyze_file(self, file_path: str) -> Optional[Dict]:
        """
        Анализирует файл и возвращает результат анализа.
        
        Args:
            file_path: Путь к файлу для анализа
            
        Returns:
            Optional[Dict]: Результат анализа или None в случае ошибки
        """
        try:
            # Проверяем кэш
            cache_key = self._get_cache_key(file_path)
            if cache_key in self.analyzed_files:
                logging.info(f"Используем кэшированный результат для {file_path}")
                return self.analyzed_files[cache_key]

            # Определяем тип файла
            file_type = self._get_file_type(file_path)
            if not file_type:
                logging.warning(f"Неподдерживаемый тип файла: {file_path}")
                return None

            # Читаем содержимое файла
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logging.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
                return None

            if not content.strip():
                logging.warning(f"Файл пуст: {file_path}")
                return None

            # Анализируем код
            try:
                result = self.llm_client.analyze_code(content, file_type)

                # Проверяем структуру результата
                if not isinstance(result, dict):
                    raise ValueError(f"Неожиданный формат результата: {type(result)}")

                required_fields = ['description', 'components']
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    raise ValueError(
                        f"В результате отсутствуют обязательные поля: {missing_fields}")

                # Сохраняем результат в кэш
                self.analyzed_files[cache_key] = result
                logging.info(f"Успешно проанализирован файл: {file_path}")
                return result

            except Exception as e:
                logging.error(f"Ошибка при анализе файла {file_path}: {str(e)}")
                return None

        except Exception as e:
            logging.error(f"Неожиданная ошибка при обработке файла {file_path}: {str(e)}")
            return None

    def _process_kotlin_file(self, file_path: str, content: str) -> Dict:
        """Обработка Kotlin файла"""
        logging.info(f"Анализ Kotlin файла: {file_path}")

        try:
            result = self.llm_client.analyze_code(content, 'kotlin')

            # Обновляем взаимосвязи
            for class_info in result.get('classes', []):
                self._update_relationships(class_info)

            return result

        except Exception as e:
            logging.error(f"Ошибка анализа Kotlin файла {file_path}: {str(e)}")
            return {'error': str(e)}

    def _process_java_file(self, file_path: str, content: str) -> Dict:
        """Обработка Java файла"""
        logging.info(f"Анализ Java файла: {file_path}")

        try:
            result = self.llm_client.analyze_code(content, 'java')

            # Обновляем взаимосвязи
            for class_info in result.get('classes', []):
                self._update_relationships(class_info)

            return result

        except Exception as e:
            logging.error(f"Ошибка анализа Java файла {file_path}: {str(e)}")
            return {'error': str(e)}

    def _process_xml_file(self, file_path: str, content: str) -> Dict:
        """Обработка Android XML файла"""
        logging.info(f"Анализ XML файла: {file_path}")

        xml_type = self._determine_xml_type(file_path)

        prompt = """[СИСТЕМНЫЕ ТРЕБОВАНИЯ]
Вы - опытный разработчик Android, специализирующийся на UI/UX. Ваша задача - проанализировать XML файл и создать подробную документацию.

[ПРАВИЛА АНАЛИЗА]
1. Определять назначение и структуру макета
2. Выявлять UI компоненты и их взаимосвязи

[ФОРМАТ ДОКУМЕНТАЦИИ]
- Писать на русском языке
- Давать исчерпывающие описания
- Указывать все ресурсные зависимости
- Документировать особенности верстки

[ТРЕБУЕМЫЙ АНАЛИЗ]
Проанализируйте следующий Android XML файл ({xml_type}):
```xml
{code}
```

[СТРУКТУРА ОТВЕТА]
Предоставьте результат в формате JSON:
{{
    "type": "{xml_type}",
    "purpose": "подробное описание назначения файла",
    "layout_type": "ConstraintLayout/LinearLayout/etc",
    "components": [
        {{
            "type": "тип компонента",
            "id": "идентификатор",
            "purpose": "назначение компонента",
            "accessibility": {{
                "content_description": "описание для TalkBack",
                "importance": "важность для скринридеров",
                "issues": ["проблемы доступности"]
            }},
            "layout_params": {{
                "width": "wrap_content/match_parent/dimension",
                "height": "wrap_content/match_parent/dimension",
                "constraints": ["описание ограничений"],
                "margins": ["отступы"],
                "padding": ["внутренние отступы"]
            }},
            "style": {{
                "theme": "используемая тема",
                "custom_attributes": ["пользовательские атрибуты"],
                "material_components": true/false
            }}
        }}
    ],
    "resource_references": [
        {{
            "type": "тип ресурса (string/color/dimen/etc)",
            "name": "имя ресурса",
            "usage": "где используется"
        }}
    ],
    "custom_attributes": [
        {{
            "name": "имя атрибута",
            "purpose": "назначение",
            "type": "тип значения",
            "default_value": "значение по умолчанию"
        }}
    ],
    "layout_analysis": {{
        "depth": "глубина вложенности",
        "performance_issues": ["потенциальные проблемы производительности"],
        "accessibility_score": "оценка доступности (1-5)",
        "material_design_compliance": ["отклонения от Material Design"],
        "recommendations": ["рекомендации по улучшению"]
    }}
}}"""

        try:
            result = self.llm_client.analyze_code(content, 'xml')
            return result

        except Exception as e:
            logging.error(f"Ошибка анализа XML файла {file_path}: {str(e)}")
            return {'error': str(e)}

    def _determine_xml_type(self, file_path: str) -> str:
        """Определение типа Android XML файла"""
        path = Path(file_path)
        if 'layout' in str(path):
            return 'layout'
        elif 'menu' in str(path):
            return 'menu'
        elif 'values' in str(path):
            return 'values'
        else:
            return 'unknown'

    def generate_documentation(self, output_file: str):
        """Генерация окончательной документации в формате README"""
        # Подготовка разделов
        sections = {
            'overview': self._generate_overview(),
            'components': self._generate_components_section(),
            'relationships': self._generate_relationships_section(),
            'layouts': self._generate_layouts_section()
        }

        # Генерация markdown
        markdown = f"""# Документация проекта

## Содержание
- [Обзор проекта](#обзор-проекта)
- [Компоненты](#компоненты)
- [Взаимосвязи](#взаимосвязи)
- [Макеты](#макеты)

## Обзор проекта
{sections['overview']}

## Компоненты
{sections['components']}

## Взаимосвязи
{sections['relationships']}

## Макеты
{sections['layouts']}
"""

        # Создаем директорию для документации, если она не существует
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохранение в файл с явным указанием кодировки
        output_path.write_text(markdown, encoding='utf-8')
        logging.info(f"Документация сгенерирована в {output_file}")

    def _generate_overview(self) -> str:
        """Генерация раздела общего обзора проекта"""
        components = [info for info in self.analyzed_files.values() if 'classes' in info]
        android_components = sum(
            1 for comp in components
            for cls in comp['classes']
            if cls.get('is_android_component', False)
        )

        return f"""Проект содержит:
- {len(components)} исходных файлов
- {android_components} Android-компонентов
- {len(self.relationships)} взаимосвязей между классами"""

    def _generate_components_section(self) -> str:
        """Генерация раздела документации компонентов"""
        sections = []

        for file_info in self.analyzed_files.values():
            if 'classes' not in file_info:
                continue

            for cls in file_info['classes']:
                # Генерация KDoc/JavaDoc для класса
                doc_comment = self._generate_doc_comment(cls)

                sections.append(f"""### {cls['name']}
{cls['purpose']}

**Тип:** {'Android-компонент' if cls['is_android_component'] else 'Обычный класс'}
**Родительские классы:** {', '.join(cls['superclasses'])}

#### Методы:
{self._format_methods(cls['methods'])}

#### Зависимости:
{', '.join(cls['dependencies'])}

#### Сгенерированная документация:
```kotlin
{doc_comment}
```
""")

        return '\n'.join(sections)

    def _generate_relationships_section(self) -> str:
        """Генерация раздела документации взаимосвязей"""
        sections = []

        for class_name, relations in self.relationships.items():
            sections.append(f"""### {class_name}
- **Наследует:** {', '.join(relations['superclasses']) or 'нет'}
- **Зависит от:** {', '.join(relations['dependencies']) or 'нет'}
""")

        return '\n'.join(sections)

    def _generate_layouts_section(self) -> str:
        """Генерация раздела документации макетов"""
        sections = []

        for file_info in self.analyzed_files.values():
            if file_info.get('type') != 'layout':
                continue

            sections.append(f"""### {file_info['purpose']}

#### Компоненты:
{self._format_components(file_info['components'])}

#### Используемые ресурсы:
{', '.join(file_info['resource_references']) or 'нет'}
""")

        return '\n'.join(sections)

    def _generate_doc_comment(self, cls: Dict) -> str:
        """Генерирует документационный комментарий для класса."""
        purpose = cls.get('purpose', 'Требует описания')
        doc_lines = [
            '/**',
            f' * {purpose}',
            ' *'
        ]
        
        # Добавляем информацию о родительских классах
        if cls.get('superclasses'):
            doc_lines.append(f' * Наследуется от: {", ".join(cls["superclasses"])}')
            
        # Добавляем информацию о зависимостях
        if cls.get('dependencies'):
            doc_lines.append(f' * Зависимости: {", ".join(cls["dependencies"])}')
            
        # Добавляем документацию методов
        if cls.get('methods'):
            for method in cls['methods']:
                if method.get('parameters'):
                    param_docs = []
                    for p in method['parameters']:
                        param_type = p.get('type', 'unknown')
                        param_name = p.get('name', 'unknown')
                        param_desc = p.get('purpose', 'Требует описания')
                        param_docs.append(f' * @param {param_name} [{param_type}] {param_desc}')
                    if param_docs:
                        doc_lines.extend(param_docs)
                        
        doc_lines.append(' */')
        return '\n'.join(doc_lines)

    def _format_methods(self, methods: List[Dict]) -> str:
        """Форматирование списка методов как markdown"""
        formatted = []
        for m in methods:
            params = m.get('parameters', [])
            returns = m.get('return', {})

            # Преобразуем returns в словарь, если это строка
            if isinstance(returns, str):
                returns = {'type': 'Unit', 'description': returns}

            # Форматируем параметры
            param_str = ', '.join(f'{p["name"]}: {p["type"]}' for p in params)

            # Форматируем описание
            desc = [f"- `{m['name']}({param_str})`: {m['purpose']}"]

            # Добавляем описания параметров
            if params:
                desc.append("  - Параметры:")
                for p in params:
                    desc.append(f"    - `{p['name']}` ({p['type']}): {p['description']}")

            # Добавляем информацию о возвращаемом значении
            if returns:
                return_type = returns.get('type', 'Unit')
                return_desc = returns.get('description', '')
                desc.append(f"  - Возвращает: [{return_type}] {return_desc}")

            formatted.append('\n'.join(desc))

        return '\n\n'.join(formatted)

    def _format_components(self, components: List[Dict]) -> str:
        """Форматирование списка компонентов как markdown"""
        return '\n'.join(
            f"- `{c['type']}` (id: `{c['id']}`): {c['purpose']}"
            for c in components
        )

    def _update_relationships(self, class_info: Dict):
        """Обновление информации о взаимосвязях"""
        class_name = class_info['name']
        logging.info(f"Обновление связей для класса: {class_name}")

        # Добавляем информацию о наследовании
        if class_info.get('superclasses'):
            logging.info(
                f"Класс {class_name} наследуется от: {', '.join(class_info['superclasses'])}")

        # Добавляем информацию о зависимостях
        if class_info.get('dependencies'):
            logging.info(
                f"Класс {class_name} зависит от: {', '.join(class_info['dependencies'])}")

        self.relationships[class_name] = {
            'superclasses': class_info.get('superclasses', []),
            'dependencies': class_info.get('dependencies', [])
        }

    def _get_file_type(self, file_path: str) -> str:
        """
        Определяет тип файла по его расширению.
        
        Args:
            file_path (str): Путь к файлу
            
        Returns:
            str: Тип файла ('kotlin' или 'java')
            
        Raises:
            ValueError: Если расширение файла не поддерживается
        """
        extension = file_path.lower().split('.')[-1]
        
        if extension == 'kt':
            return 'kotlin'
        elif extension == 'java':
            return 'java'
        else:
            raise ValueError(f"Неподдерживаемое расширение файла: .{extension}")
