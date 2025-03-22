Опишу подробный процесс анализа и документирования кода. Самым главным компонентом будет модуль взаимодействия с LLM, так как именно он отвечает за понимание и описание кода.

1. Ключевой файл: src/llm/code_analyzer.py 

```python
from typing import Dict, List, Optional
import asyncio
from pathlib import Path

class CodeAnalyzer:
    def __init__(self):
        self.context_history = {}
        self.relationships = {}
        self.current_analysis = {}
        
    async def analyze_file(self, file_path: str, content: str) → Dict:
        """
        Основной метод анализа файла
        """
        # Формируем промпт для LLM
        prompt = self._build_analysis_prompt(file_path, content)
        
        # Получаем описание от LLM
        description = await self._get_llm_analysis(prompt)
        
        # Обрабатываем и структурируем результат
        analysis = self._process_analysis(description)
        
        # Обновляем контекст
        self._update_context(file_path, analysis)
        
        return analysis

    def _build_analysis_prompt(self, file_path: str, content: str) → str:
        """
        Создает специализированный промпт для LLM
        """
        file_type = Path(file_path).suffix
        context = self._get_relevant_context(file_path)
        
        prompt = f"""
        Analyze this {file_type} code and provide:
        1. Main purpose of the code
        2. Key components and their responsibilities
        3. Interactions with other components
        4. Important parameters and their usage
        
        Previous context:
        {context}
        
        Code to analyze:
        {content}
        
        Please provide analysis in the following format:
        PURPOSE: <clear description of main purpose>
        COMPONENTS: <list of key components>
        INTERACTIONS: <list of interactions>
        PARAMETERS: <important parameters>
        """
        return prompt

    async def _get_llm_analysis(self, prompt: str) → str:
        """
        Взаимодействие с Ollama LLM
        """
        # Конфигурация для CodeLlama
        system_prompt = """
        You are an expert Android developer analyzing code.
        Focus on:
        - Clear and concise descriptions
        - Practical usage and interactions
        - Important implementation details
        Avoid:
        - Generic descriptions
        - Implementation minutiae
        - Non-essential details
        """
        
        try:
            # Вызов Ollama API
            response = await self._call_ollama(
                model="codellama:7b",
                prompt=prompt,
                system=system_prompt,
                temperature=0.2
            )
            return response
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}", exc_info=True)
            return f"ERROR: Could not analyze code: {str(e)}"

    def _process_analysis(self, raw_analysis: str) → Dict:
        """
        Структурирует ответ LLM в формализованный формат
        """
        sections = {
            'purpose': '',
            'components': [],
            'interactions': [],
            'parameters': [],
            'examples': []
        }
        
        # Парсинг ответа LLM в структурированный формат
        current_section = None
        for line in raw_analysis.split('\n'):
            if line.startswith('PURPOSE:'):
                current_section = 'purpose'
                sections[current_section] = line.replace('PURPOSE:', '').strip()
            elif line.startswith('COMPONENTS:'):
                current_section = 'components'
            elif line.startswith('INTERACTIONS:'):
                current_section = 'interactions'
            elif line.startswith('PARAMETERS:'):
                current_section = 'parameters'
            elif line.strip() and current_section in ['components', 'interactions', 'parameters']:
                sections[current_section].append(line.strip())
        
        return sections

    def _update_context(self, file_path: str, analysis: Dict):
        """
        Обновляет контекст для связанных файлов
        """
        self.context_history[file_path] = analysis
        
        # Обновляем связи между файлами
        for interaction in analysis['interactions']:
            related_file = self._extract_related_file(interaction)
            if related_file:
                if related_file not in self.relationships:
                    self.relationships[related_file] = set()
                self.relationships[related_file].add(file_path)

    def _get_relevant_context(self, file_path: str) → str:
        """
        Собирает релевантный контекст для анализа
        """
        context = []
        
        # Добавляем контекст из связанных файлов
        if file_path in self.relationships:
            for related_file in self.relationships[file_path]:
                if related_file in self.context_history:
                    context.append(f"Related file {related_file}:")
                    context.append(self.context_history[related_file]['purpose'])
        
        return "\n".join(context)

    def generate_documentation(self) → str:
        """
        Генерирует финальную документацию
        """
        docs = []
        
        # Добавляем заголовок
        docs.append("# Project Documentation\n")
        
        # Генерируем оглавление
        docs.append("## Table of Contents\n")
        docs.append("- [Overview](#overview)")
        docs.append("- [Components](#components)")
        docs.append("- [Interactions](#interactions)\n")
        
        # Добавляем обзор
        docs.append("## Overview\n")
        overview = self._generate_project_overview()
        docs.append(overview)
        
        # Документируем компоненты
        docs.append("## Components\n")
        for file_path, analysis in self.context_history.items():
            docs.append(f"### {Path(file_path).name}\n")
            docs.append(f"**Purpose:** {analysis['purpose']}\n")
            
            if analysis['components']:
                docs.append("**Key Components:**")
                for component in analysis['components']:
                    docs.append(f"- {component}")
                docs.append("")
            
            if analysis['parameters']:
                docs.append("**Important Parameters:**")
                for param in analysis['parameters']:
                    docs.append(f"- {param}")
                docs.append("")
        
        # Документируем взаимодействия
        docs.append("## Interactions\n")
        for file_path, related_files in self.relationships.items():
            docs.append(f"### {Path(file_path).name}")
            docs.append("Interacts with:")
            for related in related_files:
                docs.append(f"- {Path(related).name}")
            docs.append("")
        
        return "\n".join(docs)
```

2. Процесс анализа:

a) Подготовка:
- Создание рабочей копии проекта
- Инициализация контекста и истории анализа
- Настройка логирования

b) Анализ файлов:
- Рекурсивный обход файлов проекта
- Для каждого файла:
  - Чтение содержимого
  - Получение релевантного контекста
  - Формирование специализированного промпта
  - Отправка в LLM
  - Обработка и структурирование ответа
  - Обновление контекста и связей

c) Документирование:
- Генерация структурированного README
- Создание оглавления
- Документирование компонентов
- Описание взаимодействий
- Добавление примеров использования

3. Особенности работы:

- Контекстный анализ: каждый файл анализируется с учетом уже проанализированных связанных файлов
- Структурированный вывод: все описания форматируются в единый стиль
- Отслеживание взаимосвязей: автоматическое определение и документирование связей между компонентами
- Безопасность: работа с копией проекта
- Логирование: подробное протоколирование всех этапов анализа

4. Использование:

```python
# Пример использования
async def main(project_path: str):
    # Инициализация анализатора
    analyzer = CodeAnalyzer()
    
    # Создание рабочей копии проекта
    project_manager = ProjectManager()
    work_path = project_manager.create_project_copy(project_path)
    
    try:
        # Обход файлов
        for file_path in FileWalker().walk_directory(work_path):
            content = CodeReader().read_file(file_path)
            
            # Анализ файла
            analysis = await analyzer.analyze_file(file_path, content)
            
            # Логирование результатов
            logger.debug(f"Analyzed {file_path}: {analysis['purpose']}")
        
        # Генерация документации
        documentation = analyzer.generate_documentation()
        
        # Сохранение README
        readme_path = Path(project_path) / "README.md"
        readme_path.write_text(documentation)
        
    finally:
        # Очистка рабочей копии
        project_manager.cleanup_workspace(work_path)
```

Этот подход обеспечивает:
- Последовательный анализ кода
- Сохранение контекста между файлами
- Структурированную документацию
- Безопасность исходного кода
- Подробное логирование процесса

что нужно добавить в файл prompt.py чтобы llm хорошо понимала задачу?