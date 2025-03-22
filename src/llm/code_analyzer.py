import logging
from typing import Dict, Optional
from src.code_analyzer.ast_processor import ASTProcessor

class CodeAnalyzer:
    """Анализатор кода с использованием LLM"""
    
    def __init__(self):
        self.ast_processor = ASTProcessor()
        self.logger = logging.getLogger(__name__)
        
    async def analyze_file(self, file_path: str) -> Optional[Dict]:
        """Анализ файла с использованием AST и LLM"""
        try:
            # Обработка файла в зависимости от его типа
            ast_info = self.ast_processor.process_file(file_path)
            
            # Генерация промпта для LLM
            llm_prompt = self._build_llm_prompt(ast_info)
            
            # Здесь должен быть вызов LLM API
            # Пока возвращаем заглушку
            llm_analysis = {
                'description': f"Анализ класса {ast_info['name']}",
                'complexity': 'Средняя',
                'recommendations': ['Добавить комментарии', 'Улучшить обработку ошибок']
            }
            
            # Объединение результатов
            return self._combine_analysis(ast_info, llm_analysis)
            
        except FileNotFoundError as e:
            self.logger.error(f"Файл не найден: {file_path}")
            return None
        except ValueError as e:
            self.logger.error(f"Ошибка формата: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Ошибка при анализе {file_path}: {str(e)}", exc_info=True)
            return None
            
    def _build_llm_prompt(self, ast_info: Dict) -> str:
        """Построение промпта для LLM на основе AST"""
        prompt_parts = []
        
        # Добавление информации о классе
        prompt_parts.append(f"Анализ класса {ast_info['name']}:")
        
        # Добавление информации о пакете и аннотациях
        if ast_info.get('package'):
            prompt_parts.append(f"\nПакет: {ast_info['package']}")
            
        if ast_info.get('annotations'):
            prompt_parts.append("\nАннотации:")
            for ann in ast_info['annotations']:
                prompt_parts.append(f"- @{ann}")
                
        # Добавление информации о дженериках
        if ast_info.get('generics'):
            prompt_parts.append("\nПараметры типа:")
            for generic in ast_info['generics']:
                prompt_parts.append(f"- {generic}")
                
        # Добавление информации о методах
        if ast_info.get('methods'):
            prompt_parts.append("\nМетоды:")
            for method in ast_info['methods']:
                # Обработка параметров с учетом разных форматов
                params = method.get('params', [])
                if params and isinstance(params[0], (list, tuple)):
                    # Формат [(name, type), ...]
                    params_str = ', '.join(f"{name}: {type_}" for name, type_ in params)
                else:
                    # Формат [str, str, ...]
                    params_str = ', '.join(params)
                    
                annotations_str = ' '.join([f"@{ann}" for ann in method.get('annotations', [])])
                method_str = f"- {annotations_str} {method['name']}({params_str}): {method['return_type']}"
                prompt_parts.append(method_str.strip())
                
        # Добавление информации о полях
        if ast_info.get('fields'):
            prompt_parts.append("\nПоля:")
            for field in ast_info['fields']:
                annotations_str = ' '.join([f"@{ann}" for ann in field.get('annotations', [])])
                field_str = f"- {annotations_str} {field['name']}: {field['type']}"
                prompt_parts.append(field_str.strip())
                
        # Добавление информации о импортах
        if ast_info.get('imports'):
            prompt_parts.append("\nИмпорты:")
            for imp in ast_info['imports']:
                prompt_parts.append(f"- {imp}")
                
        return '\n'.join(prompt_parts)
        
    def _combine_analysis(self, ast_info: Dict, llm_analysis: Dict) -> Dict:
        """Объединение результатов анализа AST и LLM"""
        return {
            'structure': {
                'name': ast_info['name'],
                'package': ast_info.get('package', ''),
                'annotations': ast_info.get('annotations', []),
                'generics': ast_info.get('generics', []),
                'methods': ast_info.get('methods', []),
                'fields': ast_info.get('fields', [])
            },
            'semantic_analysis': llm_analysis,
            'relationships': self.ast_processor.extract_relationships(ast_info)
        } 