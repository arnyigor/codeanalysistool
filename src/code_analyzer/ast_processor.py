from typing import Dict, List, Optional
import javalang
from pathlib import Path
import re

class ASTProcessor:
    """Процессор для анализа кода через AST"""
    
    def __init__(self):
        self.class_info = {}
        self.relationships = {}
        
    def process_file(self, file_path: str) -> Dict:
        """Обработка файла на основе его расширения"""
        path = Path(file_path)
        
        # Проверяем тип файла
        if path.suffix not in ['.java', '.kt']:
            raise ValueError("Неподдерживаемый тип файла")
        
        # Проверяем существование файла
        if not path.exists():
            raise Exception("Ошибка при обработке: файл не существует")
        
        # Обрабатываем файл
        if path.suffix == '.java':
            return self.process_java_file(file_path)
        else:  # .kt
            return self.process_kotlin_file(file_path)
            
    def _simplify_java_type(self, type_str: str) -> str:
        """Упрощает строковое представление Java типа"""
        if not type_str:
            return 'void'
        
        # Если это уже простая строка, возвращаем как есть
        if not any(x in type_str for x in ['BasicType', 'ReferenceType']):
            return type_str
        
        # Обработка базовых типов
        if 'BasicType' in type_str:
            match = re.search(r'name=(\w+)', type_str)
            return match.group(1) if match else type_str
        
        # Обработка ссылочных типов
        if 'ReferenceType' in type_str:
            # Извлекаем имя типа
            name_match = re.search(r'name=(\w+)', type_str)
            if not name_match:
                return 'Object'
            
            base_type = name_match.group(1)
            
            # Проверяем специальные случаи
            if 'CompletableFuture' in type_str:
                return 'CompletableFuture'
            
            # Проверяем наличие аргументов типа
            if 'arguments=' in type_str:
                args_match = re.search(r'arguments=\[(.*?)\]', type_str)
                if args_match:
                    args_str = args_match.group(1)
                    type_args = []
                    for arg in args_str.split(','):
                        if 'type=' in arg:
                            arg_match = re.search(r'type=(\w+)', arg)
                            if arg_match:
                                type_args.append(arg_match.group(1))
                    if type_args:
                        return f"{base_type}<{', '.join(type_args)}>"
            
            return base_type
        
        return type_str

    def process_java_file(self, file_path: str) -> Dict:
        """Обработка Java файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        tree = javalang.parse.parse(content)
        
        # Находим основной класс (первый публичный класс)
        main_class = None
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            if 'public' in node.modifiers:
                main_class = node
                break
        
        if not main_class:
            main_class = next(tree.filter(javalang.tree.ClassDeclaration))[1]
            
        self.class_info = {
            'name': main_class.name,
            'package': str(tree.package.name) if tree.package else '',
            'imports': [imp.path for imp in tree.imports] if tree.imports else [],
            'annotations': [ann.name for ann in main_class.annotations] if main_class.annotations else [],
            'generics': [param.name for param in main_class.type_parameters] if main_class.type_parameters else [],
            'interfaces': [impl.name for impl in main_class.implements] if main_class.implements else [],
            'modifiers': list(main_class.modifiers) if main_class.modifiers else [],
            'methods': [],
            'fields': [],
            'inner_classes': []
        }
        
        # Обработка методов
        for method in main_class.methods:
            method_info = {
                'name': method.name,
                'return_type': self._simplify_java_type(str(method.return_type)),
                'params': [(param.name, self._simplify_java_type(str(param.type))) for param in method.parameters] if method.parameters else [],
                'annotations': [ann.name for ann in method.annotations] if method.annotations else [],
                'modifiers': list(method.modifiers) if method.modifiers else [],
                'is_suspend': 'suspend' in method.modifiers
            }
            self.class_info['methods'].append(method_info)
            
        # Обработка полей
        for field in main_class.fields:
            for declarator in field.declarators:
                field_info = {
                    'name': declarator.name,
                    'type': self._simplify_java_type(str(field.type)),
                    'annotations': [ann.name for ann in field.annotations] if field.annotations else [],
                    'modifiers': list(field.modifiers) if field.modifiers else []
                }
                self.class_info['fields'].append(field_info)
                
        # Обработка внутренних классов
        for inner_class in main_class.body:
            if isinstance(inner_class, javalang.tree.ClassDeclaration):
                inner_class_info = {
                    'name': inner_class.name,
                    'description': f"Внутренний класс {inner_class.name}",
                    'fields': [],
                    'methods': []
                }
                
                # Обработка полей внутреннего класса
                for field in inner_class.fields:
                    for declarator in field.declarators:
                        field_info = {
                            'name': declarator.name,
                            'type': self._simplify_java_type(str(field.type))
                        }
                        inner_class_info['fields'].append(field_info)
                        
                # Обработка методов внутреннего класса
                for method in inner_class.methods:
                    method_info = {
                        'name': method.name,
                        'return_type': self._simplify_java_type(str(method.return_type)),
                        'params': [(param.name, self._simplify_java_type(str(param.type))) for param in method.parameters] if method.parameters else [],
                        'is_suspend': 'suspend' in method.modifiers
                    }
                    inner_class_info['methods'].append(method_info)
                    
                self.class_info['inner_classes'].append(inner_class_info)
                
        return self.class_info
        
    def process_kotlin_file(self, file_path: str) -> Dict:
        """Обработка Kotlin файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Регулярные выражения для извлечения информации
        package_pattern = r'package\s+([^\s;]+)'
        import_pattern = r'import\s+([^\s;]+)'
        annotation_pattern = r'@(\w+)(?:\s*\([^)]*\))?\s*(?=class)'
        class_pattern = r'(?:@\w+\s*)*(?:public\s+)?(?:internal\s+)?(?:private\s+)?(?:protected\s+)?(?:data\s+)?(?:sealed\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?'
        method_pattern = r'(?:@\w+\s*)*(?:public\s+)?(?:private\s+)?(?:protected\s+)?(?:suspend\s+)?fun\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)\s*(?::\s*([^=\n{]+))?'
        field_pattern = r'(?:private\s+)?(?:val|var)\s+(\w+)(?:\s*:\s*([^=\n{]+))?(?:\s*=\s*([^=\n]+))?'
        
        # Извлечение информации
        package_match = re.search(package_pattern, content)
        imports = re.findall(import_pattern, content)
        annotations = re.findall(annotation_pattern, content)
        class_match = re.search(class_pattern, content)
        
        if not class_match:
            raise ValueError("Не найден основной класс в Kotlin файле")
        
        class_name = class_match.group(1)
        class_generics = class_match.group(2).split(',') if class_match.group(2) else []
        class_inheritance = class_match.group(3) if class_match.group(3) else None
        
        self.class_info = {
            'name': class_name,
            'package': package_match.group(1) if package_match else '',
            'imports': imports,
            'annotations': annotations,
            'generics': [g.strip() for g in class_generics],
            'interfaces': [iface.strip() for iface in class_inheritance.split(',')] if class_inheritance else [],
            'methods': [],
            'fields': [],
            'inner_classes': []
        }
        
        # Обработка полей
        class_body_start = content.find('{', content.find(class_name)) + 1
        class_body_end = content.rfind('}')
        class_body = content[class_body_start:class_body_end]
        
        for field_match in re.finditer(field_pattern, class_body):
            field_name = field_match.group(1)
            field_type = field_match.group(2)  # Явно указанный тип
            field_init = field_match.group(3)  # Инициализация
            
            # Если тип явно указан, используем его
            if field_type:
                field_type = field_type.strip()
            # Если есть инициализация, пытаемся определить тип из нее
            elif field_init:
                field_init = field_init.strip()
                if 'ConcurrentHashMap' in field_init:
                    # Ищем generic параметры
                    type_params = re.search(r'ConcurrentHashMap<([^>]+)>', field_init)
                    field_type = f"ConcurrentHashMap<{type_params.group(1)}>" if type_params else "ConcurrentHashMap"
                else:
                    # Для других типов коллекций
                    collection_match = re.search(r'(\w+)<([^>]+)>', field_init)
                    if collection_match:
                        field_type = f"{collection_match.group(1)}<{collection_match.group(2)}>"
                    else:
                        field_type = 'Any'
            else:
                field_type = 'Any'
            
            self.class_info['fields'].append({
                'name': field_name,
                'type': field_type
            })
        
        # Обработка методов
        for method_match in re.finditer(method_pattern, content):
            method_name = method_match.group(1)
            params_str = method_match.group(2)
            return_type = method_match.group(3).strip() if method_match.group(3) else 'Unit'
            
            # Проверяем, является ли метод suspend функцией
            is_suspend = bool(re.search(r'suspend\s+fun', method_match.group(0)))
            
            params = []
            if params_str:
                param_pairs = [p.strip() for p in params_str.split(',')]
                for pair in param_pairs:
                    if ':' in pair:
                        name, type_str = pair.split(':', 1)
                        params.append((name.strip(), type_str.strip()))

            method_info = {
                'name': method_name,
                'params': params,
                'return_type': return_type,
                'is_suspend': is_suspend
            }
            self.class_info['methods'].append(method_info)

        return self.class_info
        
    def extract_relationships(self, class_info: Dict) -> List[str]:
        """Извлечение связей между классами"""
        relationships = []
        
        # Анализ импортов
        for imp in class_info.get('imports', []):
            if 'KotlinAnalyzer' in imp:
                relationships.append(f"Использует Kotlin класс KotlinAnalyzer")
            elif 'coroutines' in imp:
                relationships.append(f"Использует корутины Kotlin ({imp})")
            elif 'concurrent.ConcurrentHashMap' in imp:
                relationships.append(f"Использует конкурентный тип ConcurrentHashMap")
            elif 'concurrent' in imp:
                relationships.append(f"Использует конкурентный тип {imp}")
            elif any(type_name in imp for type_name in ['Map', 'List', 'Set', 'Collection']):
                relationships.append(f"Использует коллекцию {imp}")
            elif '.kt' in imp or 'kotlin.' in imp:
                relationships.append(f"Использует Kotlin тип {imp}")
            
        # Анализ интерфейсов
        for interface in class_info.get('interfaces', []):
            relationships.append(f"Реализует интерфейс {interface}")
            
        # Анализ полей
        for field in class_info.get('fields', []):
            field_type = field.get('type', '')
            if 'KotlinAnalyzer' in field_type:
                relationships.append(f"Содержит поле типа KotlinAnalyzer: {field['name']}")
            elif 'ConcurrentHashMap' in field_type:
                relationships.append(f"Содержит поле типа ConcurrentHashMap: {field['name']}")
            elif 'concurrent' in field_type.lower():
                relationships.append(f"Содержит конкурентный тип в поле {field['name']}: {field_type}")
            elif any(collection in field_type for collection in ['Map', 'List', 'Set']):
                relationships.append(f"Содержит коллекцию в поле {field['name']}: {field_type}")
            elif 'Flow' in field_type:
                relationships.append(f"Использует Kotlin Flow в поле {field['name']}: {field_type}")
            
        # Анализ параметров методов
        for method in class_info.get('methods', []):
            # Проверяем suspend функции
            if method.get('is_suspend', False):
                relationships.append(f"Метод {method['name']} является корутиной (suspend)")
            
            for param_name, param_type in method.get('params', []):
                if 'Flow' in param_type:
                    relationships.append(f"Использует Kotlin Flow в параметре {param_name} метода {method['name']}")
                elif 'ConcurrentHashMap' in param_type:
                    relationships.append(f"Использует ConcurrentHashMap в параметре {param_name} метода {method['name']}")
                elif 'concurrent' in param_type.lower():
                    relationships.append(f"Использует конкурентный тип в параметре {param_name} метода {method['name']}")
                elif any(collection in param_type for collection in ['Map', 'List', 'Set']):
                    relationships.append(f"Использует коллекцию в параметре {param_name} метода {method['name']}")
            
        # Анализ возвращаемых типов
        for method in class_info.get('methods', []):
            return_type = method.get('return_type', '')
            if 'Flow' in return_type:
                relationships.append(f"Метод {method['name']} возвращает Kotlin Flow")
            elif 'ConcurrentHashMap' in return_type:
                relationships.append(f"Метод {method['name']} возвращает ConcurrentHashMap")
            elif 'concurrent' in return_type.lower():
                relationships.append(f"Метод {method['name']} возвращает конкурентный тип")
            elif any(collection in return_type for collection in ['Map', 'List', 'Set']):
                relationships.append(f"Метод {method['name']} возвращает коллекцию {return_type}")
            
        return list(set(relationships))  # Убираем дубликаты
        
    def generate_documentation(self, class_info: Optional[Dict] = None) -> str:
        """Генерация документации на основе проанализированной информации"""
        if class_info is None:
            class_info = self.class_info
        
        doc = []
        
        # Заголовок
        doc.append(f"# Класс {class_info['name']}")
        doc.append("")
        
        # Пакет
        if class_info['package']:
            doc.append(f"**Пакет:** `{class_info['package']}`")
            doc.append("")
        
        # Импорты
        if class_info['imports']:
            doc.append("## Импорты")
            for imp in class_info['imports']:
                doc.append(f"- `{imp}`")
            doc.append("")
        
        # Зависимости
        relationships = self.extract_relationships(class_info)
        if relationships:
            doc.append("## Зависимости")
            for rel in relationships:
                doc.append(f"- {rel}")
            doc.append("")
        
        # Аннотации
        if 'annotations' in class_info and class_info['annotations']:
            doc.append("## Аннотации")
            for ann in class_info['annotations']:
                doc.append(f"- @{ann}")
            doc.append("")
        
        # Дженерики
        if 'generics' in class_info and class_info['generics']:
            doc.append("## Параметры типа")
            for generic in class_info['generics']:
                doc.append(f"- `{generic}`")
            doc.append("")
        
        # Интерфейсы
        if 'interfaces' in class_info and class_info['interfaces']:
            doc.append("## Реализуемые интерфейсы")
            for interface in class_info['interfaces']:
                doc.append(f"- `{interface}`")
            doc.append("")
        
        # Поля
        if class_info['fields']:
            doc.append("## Поля")
            for field in class_info['fields']:
                field_doc = [f"### {field['name']}"]
                field_doc.append(f"- **Тип:** `{field['type']}`")
                if 'annotations' in field and field['annotations']:
                    field_doc.append("- **Аннотации:**")
                    for ann in field['annotations']:
                        field_doc.append(f"  - @{ann}")
                doc.extend(field_doc)
                doc.append("")
        
        # Методы
        if class_info['methods']:
            doc.append("## Методы")
            for method in class_info['methods']:
                method_doc = []
                
                # Добавляем suspend для корутин
                if method.get('is_suspend', False):
                    method_doc.append(f"### suspend fun {method['name']}")
                else:
                    method_doc.append(f"### {method['name']}")
                
                # Параметры метода
                if method['params']:
                    method_doc.append("**Параметры:**")
                    for param_name, param_type in method['params']:
                        method_doc.append(f"- `{param_name}`: `{param_type}`")
                
                # Тип возврата
                if 'return_type' in method:
                    method_doc.append(f"**Возвращает:** `{method['return_type']}`")
                
                # Аннотации метода
                if 'annotations' in method and method['annotations']:
                    method_doc.append("**Аннотации:**")
                    for ann in method['annotations']:
                        method_doc.append(f"- @{ann}")
                
                doc.extend(method_doc)
                doc.append("")
        
        # Внутренние классы
        if 'inner_classes' in class_info and class_info['inner_classes']:
            doc.append("## Внутренние классы")
            for inner_class in class_info['inner_classes']:
                doc.append(f"### {inner_class['name']}")
                doc.append(inner_class.get('description', ''))
                doc.append("")
                
                # Поля внутреннего класса
                if inner_class['fields']:
                    doc.append("#### Поля")
                    for field in inner_class['fields']:
                        doc.append(f"- `{field['name']}`: `{field['type']}`")
                    doc.append("")
                
                # Методы внутреннего класса
                if inner_class['methods']:
                    doc.append("#### Методы")
                    for method in inner_class['methods']:
                        method_signature = f"- `{method['name']}("
                        if method['params']:
                            params = [f"{name}: {type_}" for name, type_ in method['params']]
                            method_signature += ", ".join(params)
                        method_signature += f"): {method['return_type']}`"
                        doc.append(method_signature)
                    doc.append("")
        
        return "\n".join(doc)

    def generate_ast_report(self, output_file: str = None) -> str:
        """
        Генерирует подробный отчет о структуре AST
        
        Args:
            output_file (str, optional): Путь для сохранения отчета. 
                                       Если не указан, возвращает отчет как строку.
        
        Returns:
            str: Отчет в формате Markdown
        """
        report = []
        report.append("# AST Analysis Report\n")
        
        # Основная информация о классе
        report.append("## Class Information")
        report.append(f"- Name: `{self.class_info['name']}`")
        report.append(f"- Package: `{self.class_info['package']}`")
        report.append("")
        
        # Структура импортов
        report.append("## Import Structure")
        for imp in self.class_info['imports']:
            report.append(f"- `{imp}`")
        report.append("")
        
        # Иерархия классов
        report.append("## Class Hierarchy")
        if self.class_info.get('interfaces'):
            report.append("### Implemented Interfaces")
            for interface in self.class_info['interfaces']:
                report.append(f"- `{interface}`")
        report.append("")
        
        # Структура полей
        report.append("## Field Structure")
        for field in self.class_info['fields']:
            report.append(f"### {field['name']}")
            report.append(f"- Type: `{field['type']}`")
            if 'annotations' in field:
                report.append("- Annotations:")
                for ann in field['annotations']:
                    report.append(f"  - `@{ann}`")
            report.append("")
        
        # Структура методов
        report.append("## Method Structure")
        for method in self.class_info['methods']:
            report.append(f"### {method['name']}")
            # Параметры
            if method['params']:
                report.append("#### Parameters:")
                for param_name, param_type in method['params']:
                    report.append(f"- `{param_name}`: `{param_type}`")
            # Тип возврата
            report.append(f"#### Return Type: `{method['return_type']}`")
            # Дополнительные свойства
            if method.get('is_suspend'):
                report.append("- Is Suspend Function: Yes")
            report.append("")
        
        # Внутренние классы
        if self.class_info.get('inner_classes'):
            report.append("## Inner Classes")
            for inner in self.class_info['inner_classes']:
                report.append(f"### {inner['name']}")
                report.append(inner.get('description', ''))
                report.append("")
        
        # Связи между компонентами
        relationships = self.extract_relationships(self.class_info)
        if relationships:
            report.append("## Component Relationships")
            for rel in relationships:
                report.append(f"- {rel}")
            report.append("")
        
        final_report = "\n".join(report)
        
        # Сохраняем в файл, если указан путь
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
        
        return final_report

    def prepare_llm_data(self) -> Dict:
        """
        Подготавливает данные для отправки в LLM в структурированном формате
        
        Returns:
            Dict: Структурированные данные для анализа LLM
        """
        return {
            "file_info": {
                "class_name": self.class_info['name'],
                "package": self.class_info['package'],
                "type": "Kotlin" if any('kotlin.' in imp or '.kt' in imp for imp in self.class_info['imports']) else "Java"
            },
            "structure": {
                "imports": self.class_info['imports'],
                "interfaces": self.class_info.get('interfaces', []),
                "fields": [
                    {
                        "name": field['name'],
                        "type": field['type'],
                        "annotations": field.get('annotations', [])
                    }
                    for field in self.class_info['fields']
                ],
                "methods": [
                    {
                        "name": method['name'],
                        "parameters": method['params'],
                        "return_type": method['return_type'],
                        "is_suspend": method.get('is_suspend', False)
                    }
                    for method in self.class_info['methods']
                ]
            },
            "relationships": self.extract_relationships(self.class_info),
            "inner_classes": [
                {
                    "name": inner['name'],
                    "description": inner.get('description', ''),
                    "methods": [
                        {
                            "name": method['name'],
                            "parameters": method['params'],
                            "return_type": method['return_type']
                        }
                        for method in inner.get('methods', [])
                    ]
                }
                for inner in self.class_info.get('inner_classes', [])
            ]
        } 