import re
from typing import Dict, List
import logging


class DocAnalyzer:
    """
    Анализатор документированных файлов.
    Извлекает структурированную информацию из файлов с KDoc/JavaDoc документацией.
    """
    
    def __init__(self):
        self.doc_patterns = {
            'class': r'\/\*\*\s*(.*?)\s*\*\/\s*(?:public\s+)?(?:class|interface)\s+(\w+)',
            'method': r'\/\*\*\s*(.*?)\s*\*\/\s*(?:public|private|protected)?\s+.*?(\w+)\s*\(',
            'field': r'\/\*\*\s*(.*?)\s*\*\/\s*(?:public|private|protected)?\s+.*?(\w+)\s*[=;]'
        }
        
    def analyze_file(self, content: str, file_type: str = 'kotlin') -> Dict:
        """
        Анализирует документированный файл и извлекает структурированную информацию.
        
        Args:
            content (str): Содержимое документированного файла
            file_type (str): Тип файла ('kotlin' или 'java')
            
        Returns:
            dict: Структурированная информация о коде
        """
        try:
            # Извлекаем информацию о пакете и импортах
            package = self._extract_package(content)
            imports = self._extract_imports(content)
            
            # Анализируем классы
            classes = self._analyze_classes(content)
            
            return {
                'file_info': {
                    'package': package,
                    'imports': imports
                },
                'classes': classes
            }
            
        except Exception as e:
            logging.error(f"Ошибка при анализе файла: {str(e)}")
            return self._create_error_response(str(e))
            
    def _extract_package(self, content: str) -> str:
        """Извлекает информацию о пакете."""
        match = re.search(r'package\s+([\w.]+)', content)
        return match.group(1) if match else ""
        
    def _extract_imports(self, content: str) -> List[str]:
        """Извлекает список импортов."""
        return re.findall(r'import\s+([\w.]+(?:\s*\*)?)', content)
        
    def _analyze_classes(self, content: str) -> List[Dict]:
        """Анализирует классы в файле."""
        classes = []
        
        # Ищем все классы
        class_matches = re.finditer(self.doc_patterns['class'], content, re.DOTALL)
        for match in class_matches:
            doc, name = match.groups()
            
            # Извлекаем информацию о классе
            class_info = {
                'name': name,
                'documentation': self._parse_doc_comment(doc),
                'methods': self._analyze_methods(content, name),
                'fields': self._analyze_fields(content, name)
            }
            
            classes.append(class_info)
            
        return classes
        
    def _analyze_methods(self, content: str, class_name: str) -> List[Dict]:
        """Анализирует методы класса."""
        methods = []
        
        # Ищем все методы
        method_matches = re.finditer(self.doc_patterns['method'], content, re.DOTALL)
        for match in method_matches:
            doc, name = match.groups()
            
            method_info = {
                'name': name,
                'documentation': self._parse_doc_comment(doc)
            }
            
            methods.append(method_info)
            
        return methods
        
    def _analyze_fields(self, content: str, class_name: str) -> List[Dict]:
        """Анализирует поля класса."""
        fields = []
        
        # Ищем все документированные поля
        field_matches = re.finditer(self.doc_patterns['field'], content, re.DOTALL)
        for match in field_matches:
            doc, name = match.groups()
            
            field_info = {
                'name': name,
                'documentation': self._parse_doc_comment(doc)
            }
            
            fields.append(field_info)
            
        return fields
        
    def _parse_doc_comment(self, doc: str) -> Dict:
        """Разбирает документационный комментарий."""
        # Очищаем комментарий
        doc = re.sub(r'\s*\*\s*', ' ', doc).strip()
        
        # Извлекаем основные части
        description = self._extract_description(doc)
        params = self._extract_params(doc)
        returns = self._extract_returns(doc)
        throws = self._extract_throws(doc)
        
        return {
            'description': description,
            'params': params,
            'returns': returns,
            'throws': throws
        }
        
    def _extract_description(self, doc: str) -> str:
        """Извлекает основное описание."""
        match = re.search(r'^(.*?)(?=@|\Z)', doc, re.DOTALL)
        return match.group(1).strip() if match else ""
        
    def _extract_params(self, doc: str) -> List[Dict]:
        """Извлекает информацию о параметрах."""
        params = []
        param_matches = re.finditer(r'@param\s+(\w+)\s+(.+?)(?=@|\Z)', doc)
        for match in param_matches:
            name, desc = match.groups()
            params.append({
                'name': name,
                'description': desc.strip()
            })
        return params
        
    def _extract_returns(self, doc: str) -> str:
        """Извлекает информацию о возвращаемом значении."""
        match = re.search(r'@return\s+(.+?)(?=@|\Z)', doc)
        return match.group(1).strip() if match else ""
        
    def _extract_throws(self, doc: str) -> List[Dict]:
        """Извлекает информацию об исключениях."""
        throws = []
        throw_matches = re.finditer(r'@throws\s+(\w+)\s+(.+?)(?=@|\Z)', doc)
        for match in throw_matches:
            type_name, desc = match.groups()
            throws.append({
                'type': type_name,
                'description': desc.strip()
            })
        return throws
        
    def _create_error_response(self, error_message: str) -> Dict:
        """Создает ответ с ошибкой."""
        return {
            'error': error_message,
            'file_info': {
                'package': '',
                'imports': []
            },
            'classes': []
        } 