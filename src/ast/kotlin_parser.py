from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KotlinASTParser:
    """Парсер AST для Kotlin кода с использованием tree-sitter."""
    
    def __init__(self):
        """Инициализация парсера."""
        try:
            # Получаем язык Kotlin из предустановленных грамматик
            self.kotlin_language = get_language('kotlin')
            self.parser = Parser()
            self.parser.set_language(self.kotlin_language)
            logger.info("Kotlin parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kotlin parser: {str(e)}")
            raise

    def parse_code(self, code: str) -> dict:
        """
        Парсит Kotlin код и возвращает AST.
        
        Args:
            code (str): Исходный код на Kotlin
            
        Returns:
            dict: Структурированное представление кода
        """
        try:
            # Парсим код в AST
            tree = self.parser.parse(bytes(code, "utf8"))
            
            # Получаем корневой узел
            root_node = tree.root_node
            
            # Базовая информация о файле
            result = {
                "imports": self._extract_imports(root_node),
                "package": self._extract_package(root_node),
                "classes": self._extract_classes(root_node)
            }
            
            return result
        except Exception as e:
            logger.error(f"Failed to parse Kotlin code: {str(e)}")
            raise

    def _extract_imports(self, root_node) -> list:
        """Извлекает импорты из AST."""
        imports = []
        for node in root_node.children:
            if node.type == "import_header":
                import_text = node.text.decode('utf-8')
                imports.append(import_text.strip())
        return imports

    def _extract_package(self, root_node) -> str:
        """Извлекает имя пакета из AST."""
        for node in root_node.children:
            if node.type == "package_header":
                return node.text.decode('utf-8').replace("package", "").strip()
        return ""

    def _extract_classes(self, root_node) -> list:
        """Извлекает информацию о классах из AST."""
        classes = []
        for node in root_node.children:
            if node.type == "class_declaration":
                class_info = self._extract_class_info(node)
                classes.append(class_info)
        return classes

    def _extract_class_info(self, class_node) -> dict:
        """Извлекает детальную информацию о классе."""
        class_info = {
            "name": "",
            "superclass": None,
            "interfaces": [],
            "annotations": [],
            "modifiers": []
        }

        # Извлекаем имя класса
        type_identifier = next(
            (child for child in class_node.children if child.type == "type_identifier"),
            None
        )
        if type_identifier:
            class_info["name"] = type_identifier.text.decode('utf-8')

        # Извлекаем модификаторы и аннотации
        for child in class_node.children:
            if child.type == "modifiers":
                for modifier in child.children:
                    if modifier.type == "annotation":
                        class_info["annotations"].append(modifier.text.decode('utf-8'))
                    else:
                        class_info["modifiers"].append(modifier.text.decode('utf-8'))

        return class_info

if __name__ == "__main__":
    # Пример использования
    parser = KotlinASTParser()
    test_code = """
    package com.example.app

    import android.os.Bundle
    import androidx.fragment.app.Fragment
    import dagger.hilt.android.AndroidEntryPoint

    @AndroidEntryPoint
    class HomeFragment : Fragment(), OnSearchListener {
        // Class content
    }
    """
    
    try:
        result = parser.parse_code(test_code)
        print("Parsed result:", result)
    except Exception as e:
        print(f"Error parsing code: {e}") 