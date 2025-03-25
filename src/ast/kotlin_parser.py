import logging
import os

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

LOG_FILE = os.path.abspath("kotlin_parser.log")


def setup_logging():
    """Настройка логгера для тестов"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s '
        '[%(filename)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s → %(filename)s:%(lineno)d'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


# Настройка логгирования перед тестами
setup_logging()


class KotlinASTParser:
    """Парсер AST для Kotlin кода с использованием tree-sitter."""

    def __init__(self):
        """Инициализация парсера."""
        try:
            # Загрузка предкомпилированного языка
            self.parser = get_parser('kotlin')
            print("Parser initialized successfully")
        except Exception as e:
            print(f"Error: {e}")
            raise

    def parse_code(self, code: str) -> dict:
        """
        Парсит Kotlin код и возвращает структурированное представление

        Args:
            code: Исходный код на Kotlin

        Returns:
            Словарь с package, import, классами и их структурой
        """
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            root_node = tree.root_node

            def print_ast(node, level=0):
                print('  ' * level + f"{node.type}: {node.text.decode('utf-8')}")
                for child in node.children:
                    print_ast(child, level + 1)

            # print_ast(root_node)
            return {
                "package": self._extract_package(root_node),
                "imports": self._extract_imports(root_node),
                "declarations": self._extract_declarations(root_node)
            }
        except Exception as e:
            logging.error(f"Parsing failed: {e}")
            raise

    def _extract_package(self, root_node: Node) -> str:
        """Извлекает package из корневого узла"""
        package_node = self._find_child_by_type(root_node, "package_header")
        if package_node:
            return package_node.text.decode('utf-8').replace("package", "").strip()
        return ""

    def _extract_imports(self, root_node: Node) -> list:
        imports = []
        for node in root_node.children:
            if node.type == "import_list":
                for child in node.children:
                    if child.type == "import_header":
                        imports.append(child.text.decode('utf-8').strip())
        return imports

    def _extract_declarations(self, root_node: Node) -> list:
        """Извлекает все объявления (классы, объекты, интерфейсы)"""
        declarations = []
        for node in root_node.children:
            if node.type == "class_declaration":
                declarations.append(self._parse_class(node))
            elif node.type == "object_declaration":
                declarations.append(self._parse_object(node))
            elif node.type == "interface_declaration":
                declarations.append(self._parse_interface(node))
        return declarations

    def _parse_class(self, class_node: Node) -> dict:
        superclass = None
        implements = []
        for child in class_node.children:
            if child.type == "delegation_specifier":
                text = child.text.decode('utf-8')
                if '(' in text:
                    superclass = text.split('(')[0]
                else:
                    implements.append(text)

        return {
            "type": "class",
            "name": self._get_identifier(class_node),
            "annotations": self._extract_annotations(class_node),
            "modifiers": self._extract_modifiers(class_node),
            "superclass": superclass,
            "implements": implements,
            "body": self._extract_class_body(class_node)
        }

    def _extract_functions(self, class_node: Node) -> list:
        functions = []
        for node in self._traverse(class_node, 'function_declaration'):
            name_node = self._find_child_by_field(node, 'name')
            functions.append(name_node.text.decode() if name_node else '')
        return functions

    def _extract_properties(self, class_node: Node) -> list:
        properties = []
        for node in self._traverse(class_node, 'property_declaration'):
            name_node = self._find_child_by_field(node, 'name')
            properties.append(name_node.text.decode() if name_node else '')
        return properties

    def _traverse(self, node: Node, node_type: str) -> list:
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from self._traverse(child, node_type)

    def _find_child_by_field(self, node: Node, field: str) -> Node:
        return node.child_by_field_name(field)

    def _parse_object(self, object_node: Node) -> dict:
        """Парсит объявление объекта"""
        return {
            "type": "object",
            "name": self._get_identifier(object_node),
            "annotations": self._extract_annotations(object_node),
            "modifiers": self._extract_modifiers(object_node),
            "body": self._extract_class_body(object_node)
        }

    def _parse_interface(self, interface_node: Node) -> dict:
        """Парсит объявление интерфейса"""
        return {
            "type": "interface",
            "name": self._get_identifier(interface_node),
            "annotations": self._extract_annotations(interface_node),
            "modifiers": self._extract_modifiers(interface_node),
            "body": self._extract_class_body(interface_node)
        }

    def _extract_annotations(self, node: Node) -> list:
        annotations = []
        modifiers_node = self._find_child_by_type(node, "modifiers")
        if modifiers_node:
            for child in modifiers_node.children:
                if child.type == "annotation":
                    annotations.append(child.text.decode('utf-8'))
        return annotations

    def _extract_modifiers(self, node: Node) -> list:
        modifiers = []
        modifiers_node = self._find_child_by_type(node, "modifiers")
        if modifiers_node:
            for mod in modifiers_node.children:
                if mod.type in ["modifier", "visibility_modifier"]:
                    modifiers.append(mod.text.decode('utf-8'))
        return modifiers

    def _extract_implements(self, class_node: Node) -> list:
        """Извлекает реализуемые интерфейсы"""
        implements_clause = self._find_child_by_type(class_node, "implements_clause")
        if implements_clause:
            return [
                n.text.decode('utf-8')
                for n in self._find_children_by_type(implements_clause, "type_reference")
            ]
        return []

    def _extract_class_body(self, class_node: Node) -> dict:
        """Учитывает Kotlin-синтаксис (class_body)"""
        body = {
            "properties": [],
            "functions": []
        }
        class_body = self._find_child_by_type(class_node, "class_body")
        if not class_body:
            return body

        for node in class_body.children:
            if node.type == "property_declaration":
                body["properties"].append(self._parse_property(node))
            elif node.type == "function_declaration":
                body["functions"].append(self._parse_function(node))
        return body

    def _parse_property(self, prop_node: Node) -> dict:
        modifiers = []
        annotations = []
        name = type_ = delegate = ""

        # Аннотации
        annotations = self._extract_annotations(prop_node)

        # Модификаторы (visibility + другие, например, open)
        modifiers_node = self._find_child_by_type(prop_node, "modifiers")
        if modifiers_node:
            modifiers = [mod.text.decode('utf-8') for mod in modifiers_node.children
                         if mod.type in ["visibility_modifier", "modifier"]]

        # Имя и тип
        var_decl = self._find_child_by_type(prop_node, "variable_declaration")
        if var_decl:
            name_node = self._find_child_by_type(var_decl, "simple_identifier")
            type_node = self._find_child_by_type(var_decl, "type_reference") or \
                        self._find_child_by_type(var_decl, "user_type")
            name = name_node.text.decode('utf-8') if name_node else ""
            type_ = type_node.text.decode('utf-8') if type_node else ""

        # Делегат
        delegate_node = self._find_child_by_type(prop_node, "property_delegate")
        if delegate_node:
            delegate = delegate_node.text.decode('utf-8')

        return {
            "name": name,
            "type": type_,
            "delegate": delegate,
            "modifiers": modifiers,
            "annotations": annotations
        }

    def _parse_function(self, func_node: Node) -> dict:
        name = ""
        parameters = []
        return_type = ""
        annotations = []
        modifiers = []

        # Аннотации
        annotations = self._extract_annotations(func_node)

        # Модификаторы
        modifiers = self._extract_modifiers(func_node)

        # Имя функции
        name_node = self._find_child_by_type(func_node, "simple_identifier")
        name = name_node.text.decode('utf-8') if name_node else ""

        # Параметры (с поддержкой default_value)
        params_node = self._find_child_by_type(func_node, "function_value_parameters")
        if params_node:
            for param in params_node.children:
                if param.type == "parameter":
                    param_name = self._find_child_by_type(param, "simple_identifier").text.decode('utf-8')
                    param_type = self._find_child_by_type(param, "type_reference").text.decode('utf-8') if \
                                self._find_child_by_type(param, "type_reference") else ""
                    default_node = self._find_child_by_type(param, "default_value")
                    default_value = default_node.text.decode('utf-8') if default_node else None
                    parameters.append({
                        "name": param_name,
                        "type": param_type,
                        "default_value": default_value
                    })

                    # Возвращаемый тип
                    return_type_node = self._find_child_by_type(func_node, "type_reference") or \
                                       self._find_child_by_type(func_node, "user_type")
                    return_type = return_type_node.text.decode('utf-8') if return_type_node else "Unit"

        return {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "annotations": annotations,
            "modifiers": modifiers
        }

    def _extract_parameters(self, func_node: Node) -> list:
        param_list = func_node.child_by_field_name("parameters")
        if not param_list:
            return []

        return [
            {
                "name": p.child_by_field_name("name").text.decode(),
                "type": p.child_by_field_name("type").text.decode()
            }
            for p in param_list.children
            if p.type == "parameter"
        ]

    def _parse_parameter(self, param_node: Node) -> dict:
        """Парсит параметр функции"""
        return {
            "name": self._get_identifier(param_node),
            "type": self._find_child_by_type(param_node, "type_reference"),
            "defaultValue": self._find_child_by_type(param_node, "expression")
        }

    def _get_identifier(self, node: Node) -> str:
        """Корректно извлекает имя для Kotlin (использует type_identifier)"""
        identifier = self._find_child_by_type(node, "type_identifier")  # Для классов
        if not identifier:
            identifier = self._find_child_by_type(node, "simple_identifier")  # Для методов/свойств
        return identifier.text.decode('utf-8') if identifier else ""

    def _find_child_by_type(self, node: Node, type_name: str) -> Node:
        """Ищет первого потомка с указанным типом"""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_children_by_type(self, node: Node, type_name: str) -> list:
        """Ищет всех потомков с указанным типом"""
        return [child for child in node.children if child.type == type_name]


if __name__ == "__main__":
    parser = KotlinASTParser()
    test_code = """
package ru.psbank.msb.landingsbuilder.ui.builder.mappers.text

import android.content.Context
import android.view.View
import android.widget.LinearLayout
import android.widget.LinearLayout.LayoutParams.MATCH_PARENT
import android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
import android.widget.TextView
import ru.psbank.msb.extensions.dp2px
import ru.psbank.msb.landingsbuilder.api.model.components.PageComponent
import ru.psbank.msb.landingsbuilder.api.model.components.text.Header1PageComponent
import ru.psbank.msb.landingsbuilder.ui.builder.mappers.MappersMargins.HORIZONTAL_MARGIN_DP
import ru.psbank.msb.landingsbuilder.ui.builder.mappers.MappersMargins.NO_MARGIN_DP
import ru.psbank.msb.landingsbuilder.ui.builder.mappers.base.PageComponentMapper

internal class Header1Mapper(private val context: Context) : PageComponentMapper(context) {
    override fun map(component: PageComponent): View {
        return if (component is Header1PageComponent) {
            TextView(context).apply {
                tag = component

                layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply {
                    setMargins(
                        dp2px(HORIZONTAL_MARGIN_DP).toInt(),
                        dp2px(NO_MARGIN_DP).toInt(),
                        dp2px(HORIZONTAL_MARGIN_DP).toInt(),
                        dp2px(NO_MARGIN_DP).toInt()
                    )
                }

                setTextAppearance(ru.psbank.msb.uicomponents.R.style.Title1_1)

                text = component.text
            }
        } else {
            super.map(component)
        }
    }
}

"""

    try:
        result = parser.parse_code(test_code)
        print("Parsed result:")
        print(f"Package: {result['package']}")
        print(f"Imports: {', '.join(result['imports'])}")

        if result['declarations']:
            declaration = result['declarations'][0]
        else:
            declaration = {}

        prompt = (
            f"""
            Сгенерируйте документацию для класса {declaration.get('name', 'Unknown')}.

            Контекст:
            - Пакет: {result.get('package', 'Unknown')}
            - Импорты: {', '.join(result.get('imports', []))}
            - Наследование: {declaration.get('superclass', 'None')}
            - Реализует: {', '.join(declaration.get('implements', []))}
            """
        )

        # Свойства
        properties = []
        for p in declaration.get('body', {}).get('properties', []):
            properties.append(
                f"- {p.get('name', 'N/A')} ({p.get('type', 'N/A')})" +
                f" Модификаторы: {', '.join(p.get('modifiers', []))}" +
                f" Делегат: {p.get('delegate', 'None')}"
            )
        prompt += r"""
            Свойства:
            """ + '\n'.join(properties)

        # Методы
        functions = []
        for f in declaration.get('body', {}).get('functions', []):
            params = ', '.join(
                f"{param.get('name', 'N/A')}: {param.get('type', 'N/A')}"
                for param in f.get('parameters', [])
            )
            functions.append(
                f"- {f.get('name', 'N/A')}({params}) → {f.get('return_type', 'Void')}"
            )
        prompt += r"""
            Методы:
            """ + '\n'.join(properties)

        print(prompt)  # Для отладки
        for decl in result['declarations']:
            print(f"- {decl['type'].capitalize()}: {decl['name']}")
            print(f"  Annotations: {decl['annotations']}")
            print(f"  Modifiers: {decl['modifiers']}")
            if decl['type'] == 'class':
                print(f"  Superclass: {decl.get('superclass', 'None')}")
                print(f"  Implements: {decl.get('implements', [])}")

                # Печать свойств с деталями
                print("  Properties:")
                for prop in decl['body']['properties']:
                    print(f"    - Name: {prop['name']}")
                    print(f"      Type: {prop['type']}")
                    print(f"      Delegate: {prop['delegate']}")
                    print(f"      Modifiers: {prop['modifiers']}")
                    print(f"      Annotations: {prop['annotations']}")

                # Печать методов с параметрами
                print("  Functions:")
                for func in decl['body']['functions']:
                    print(f"    - Name: {func['name']}")
                    print(f"      Return Type: {func['return_type']}")
                    print(f"      Modifiers: {func['modifiers']}")
                    print(f"      Annotations: {func['annotations']}")
                    print("      Parameters:")
                    for param in func['parameters']:
                        print(f"        - Name: {param['name']}")
                        print(f"          Type: {param['type']}")
                        print(f"          Default Value: {param.get('default_value', 'None')}")
    except Exception as e:
        print(f"Error parsing code: {e}")
