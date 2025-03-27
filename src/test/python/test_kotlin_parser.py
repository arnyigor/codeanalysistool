# import pytest
# from code_ast.kotlin_parser import KotlinASTParser
#
# @pytest.fixture
# def parser():
#     return KotlinASTParser()
#
# def test_parser_initialization(parser):
#     """Проверяем, что парсер инициализируется корректно."""
#     assert parser is not None
#     assert parser.parser is not None
#
# def test_parse_simple_class(parser):
#     """Тест парсинга простого класса."""
#     code = """
#     package com.example.app
#
#     import android.os.Bundle
#     import androidx.fragment.app.Fragment
#
#     class SimpleClass {
#         // Empty class
#     }
#     """
#
#     result = parser.parse_code(code)
#
#     # Проверяем базовую структуру
#     assert "imports" in result
#     assert "package" in result
#     assert "declarations" in result
#
#     # Проверяем пакет
#     assert result["package"] == "com.example.app"
#
#     # Проверяем импорты
#     assert len(result["imports"]) == 2
#     assert "import android.os.Bundle" in result["imports"]
#     assert "import androidx.fragment.app.Fragment" in result["imports"]
#
#     # Проверяем информацию о классе
#     assert len(result["declarations"]) == 1
#     class_info = result["declarations"][0]
#     assert class_info["name"] == "SimpleClass"
#     assert class_info["type"] == "class"
#
# def test_parse_annotated_class(parser):
#     """Тест парсинга класса с аннотациями."""
#     code = """
#     package com.example.app
#
#     import dagger.hilt.android.AndroidEntryPoint
#
#     @AndroidEntryPoint
#     class AnnotatedClass {
#         // Empty class
#     }
#     """
#
#     result = parser.parse_code(code)
#
#     # Проверяем аннотации
#     class_info = result["declarations"][0]
#     assert "@AndroidEntryPoint" in class_info["annotations"]
#     assert class_info["type"] == "class"
#
# def test_parse_class_with_inheritance(parser):
#     """Тест парсинга класса с наследованием."""
#     code = """
#     package com.example.app
#
#     import androidx.fragment.app.Fragment
#
#     class InheritedClass : Fragment(), OnSearchListener {
#         // Empty class
#     }
#     """
#
#     result = parser.parse_code(code)
#
#     # Проверяем наследование
#     class_info = result["declarations"][0]
#     assert class_info["name"] == "InheritedClass"
#     assert class_info["type"] == "class"
#     assert class_info["superclass"] == "Fragment"
#     assert "OnSearchListener" in class_info["implements"]