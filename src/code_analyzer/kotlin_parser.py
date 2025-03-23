from lark import Lark, Tree, Token, UnexpectedToken
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class KotlinParser:
    """Parser for Kotlin source code using Lark grammar"""
    
    def __init__(self):
        self.grammar = self._load_grammar()
        self.parser = Lark(self.grammar, 
                          parser='lalr',
                          start='kotlinFile',
                          propagate_positions=True)
        
    def _load_grammar(self) -> str:
        """Load Kotlin grammar from file"""
        grammar_path = Path(__file__).parent / "grammars" / "kotlin.lark"
        if not grammar_path.exists():
            raise FileNotFoundError(f"Grammar file not found at {grammar_path}")
        return grammar_path.read_text(encoding='utf-8')
    
    def parse(self, code: str) -> Optional[Tree]:
        """Parse Kotlin source code and return AST"""
        try:
            return self.parser.parse(code)
        except UnexpectedToken as e:
            logger.error(f"Failed to parse Kotlin code: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while parsing Kotlin code: {e}")
            return None
            
    def extract_classes(self, tree: Tree) -> List[Dict[str, Any]]:
        """Extract class definitions from AST"""
        if not tree:
            return []
            
        classes = []
        for node in tree.find_data('classDeclaration'):
            class_info = self._extract_class_info(node)
            if class_info:
                classes.append(class_info)
                
        return classes
        
    def _extract_class_info(self, node: Tree) -> Dict[str, Any]:
        """Extract information about a class from its AST node"""
        info = {
            'name': '',
            'superclasses': [],
            'methods': [],
            'is_android_component': False
        }
        
        # Extract class name
        for child in node.children:
            if isinstance(child, Token) and child.type == 'IDENTIFIER':
                info['name'] = str(child)
                break
                
        # Extract superclasses from delegation specifiers
        for deleg_spec in node.find_data('delegationSpecifiers'):
            for user_type in deleg_spec.find_data('userType'):
                superclass = self._extract_type_name(user_type)
                if superclass:
                    info['superclasses'].append(superclass)
                    
        # Extract methods from class body
        if class_body := next(node.find_data('classBody'), None):
            for func_decl in class_body.find_data('functionDeclaration'):
                method_name = next(
                    (str(child) for child in func_decl.children 
                     if isinstance(child, Token) and child.type == 'IDENTIFIER'),
                    None
                )
                if method_name:
                    info['methods'].append(method_name)
                    
        # Check if it's an Android component
        info['is_android_component'] = self._is_android_component(info['superclasses'])
        
        return info
        
    def _extract_type_name(self, node: Tree) -> Optional[str]:
        """Extract full type name from userType node"""
        names = []
        for simple_type in node.find_data('simpleUserType'):
            for child in simple_type.children:
                if isinstance(child, Token) and child.type == 'IDENTIFIER':
                    names.append(str(child))
        return '.'.join(names) if names else None
        
    def _is_android_component(self, superclasses: List[str]) -> bool:
        """Check if class extends any Android component"""
        android_components = {
            'Activity',
            'Fragment',
            'Service',
            'BroadcastReceiver',
            'ContentProvider',
            'Application',
            'View',
            'ViewGroup'
        }
        return any(any(component in superclass for component in android_components) 
                  for superclass in superclasses) 