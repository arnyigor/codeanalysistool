import unittest
from pathlib import Path
from code_analyzer.kotlin_parser import KotlinParser

class TestKotlinParser(unittest.TestCase):
    def setUp(self):
        self.parser = KotlinParser()
        
    def test_parse_simple_class(self):
        code = """
        class SimpleClass {
        }
        """
        tree = self.parser.parse(code)
        self.assertIsNotNone(tree)
        
        classes = self.parser.extract_classes(tree)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]['name'], 'SimpleClass')
        self.assertEqual(classes[0]['superclasses'], [])
        self.assertFalse(classes[0]['is_android_component'])
        
    def test_parse_android_activity(self):
        code = """
        class MainActivity : AppCompatActivity() {
            override fun onCreate(savedInstanceState: Bundle?) {
                super.onCreate(savedInstanceState)
                setContentView(R.layout.activity_main)
            }
        }
        """
        tree = self.parser.parse(code)
        self.assertIsNotNone(tree)
        
        classes = self.parser.extract_classes(tree)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]['name'], 'MainActivity')
        self.assertEqual(classes[0]['superclasses'], ['AppCompatActivity'])
        self.assertTrue(classes[0]['is_android_component'])
        
    def test_parse_android_fragment(self):
        code = """
        class HomeFragment : Fragment() {
            override fun onCreateView(
                inflater: LayoutInflater,
                container: ViewGroup?,
                savedInstanceState: Bundle?
            ): View? {
                return inflater.inflate(R.layout.fragment_home, container, false)
            }
        }
        """
        tree = self.parser.parse(code)
        self.assertIsNotNone(tree)
        
        classes = self.parser.extract_classes(tree)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]['name'], 'HomeFragment')
        self.assertEqual(classes[0]['superclasses'], ['Fragment'])
        self.assertTrue(classes[0]['is_android_component'])
        
    def test_parse_multiple_classes(self):
        code = """
        class MainActivity : AppCompatActivity() {
        }
        
        class HomeFragment : Fragment() {
        }
        
        class UserData {
        }
        """
        tree = self.parser.parse(code)
        self.assertIsNotNone(tree)
        
        classes = self.parser.extract_classes(tree)
        self.assertEqual(len(classes), 3)
        
        # Check MainActivity
        self.assertEqual(classes[0]['name'], 'MainActivity')
        self.assertTrue(classes[0]['is_android_component'])
        
        # Check HomeFragment
        self.assertEqual(classes[1]['name'], 'HomeFragment')
        self.assertTrue(classes[1]['is_android_component'])
        
        # Check UserData
        self.assertEqual(classes[2]['name'], 'UserData')
        self.assertFalse(classes[2]['is_android_component'])
        
    def test_parse_invalid_code(self):
        code = """
        class Invalid Syntax {
        """
        tree = self.parser.parse(code)
        self.assertIsNone(tree)
        
if __name__ == '__main__':
    unittest.main() 