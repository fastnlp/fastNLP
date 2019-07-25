import unittest

from fastNLP import Instance


class TestCase(unittest.TestCase):
    
    def test_init(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6]}
        ins = Instance(x=[1, 2, 3], y=[4, 5, 6])
        self.assertTrue(isinstance(ins.fields, dict))
        self.assertEqual(ins.fields, fields)
        
        ins = Instance(**fields)
        self.assertEqual(ins.fields, fields)
    
    def test_add_field(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6]}
        ins = Instance(**fields)
        ins.add_field("z", [1, 1, 1])
        fields.update({"z": [1, 1, 1]})
        self.assertEqual(ins.fields, fields)
    
    def test_get_item(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [1, 1, 1]}
        ins = Instance(**fields)
        self.assertEqual(ins["x"], [1, 2, 3])
        self.assertEqual(ins["y"], [4, 5, 6])
        self.assertEqual(ins["z"], [1, 1, 1])
    
    def test_repr(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [1, 1, 1]}
        ins = Instance(**fields)
        # simple print, that is enough.
        print(ins)
