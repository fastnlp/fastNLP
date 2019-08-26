import unittest
from reproduction.text_classification.data.yelpLoader import yelpLoader

class TestDatasetLoader(unittest.TestCase):
    def test_yelpLoader(self):
        ds = yelpLoader().load('sample_yelp.json')
        assert len(ds) == 20