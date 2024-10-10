import unittest
from core.models.media_factory import AudioFactory
from core.operators import detect_lang_of_video

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # initialize operator
        detect_lang_of_video.initialize(param={})

    @classmethod
    def tearDownClass(cls):
        # delete config files
        pass

    def test_english_detection(self):
        audio_file_path = "core/operators/sample_data/example.mp4"
        lang = detect_lang_of_video.run(audio_file_path)
        self.assertEqual(lang["id"], "en")
        self.assertEqual(lang["language"], "english")
