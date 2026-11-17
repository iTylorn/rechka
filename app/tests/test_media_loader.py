from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from media_loader import MediaCancelled, MediaLoader
from media_validation import check_m4a


class MediaLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = MediaLoader()

    def test_aac_m4a_mono_decodes_without_system_codec(self):
        check_m4a(self.loader.find_ffmpeg(), 'aac', 1)

    def test_alac_m4a_stereo_retains_two_channels(self):
        check_m4a(self.loader.find_ffmpeg(), 'alac', 2)

    def test_corrupt_m4a_reports_format_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'corrupt.m4a'
            source.write_bytes(b'invalid audio')
            with patch('media_loader.tempfile.tempdir', folder):
                with self.assertRaisesRegex(RuntimeError, 'M4A'):
                    self.loader.prepare(source)
            self.assertEqual(list(Path(folder).iterdir()), [source])

    def test_pre_cancelled_m4a_never_starts_decoder(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'cancelled.m4a'
            source.touch()
            with patch.object(self.loader, 'find_ffmpeg') as find_ffmpeg:
                with self.assertRaises(MediaCancelled):
                    self.loader.prepare(source, cancel_check=lambda: True)
            find_ffmpeg.assert_not_called()
