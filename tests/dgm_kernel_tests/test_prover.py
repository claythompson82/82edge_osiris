import unittest
from unittest.mock import patch, MagicMock
import subprocess
from pathlib import Path
from dgm_kernel.prover import _get_pylint_score


class TestProver(unittest.TestCase):

    DUMMY_ID = "test_id_123"
    DUMMY_DIFF = "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new"
    DUMMY_CODE = "print('hello world')"

    @patch("dgm_kernel.prover.tempfile.NamedTemporaryFile")
    @patch("dgm_kernel.prover.subprocess.run")
    @patch("dgm_kernel.prover.Path")
    def test_get_pylint_score_success(
        self, mock_path_constructor, mock_subprocess_run, mock_tempfile
    ):
        # Setup mock for NamedTemporaryFile
        mock_temp_file = MagicMock()
        mock_temp_file.name = "dummy_temp_file.py"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_tempfile.return_value = mock_temp_file

        # Setup mock for Path
        mock_path_instance = MagicMock()
        mock_path_constructor.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True  # Assume file exists for unlink

        # Setup mock for subprocess.run
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["pylint", "dummy_temp_file.py"],
            returncode=0,  # Pylint usually exits with non-zero for issues, but score parsing is based on stdout
            stdout="Some pylint output...\nYour code has been rated at 9.75/10\nMore output...",
            stderr="",
        )

        score = _get_pylint_score(self.DUMMY_CODE)
        self.assertEqual(score, 9.75)

        mock_tempfile.assert_called_once_with(mode="w", suffix=".py", delete=False)
        mock_temp_file.write.assert_called_once_with(self.DUMMY_CODE)
        mock_subprocess_run.assert_called_once_with(
            ["pylint", "dummy_temp_file.py"], capture_output=True, text=True, timeout=30
        )
        mock_path_constructor.assert_any_call(
            "dummy_temp_file.py"
        )  # Called for exists and unlink
        mock_path_instance.unlink.assert_called_once()

    @patch("dgm_kernel.prover.tempfile.NamedTemporaryFile")
    @patch("dgm_kernel.prover.subprocess.run")
    @patch(
        "dgm_kernel.prover.Path"
    )  # Mock Path to avoid actual file operations in finally
    def test_get_pylint_score_pylint_not_found(
        self, mock_path_constructor, mock_subprocess_run, mock_tempfile
    ):
        mock_temp_file = MagicMock()
        mock_temp_file.name = "dummy_temp_file.py"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_tempfile.return_value = mock_temp_file

        mock_path_instance = MagicMock()
        mock_path_constructor.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_subprocess_run.side_effect = FileNotFoundError("pylint not found")

        with self.assertLogs("dgm_kernel.prover", level="ERROR") as cm:
            score = _get_pylint_score(self.DUMMY_CODE)
        self.assertEqual(score, 0.0)
        self.assertIn("pylint not found – skipping lint score.", cm.output[0])
        mock_path_instance.unlink.assert_called_once()

    @patch("dgm_kernel.prover.tempfile.NamedTemporaryFile")
    @patch("dgm_kernel.prover.subprocess.run")
    @patch("dgm_kernel.prover.Path")
    def test_get_pylint_score_pylint_timeout(
        self, mock_path_constructor, mock_subprocess_run, mock_tempfile
    ):
        mock_temp_file = MagicMock()
        mock_temp_file.name = "dummy_temp_file.py"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_tempfile.return_value = mock_temp_file

        mock_path_instance = MagicMock()
        mock_path_constructor.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
            cmd="pylint", timeout=30
        )

        with self.assertLogs("dgm_kernel.prover", level="ERROR") as cm:
            score = _get_pylint_score(self.DUMMY_CODE)
        self.assertEqual(score, 0.0)
        self.assertIn("pylint timed-out.", cm.output[0])
        mock_path_instance.unlink.assert_called_once()

    @patch("dgm_kernel.prover.tempfile.NamedTemporaryFile")
    @patch("dgm_kernel.prover.subprocess.run")
    @patch("dgm_kernel.prover.Path")
    def test_get_pylint_score_pylint_output_no_score(
        self, mock_path_constructor, mock_subprocess_run, mock_tempfile
    ):
        mock_temp_file = MagicMock()
        mock_temp_file.name = "dummy_temp_file.py"
        mock_temp_file.__enter__.return_value = mock_temp_file
        mock_tempfile.return_value = mock_temp_file

        mock_path_instance = MagicMock()
        mock_path_constructor.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["pylint", "dummy_temp_file.py"],
            returncode=0,
            stdout="Pylint output without any score line.",
            stderr="",
        )

        with self.assertLogs("dgm_kernel.prover", level="WARNING") as cm:
            score = _get_pylint_score(self.DUMMY_CODE)
        self.assertEqual(score, 0.0)
        self.assertIn("Could not parse pylint score from output.", cm.output[0])
        mock_path_instance.unlink.assert_called_once()


if __name__ == "__main__":
    unittest.main()
