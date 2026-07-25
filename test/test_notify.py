from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from lpap.notify import (
    NotifyError,
    env_flag,
    notify_training_finished,
    send_pushover,
)


class NotifyTest(unittest.TestCase):
    def test_env_flag(self) -> None:
        with patch.dict(os.environ, {"LPAP_NOTIFY_ON_FINISHED": "1"}, clear=False):
            self.assertTrue(env_flag("LPAP_NOTIFY_ON_FINISHED"))
        with patch.dict(os.environ, {"LPAP_NOTIFY_ON_FINISHED": "false"}, clear=False):
            self.assertFalse(env_flag("LPAP_NOTIFY_ON_FINISHED"))
        with patch.dict(os.environ, {"LPAP_NOTIFY_ON_FINISHED": ""}, clear=False):
            self.assertFalse(env_flag("LPAP_NOTIFY_ON_FINISHED"))
            self.assertTrue(env_flag("LPAP_NOTIFY_ON_FINISHED", default=True))

    def test_send_pushover_posts_form(self) -> None:
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(
            {"status": 1, "request": "req-123"}
        ).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            data = send_pushover(
                "hello",
                title="t",
                token="app-token",
                user="user-key",
                priority=1,
            )
        self.assertEqual(data["request"], "req-123")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.pushover.net/1/messages.json")
        body = request.data.decode("utf-8")
        self.assertIn("token=app-token", body)
        self.assertIn("user=user-key", body)
        self.assertIn("message=hello", body)
        self.assertIn("title=t", body)

    def test_send_pushover_requires_credentials(self) -> None:
        with patch.dict(os.environ, {"PUSHOVER_TOKEN": "", "PUSHOVER_USER": ""}, clear=False):
            with self.assertRaises(NotifyError):
                send_pushover("x")

    def test_notify_training_finished_formats_message(self) -> None:
        with patch(
            "lpap.notify.send_pushover", return_value={"status": 1}
        ) as send_mock:
            notify_training_finished(
                run_id="ae:test",
                step=100,
                total_steps=1000,
                best_metric=0.01234,
                status="finished",
            )
        send_mock.assert_called_once()
        message = send_mock.call_args.args[0]
        self.assertIn("run=ae:test", message)
        self.assertIn("step=100/1000", message)
        self.assertIn("best=0.01234", message)
        self.assertEqual(send_mock.call_args.kwargs["title"], "LPAP finished")


if __name__ == "__main__":
    unittest.main()
