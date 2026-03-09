# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 David Vilela Freire
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Tests for behaviour_classes/mech.py."""

# pylint: disable=protected-access,unused-argument,used-before-assignment,useless-return,broad-exception-raised,unreachable,import-outside-toplevel

import json
from typing import Any
from unittest.mock import MagicMock, patch

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.mech import (
    FailedMechRequestBehaviour,
    FailedMechResponseBehaviour,
    PostMechResponseBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    FailedMechRequestRound,
    FailedMechResponseRound,
    PostMechResponseRound,
)

from .conftest import make_mock_context, make_mock_params, make_mock_synchronized_data


class TestMatchingRounds:
    """Tests for matching_round assignments."""

    def test_post_mech_response_matching_round(self) -> None:
        """Test PostMechResponseBehaviour has correct matching_round."""
        assert PostMechResponseBehaviour.matching_round is PostMechResponseRound

    def test_failed_mech_request_matching_round(self) -> None:
        """Test FailedMechRequestBehaviour has correct matching_round."""
        assert FailedMechRequestBehaviour.matching_round is FailedMechRequestRound

    def test_failed_mech_response_matching_round(self) -> None:
        """Test FailedMechResponseBehaviour has correct matching_round."""
        assert FailedMechResponseBehaviour.matching_round is FailedMechResponseRound


class TestProcessMechResponseAndFetchMedia:
    """Tests for _process_mech_response_and_fetch_media."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_no_mech_responses(self) -> None:
        """Test returns False when no mech responses."""
        behaviour = self._make_behaviour()

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, []
        )
        result = None
        try:
            result = next(gen)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_none_result(self) -> None:
        """Test returns False when response result is None."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = None

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            result = next(gen)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_invalid_json_result(self) -> None:
        """Test returns False when response result is invalid JSON."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = "not valid json{"

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            result = next(gen)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_no_video_no_image(self) -> None:
        """Test returns False when no video or image in response."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps({"text": "some text"})

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            result = next(gen)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_video_fetch_success(self) -> None:
        """Test returns True when video is successfully fetched."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps({"video": "QmHash123"})

        behaviour._fetch_media_from_ipfs_hash = MagicMock(return_value="/tmp/video.mp4")

        def mock_save_media_info(media_path, media_type, ipfs_hash):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._save_media_info = mock_save_media_info

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_video_fetch_failure_falls_through_to_image(self) -> None:
        """Test falls through to image when video fetch fails."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps(
            {"video": "QmHash123", "image_hash": "QmImageHash"}
        )

        # Video fetch fails
        behaviour._fetch_media_from_ipfs_hash = MagicMock(
            side_effect=[None, "/tmp/image.png"]
        )

        def mock_save_media_info(media_path, media_type, ipfs_hash):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._save_media_info = mock_save_media_info

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_image_fetch_success(self) -> None:
        """Test returns True when image is successfully fetched."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps({"image_hash": "QmImageHash"})

        behaviour._fetch_media_from_ipfs_hash = MagicMock(return_value="/tmp/image.png")

        def mock_save_media_info(media_path, media_type, ipfs_hash):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._save_media_info = mock_save_media_info

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_image_fetch_failure(self) -> None:
        """Test returns False when image fetch fails."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps({"image_hash": "QmImageHash"})

        behaviour._fetch_media_from_ipfs_hash = MagicMock(return_value=None)

        gen = PostMechResponseBehaviour._process_mech_response_and_fetch_media(
            behaviour, [response]
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False


class TestSaveMediaInfo:
    """Tests for _save_media_info."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_save_media_info_success(self) -> None:
        """Test _save_media_info returns True on success."""
        behaviour = self._make_behaviour()

        def mock_store_media_info_list(media_info):  # type: ignore[no-untyped-def]
            yield
            return None

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._store_media_info_list = mock_store_media_info_list
        behaviour._write_kv = mock_write_kv

        gen = PostMechResponseBehaviour._save_media_info(
            behaviour, "/tmp/test.png", "image", "QmHash"
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_save_media_info_exception(self) -> None:
        """Test _save_media_info returns False on exception."""
        behaviour = self._make_behaviour()

        def mock_store_media_info_list(media_info):  # type: ignore[no-untyped-def]
            raise Exception("write error")
            yield  # noqa: E501  # make it a generator

        behaviour._store_media_info_list = mock_store_media_info_list

        gen = PostMechResponseBehaviour._save_media_info(
            behaviour, "/tmp/test.png", "image", "QmHash"
        )
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False


class TestDownloadAndSaveMedia:
    """Tests for _download_and_save_media."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params(store_path="/tmp/test")
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    @patch("os.makedirs")
    @patch("os.fsync")
    def test_empty_download(self, mock_fsync: Any, mock_makedirs: Any) -> None:
        """Test returns None when download is empty."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.iter_content.return_value = []

        with patch("builtins.open", MagicMock()):
            result = PostMechResponseBehaviour._download_and_save_media(
                behaviour,
                response,
                "https://gateway.example.com/ipfs/QmHash",
                ".png",
                "QmHash",
            )
        assert result is None

    @patch("os.makedirs")
    @patch("os.fsync")
    def test_successful_download(self, mock_fsync: Any, mock_makedirs: Any) -> None:
        """Test returns path on successful download."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.iter_content.return_value = [b"chunk1", b"chunk2"]

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.fileno.return_value = 3

        with patch("builtins.open", return_value=mock_file):
            result = PostMechResponseBehaviour._download_and_save_media(
                behaviour,
                response,
                "https://gateway.example.com/ipfs/QmHash",
                ".png",
                "QmHash",
            )
        assert result is not None
        assert result.endswith(".png")

    @patch("os.makedirs")
    def test_io_error(self, mock_makedirs: Any) -> None:
        """Test returns None on IO error."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.iter_content.side_effect = IOError("disk full")

        with patch("builtins.open", MagicMock()):
            result = PostMechResponseBehaviour._download_and_save_media(
                behaviour,
                response,
                "https://gateway.example.com/ipfs/QmHash",
                ".png",
                "QmHash",
            )
        assert result is None


class TestFetchMediaFromIpfsHash:
    """Tests for _fetch_media_from_ipfs_hash."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour._cleanup_temp_file = MagicMock()
        return behaviour

    @patch("requests.get")
    def test_timeout_error(self, mock_get: Any) -> None:
        """Test returns None on timeout."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        behaviour = self._make_behaviour()

        result = PostMechResponseBehaviour._fetch_media_from_ipfs_hash(
            behaviour, "QmHash", "video", ".mp4"
        )
        assert result is None

    @patch("requests.get")
    def test_http_error(self, mock_get: Any) -> None:
        """Test returns None on HTTP error."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)
        behaviour = self._make_behaviour()

        result = PostMechResponseBehaviour._fetch_media_from_ipfs_hash(
            behaviour, "QmHash", "image", ".png"
        )
        assert result is None

    @patch("requests.get")
    def test_request_exception(self, mock_get: Any) -> None:
        """Test returns None on general request exception."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("network error")
        behaviour = self._make_behaviour()

        result = PostMechResponseBehaviour._fetch_media_from_ipfs_hash(
            behaviour, "QmHash", "image", ".png"
        )
        assert result is None

    @patch("requests.get")
    def test_successful_fetch(self, mock_get: Any) -> None:
        """Test returns path on successful fetch."""
        behaviour = self._make_behaviour()
        behaviour._download_and_save_media = MagicMock(return_value="/tmp/test.mp4")

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = PostMechResponseBehaviour._fetch_media_from_ipfs_hash(
            behaviour, "QmHash", "video", ".mp4"
        )
        assert result == "/tmp/test.mp4"

    @patch("requests.get")
    def test_fetch_empty_content(self, mock_get: Any) -> None:
        """Test returns None when download returns None (empty content)."""
        behaviour = self._make_behaviour()
        behaviour._download_and_save_media = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = PostMechResponseBehaviour._fetch_media_from_ipfs_hash(
            behaviour, "QmHash", "image", ".png"
        )
        assert result is None
