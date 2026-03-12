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
from packages.dvilela.skills.memeooorr_abci.payloads import MechPayload
from packages.dvilela.skills.memeooorr_abci.rounds import Event, SynchronizedData
from packages.dvilela.skills.memeooorr_abci.tests.conftest import (
    MemeooorrFSMBehaviourBaseCase,
    SAFE_ADDRESS,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)
from packages.valory.skills.abstract_round_abci.base import AbciAppDB


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

    def test_video_save_fails_falls_through_to_image(self) -> None:
        """Test returns True when video save fails but image succeeds (branch 131->142)."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps(
            {"video": "QmHash123", "image_hash": "QmImageHash"}
        )

        # Video fetch succeeds but save fails, then image fetch and save succeeds
        behaviour._fetch_media_from_ipfs_hash = MagicMock(
            side_effect=["/tmp/video.mp4", "/tmp/image.png"]
        )

        call_count = [0]

        def mock_save_media_info(media_path, media_type, ipfs_hash):  # type: ignore[no-untyped-def]
            call_count[0] += 1
            yield
            if call_count[0] == 1:
                return False  # Video save fails
            return True  # Image save succeeds

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

    def test_image_save_fails(self) -> None:
        """Test returns False when image fetch succeeds but save fails (branch 155->160)."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        response.result = json.dumps({"image_hash": "QmImageHash"})

        behaviour._fetch_media_from_ipfs_hash = MagicMock(return_value="/tmp/image.png")

        def mock_save_media_info(media_path, media_type, ipfs_hash):  # type: ignore[no-untyped-def]
            yield
            return False

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
        assert result is False


class TestPostMechResponseAsyncActPayload:
    """Tests for PostMechResponseBehaviour.async_act payload construction."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "test_behaviour"
        return behaviour

    def _run_async_act(self, process_result: bool) -> MechPayload:
        """Run async_act with a mocked _process_mech_response_and_fetch_media result."""
        behaviour = self._make_behaviour()
        payloads_sent: list = []

        def mock_process_mech_response_and_fetch_media(mech_responses):  # type: ignore[no-untyped-def]
            yield
            return process_result

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            payloads_sent.append(payload)
            yield
            return None

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._process_mech_response_and_fetch_media = (
            mock_process_mech_response_and_fetch_media
        )
        behaviour.send_a2a_transaction = mock_send_a2a_transaction
        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = PostMechResponseBehaviour.async_act(behaviour)
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration:
            pass

        behaviour.set_done.assert_called_once()
        assert len(payloads_sent) == 1
        return payloads_sent[0]

    def test_async_act_success_path(self) -> None:
        """Test payload when mech response processed successfully (covers line 67)."""
        payload = self._run_async_act(process_result=True)
        assert payload.mech_for_twitter is True
        assert payload.failed_mech is False

    def test_async_act_failure_path(self) -> None:
        """Test payload when mech response processing fails."""
        payload = self._run_async_act(process_result=False)
        assert payload.mech_for_twitter is False
        assert payload.failed_mech is True


class TestSaveMediaInfo:
    """Tests for _save_media_info."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PostMechResponseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_save_media_info_constructs_and_stores_correctly(self) -> None:
        """Test _save_media_info builds the right media_info dict and writes it to KV."""
        behaviour = self._make_behaviour()
        stored_infos: list = []
        kv_writes: list = []

        def mock_store_media_info_list(media_info):  # type: ignore[no-untyped-def]
            stored_infos.append(media_info)
            yield
            return None

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            kv_writes.append(data)
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

        # Verify media_info dict is constructed from args
        expected_info = {
            "path": "/tmp/test.png",
            "type": "image",
            "ipfs_hash": "QmHash",
        }
        assert stored_infos == [expected_info]

        # Verify KV write serializes the same dict under "latest_media_info"
        assert len(kv_writes) == 1
        assert "latest_media_info" in kv_writes[0]
        assert json.loads(kv_writes[0]["latest_media_info"]) == expected_info

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
        """Test returns path, writes chunks, and fsyncs on successful download."""
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
        assert "QmHash" in result

        # Verify chunks were written
        assert mock_file.write.call_count == 2
        mock_file.write.assert_any_call(b"chunk1")
        mock_file.write.assert_any_call(b"chunk2")

        # Verify fsync for durability
        mock_fsync.assert_called_once_with(3)

        # Verify storage directory was created
        mock_makedirs.assert_called_once()

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

    @patch("os.makedirs")
    @patch("os.fsync")
    def test_download_with_empty_chunks_filtered(
        self, mock_fsync: Any, mock_makedirs: Any
    ) -> None:
        """Test that empty chunks are filtered during download (branch 210->209)."""
        behaviour = self._make_behaviour()
        response = MagicMock()
        # Mix of non-empty and empty chunks
        response.iter_content.return_value = [b"chunk1", b"", b"chunk2"]

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
        # Only 2 non-empty chunks should be written
        assert mock_file.write.call_count == 2


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


class _MechBehaviourTestBase(MemeooorrFSMBehaviourBaseCase):
    """Shared setup for mech behaviour FSM tests."""

    _default_data = dict(
        all_participants=["0x" + "0" * 40],
        participants=["0x" + "0" * 40],
        consensus_threshold=1,
        safe_contract_address=SAFE_ADDRESS,
        mech_responses=json.dumps([]),
        mech_for_twitter=False,
        failed_mech=False,
    )


class TestPostMechResponseBehaviourAsyncAct(_MechBehaviourTestBase):
    """Tests for PostMechResponseBehaviour.async_act using FSMBehaviourBaseCase."""

    def test_async_act_no_responses(self) -> None:
        """Test async_act with empty mech responses sends payload and completes."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            PostMechResponseBehaviour.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(setup_data=AbciAppDB.data_to_lists(self._default_data))
            ),
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=Event.DONE)


class TestFailedMechRequestBehaviourAsyncAct(_MechBehaviourTestBase):
    """Tests for FailedMechRequestBehaviour.async_act using FSMBehaviourBaseCase."""

    def test_async_act(self) -> None:
        """Test async_act sends payload and transitions."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            FailedMechRequestBehaviour.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(setup_data=AbciAppDB.data_to_lists(self._default_data))
            ),
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=Event.DONE)


class TestFailedMechResponseBehaviourAsyncAct(_MechBehaviourTestBase):
    """Tests for FailedMechResponseBehaviour.async_act using FSMBehaviourBaseCase."""

    def test_async_act(self) -> None:
        """Test async_act sends payload with failed_mech=True and transitions."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            FailedMechResponseBehaviour.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(setup_data=AbciAppDB.data_to_lists(self._default_data))
            ),
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=Event.DONE)
