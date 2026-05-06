# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""This package contains round behaviours of MemeooorrAbciApp."""

import json
import os
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Generator, List, Optional, Type

import requests

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.mech_interact_abci.states.base import (
    MechInteractionResponse,
)
from packages.valory.skills.memeooorr_abci.behaviour_classes.base import (
    MemeooorrBaseBehaviour,
)
from packages.valory.skills.memeooorr_abci.payloads import MechPayload
from packages.valory.skills.memeooorr_abci.rounds import (
    FailedMechRequestRound,
    FailedMechResponseRound,
    PostMechResponseRound,
)

IPFS_REQUEST_TIMEOUT = 30
IPFS_POLL_INTERVAL_SECONDS = 1.0


class PostMechResponseBehaviour(
    MemeooorrBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """PostMechResponseBehaviour"""

    matching_round: Type[AbstractRound] = PostMechResponseRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        mech_for_twitter = False
        failed_mech = False
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            self.context.logger.info(
                f"Processing mech response: {self.synchronized_data.mech_responses}"
            )

            # Process response and fetch media using the helper method
            mech_for_twitter = yield from self._process_mech_response_and_fetch_media(
                self.synchronized_data.mech_responses
            )

            if mech_for_twitter:
                self.context.logger.info("Mech response processed successfully")
            else:
                failed_mech = True
                self.context.logger.error("Failed to process mech response")

            sender = self.context.agent_address
            payload = MechPayload(
                sender=sender,
                mech_for_twitter=mech_for_twitter,
                failed_mech=failed_mech,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def _process_mech_response_and_fetch_media(
        self, mech_responses: List[MechInteractionResponse]
    ) -> Generator[None, None, bool]:
        """
        Process the mech response, fetch media (video or image) based on the format, and save media info.

        :param mech_responses: The list of mech interaction responses.
        :yields: None while processing.
        :return: True if media was successfully processed and saved, False otherwise.
        """
        if not mech_responses:
            self.context.logger.error("No mech responses found")
            return False

        response_data = mech_responses[0]
        if response_data.result is None:
            self.context.logger.error("Mech response result is None")
            return False  # Early return for None result

        # Handle JSON parsing separately and immediately
        try:
            result_json = json.loads(response_data.result)
        except json.JSONDecodeError as e:
            self.context.logger.error(
                f"Error decoding JSON from mech response result: {response_data.result} - {e}"
            )
            return False  # Fail fast on JSON decode error

        # Main fetch/save logic block
        video_hash = result_json.get("video")
        image_hash = result_json.get("image_hash")

        # Case 1: Video format (Attempt first if key exists)
        if video_hash:
            self.context.logger.info(f"Attempting video fetch for hash: {video_hash}")
            # _fetch_media_from_ipfs_hash handles its own network/IO errors and returns Optional[str]
            video_path = yield from self._fetch_media_from_ipfs_hash(
                video_hash, "video", ".mp4"
            )

            if video_path:  # Video fetch succeeded
                self.context.logger.info(
                    f"Video downloaded to: {video_path}. Saving metadata..."
                )
                # Attempt to save metadata. _save_media_info handles errors and returns True/False
                save_success = yield from self._save_media_info(
                    video_path, "video", video_hash
                )
                if save_success:
                    self.context.logger.info("Video metadata saved successfully.")
                    return True

            else:
                self.context.logger.warning(
                    f"Video fetch failed for hash: {video_hash}. Will check for image fallback."
                )
                # Proceed to image check

        # Case 2: Image hash format
        if image_hash:
            self.context.logger.info(f"Attempting image fetch for hash: {image_hash}")
            # _fetch_media_from_ipfs_hash handles its own network/IO errors and returns Optional[str]
            image_path = yield from self._fetch_media_from_ipfs_hash(
                image_hash, "image", ".png"
            )

            if image_path:  # Image fetch succeeded
                self.context.logger.info(
                    f"Image downloaded to: {image_path}. Saving metadata..."
                )
                # Attempt to save metadata. _save_media_info handles errors and returns True/False
                save_success = yield from self._save_media_info(
                    image_path, "image", image_hash
                )
                if save_success:
                    self.context.logger.info("Image metadata saved successfully.")
                    return True  # SUCCESS

        # If we reach here, neither video nor image processing succeeded.
        self.context.logger.warning(
            "Could not process mech response: No video/image successfully fetched and saved. Looked for keys 'video' and 'image_hash'."
        )
        return False  # FAILURE

    def _save_media_info(
        self, media_path: str, media_type: str, ipfs_hash: str
    ) -> Generator[None, None, bool]:
        """Helper method to save media information to the key-value store. Returns True on success, False on failure."""
        media_info = {"path": media_path, "type": media_type, "ipfs_hash": ipfs_hash}
        try:
            yield from self._store_media_info_list(media_info)

            # this is for backwards compatibility as we are using this in many places inside twitter.py
            yield from self._write_kv({"latest_media_info": json.dumps(media_info)})
            self.context.logger.info(
                f"Stored media info ({media_type}) via _write_kv: {media_info}"
            )
            return True  # Success
        except Exception as e:  # pylint: disable=broad-except
            # Catch potential errors from _write_kv method
            self.context.logger.error(
                f"Failed to save media metadata ({media_type}) for path {media_path} via _write_kv: {e}"
            )
            self.context.logger.error(traceback.format_exc())
            return False  # Failure

    def _download_and_save_media(
        self,
        response: requests.Response,
        ipfs_gateway_url: str,
        suffix: str,
        ipfs_hash: str,
    ) -> Optional[str]:
        """Download media stream from response and save to a designated media directory."""
        media_path = None
        downloaded_size = 0
        chunk_count = 0
        try:
            # Get storage path from params and ensure it exists
            storage_path = os.path.join(self.params.store_path, "media")
            os.makedirs(storage_path, exist_ok=True)

            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{ipfs_hash}{suffix}"
            media_path = os.path.join(storage_path, filename)

            with open(media_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        chunk_count += 1
                f.flush()
                os.fsync(f.fileno())

            if downloaded_size == 0:
                self.context.logger.error(
                    f"Received empty content (0 bytes downloaded) from {ipfs_gateway_url}"
                )
                # Cleanup is handled by the caller using _cleanup_temp_file
                return None

            self.context.logger.info(
                f"Successfully fetched media (size: {downloaded_size} bytes, chunks: {chunk_count}) to: {media_path}"
            )
            return media_path

        except IOError as e:
            self.context.logger.error(
                f"File I/O error during media download/save to {media_path}: {e}"
            )
            self.context.logger.error(traceback.format_exc())
            # Cleanup handled by the caller
            return None

    def _ipfs_fetch_worker(
        self, ipfs_gateway_url: str, suffix: str, ipfs_hash: str, media_type: str
    ) -> Optional[str]:
        """Synchronous worker: fetch from IPFS gateway and save to disk.

        Runs inside a worker thread so the main FSM remains responsive.

        :param ipfs_gateway_url: The full IPFS gateway URL to fetch.
        :param suffix: File suffix used when saving (e.g. ``.mp4``).
        :param ipfs_hash: The IPFS hash being fetched (logging only).
        :param media_type: The media kind label (``"video"`` or ``"image"``).
        :return: The saved media path on success, ``None`` on any error.
        """
        media_path = None
        error_reason = "unknown error"
        try:
            self.context.logger.info(
                f"Attempting synchronous {media_type} fetch via requests: {ipfs_hash}"
            )
            self.context.logger.info(f"Using IPFS gateway URL: {ipfs_gateway_url}")

            with requests.get(
                ipfs_gateway_url, timeout=IPFS_REQUEST_TIMEOUT, stream=True
            ) as response:
                response.raise_for_status()

                media_path = self._download_and_save_media(
                    response, ipfs_gateway_url, suffix, ipfs_hash
                )

                if media_path is None:
                    # Empty download: nothing to clean up (no file was
                    # written) and the helper has already logged the
                    # error. Just surface the failure.
                    return None

            return media_path

        except requests.exceptions.Timeout as e:
            self.context.logger.error(
                f"Timeout occurred while fetching {ipfs_gateway_url}: {e}"
            )
            error_reason = "timeout"
        except requests.exceptions.HTTPError as e:
            self.context.logger.error(
                f"HTTP error occurred for {ipfs_gateway_url}: {e.response.status_code} - {e.response.reason}"
            )
            error_reason = "http error"
        except requests.exceptions.RequestException as e:
            self.context.logger.error(
                f"HTTP request failed for {ipfs_gateway_url}: {e}"
            )
            error_reason = "request exception"

        self._cleanup_temp_file(media_path, error_reason)
        return None

    def _fetch_media_from_ipfs_hash(
        self, ipfs_hash: str, media_type: str, suffix: str
    ) -> Generator[None, None, Optional[str]]:
        """Fetch media from IPFS via a worker thread and yield the FSM control.

        The blocking HTTP fetch runs inside a ThreadPoolExecutor so the
        agent's tick loop and other handlers stay responsive. The
        generator polls the future every IPFS_POLL_INTERVAL_SECONDS,
        yielding via self.sleep between checks.

        Note on direct requests usage: the standard IPFS helpers
        (e.g. get_from_ipfs) were unreliable for large or video files;
        direct requests gave more control over streaming and gateway
        behaviour. Revisit if the built-in helpers become reliable.

        :param ipfs_hash: The IPFS hash to fetch.
        :param media_type: The media kind label (``"video"`` or ``"image"``).
        :param suffix: File suffix used when saving (e.g. ``.mp4``).
        :yield: ``None`` while polling the worker future.
        :return: The saved media path on success, ``None`` on any error.
        """
        ipfs_gateway_url = f"https://gateway.autonolas.tech/ipfs/{ipfs_hash}"

        executor = ThreadPoolExecutor(max_workers=1)
        future: Optional[Future] = None
        try:
            future = executor.submit(
                self._ipfs_fetch_worker,
                ipfs_gateway_url,
                suffix,
                ipfs_hash,
                media_type,
            )
            while not future.done():
                yield from self.sleep(IPFS_POLL_INTERVAL_SECONDS)
            try:
                return future.result()
            except Exception as exc:  # pylint: disable=broad-except
                # The worker's documented contract is Optional[str]. Any
                # error not handled inside ``_ipfs_fetch_worker``
                # (filesystem errors, MemoryError, library bugs) would
                # otherwise propagate out of the generator and abort
                # the round. Convert to ``None`` to honour the contract.
                self.context.logger.error(
                    f"Unexpected error fetching {media_type} {ipfs_hash}: "
                    f"{exc}\n{traceback.format_exc()}"
                )
                return None
        finally:
            # ``cancel_futures=True`` only stops pending submissions;
            # an in-flight requests.get cannot be interrupted from
            # Python and will drain on its own once its socket
            # timeout fires. Log abandonment so the leak is visible
            # rather than silent.
            if future is not None and not future.done():
                self.context.logger.warning(
                    f"IPFS fetch worker abandoned for {ipfs_hash} "
                    f"({media_type}); thread will drain on the request "
                    "socket timeout."
                )
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)


class FailedMechRequestBehaviour(
    MemeooorrBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """FailedMechRequestBehaviour"""

    matching_round: Type[AbstractRound] = FailedMechRequestRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            self.context.logger.info(
                f"FailedMechRequest: mech_responses = {self.synchronized_data.mech_responses}"
            )

            sender = self.context.agent_address
            payload = MechPayload(
                sender=sender,
                mech_for_twitter=False,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()


class FailedMechResponseBehaviour(
    MemeooorrBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """FailedMechResponseBehaviour"""

    matching_round: Type[AbstractRound] = FailedMechResponseRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            self.context.logger.info(
                f"FailedMechResponse: mech_responses = {self.synchronized_data.mech_responses}"
            )

            sender = self.context.agent_address
            payload = MechPayload(
                sender=sender,
                mech_for_twitter=False,
                failed_mech=True,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()
