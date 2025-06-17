#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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
"""
This script checks Mech requests, processes their output, and saves images.

It interacts with the 'mechx' command-line tool to send prompts and retrieve results,
specifically looking for image data encoded in base64 format within the JSON output.
The script handles multiple calls, logs stdout, and manages file saving for the
generated images.
"""

import base64
import json
import logging
import os
import re
import subprocess  # nosec
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional


# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _save_stdout_to_log(
    stdout_content: str,
    output_dir: str,
    base_prompt_text_for_filename: str,
    call_number: int,
    is_timeout_log: bool = False,
) -> None:
    """Saves the provided stdout content to a log file with prompt and call number."""
    log_identifier_for_messages = f"prompt '{base_prompt_text_for_filename}', call {call_number}{'_timeout' if is_timeout_log else ''}"
    if not stdout_content:
        logging.info(
            f"No stdout content to save for {log_identifier_for_messages} (stdout was empty)."
        )
        return
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        timeout_suffix = "_timeout" if is_timeout_log else ""
        # Ensure base_prompt_text_for_filename is already sanitized and suitable for filenames
        log_file_name = f"{base_prompt_text_for_filename}_call_{call_number}{timeout_suffix}_{timestamp}.log"
        log_file_path = os.path.join(output_dir, log_file_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(stdout_content)
        logging.info(
            f"Saved MechX stdout for {log_identifier_for_messages} to: {log_file_path}"
        )
    except IOError as e_io_save:
        logging.error(
            f"Failed to save MechX stdout for {log_identifier_for_messages} to file: {e_io_save}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.error(
            f"An unexpected error occurred while saving stdout for {log_identifier_for_messages}: {exc}\\n{traceback.format_exc()}"
        )


class MechImageFetcher:  # pylint: disable=too-few-public-methods
    """
    Fetches and processes images from MechX output.

    This class is responsible for parsing the stdout of the MechX tool,
    extracting base64 encoded image data from a JSON structure,
    decoding it, and saving it to a file.
    """

    def __init__(self, output_dir="outputs"):  # pylint: disable=too-few-public-methods
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _extract_json_str_from_stdout(
        mech_stdout_str: str, prompt_text_for_filename: str
    ) -> Optional[str]:
        """Extracts the relevant JSON string from MechX stdout."""
        DATA_MARKER = "- Data from agent:"
        json_to_parse = None
        marker_index = mech_stdout_str.find(DATA_MARKER)

        if marker_index != -1:
            search_area = mech_stdout_str[marker_index + len(DATA_MARKER) :]
            first_brace = search_area.find("{")
            last_brace = search_area.rfind("}")

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_to_parse = search_area[first_brace : last_brace + 1]
                logging.info(
                    f"Extracted JSON block for parsing (length {len(json_to_parse)}). Preview: {json_to_parse[:100]}...{json_to_parse[-100:] if len(json_to_parse) > 200 else ''}"
                )
            else:
                logging.error(
                    f"Could not find valid JSON braces {{...}} after '{DATA_MARKER}'. Search area preview: {search_area[:200]}"
                )
        else:
            logging.warning(
                f"'{DATA_MARKER}' not found in output for '{prompt_text_for_filename}'. Attempting fallback JSON extraction."
            )
            first_brace = mech_stdout_str.find("{")
            last_brace = mech_stdout_str.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_to_parse = mech_stdout_str[first_brace : last_brace + 1]
                logging.info(
                    f"Fallback extracted JSON block (length {len(json_to_parse)}). Preview: {json_to_parse[:100]}...{json_to_parse[-100:] if len(json_to_parse) > 200 else ''}"
                )
            else:
                logging.error(
                    f"Fallback JSON extraction failed for '{prompt_text_for_filename}': Could not find valid JSON braces {{...}} in the entire output. Output preview: {mech_stdout_str[:200]}"
                )
        return json_to_parse

    @staticmethod
    def _parse_outer_json(json_str: str) -> Optional[Any]:
        """Parses the outer JSON and returns the 'result' field."""
        try:
            outer_json = json.loads(json_str)
            request_id = outer_json.get("requestId")
            if request_id:
                logging.info(f"MechX Request ID: {request_id}")
            else:
                logging.info("MechX Request ID not found in the JSON structure.")
            return outer_json.get("result")
        except json.JSONDecodeError as e_json_outer:
            logging.error(
                f"JSON decoding error for extracted MechX output: {e_json_outer}. Extracted string was: {json_str[:300]}"
            )
        return None

    @staticmethod
    def _parse_inner_json_from_result(result_val: Any) -> Optional[Dict[str, Any]]:
        """Parses the inner JSON from the 'result' field."""
        if result_val is None:
            logging.error("'result' field is missing or null in MechX output.")
            return None

        if isinstance(result_val, str):
            if not result_val.strip():
                logging.error(
                    "'result' field is an empty string, cannot parse as JSON."
                )
                return None
            try:
                return json.loads(result_val)
            except json.JSONDecodeError as e_json_inner:
                logging.error(
                    f"JSON decoding error for 'result' field string: {e_json_inner}. Result string was: {result_val[:200]}"
                )
                return None
        elif isinstance(result_val, dict):
            return result_val
        else:
            logging.error(
                f"'result' field is not a string or a dict. Got: {type(result_val)}"
            )
        return None

    def _save_image_from_artifacts(
        self,
        inner_json: Dict[str, Any],
        prompt_text_for_filename: str,
        call_number: int,
    ) -> Optional[str]:
        """Extracts image data from artifacts and saves it to a file."""
        artifacts = inner_json.get("artifacts")
        if not (
            isinstance(artifacts, list)
            and artifacts
            and isinstance(artifacts[0], dict)
            and "base64" in artifacts[0]
        ):
            logging.error(
                f"Invalid or missing 'artifacts' structure in inner JSON. Artifacts: {str(artifacts)[:200]}"
            )
            return None

        try:
            image_base64 = artifacts[0]["base64"]
            image_data = base64.b64decode(image_base64)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
            prompt_part = re.sub(
                r"[^a-zA-Z0-9_-]", "", prompt_text_for_filename.replace(" ", "_")
            )[:25]
            if not prompt_part:
                prompt_part = "image"
            file_name = f"{prompt_part}_call_{call_number}_{timestamp}.png"
            image_path = os.path.join(self.output_dir, file_name)

            with open(image_path, "wb") as f:
                f.write(image_data)
            logging.info(f"Successfully saved image: {image_path}")
            return image_path
        except (TypeError, KeyError, IndexError, base64.binascii.Error) as e_data_proc:
            logging.error(f"Image data processing or decoding error: {e_data_proc}")
        except IOError as e_io:
            logging.error(f"File saving error: {e_io}")
        return None

    def process_direct_mech_output(
        self, mech_stdout_str: str, prompt_text_for_filename: str, call_number: int
    ) -> Optional[str]:
        """Parses direct stdout from mechx, decodes base64 image, and saves it."""
        logging.info(
            f"Processing direct MechX output for prompt: '{prompt_text_for_filename}', call {call_number}"
        )
        try:
            json_str = MechImageFetcher._extract_json_str_from_stdout(
                mech_stdout_str, prompt_text_for_filename
            )
            if not json_str:
                logging.error(
                    f"Failed to extract a JSON string from MechX output for prompt '{prompt_text_for_filename}', call {call_number}."
                )
                return None

            result_val = MechImageFetcher._parse_outer_json(json_str)
            if result_val is None:
                # Check if 'result' key is truly missing or just has a null value
                try:
                    parsed_outer_json = json.loads(json_str)
                    if "result" not in parsed_outer_json:
                        logging.error(
                            f"'result' field is missing in MechX output for prompt '{prompt_text_for_filename}', call {call_number}. Keys: {list(parsed_outer_json.keys())}"
                        )
                except json.JSONDecodeError:
                    pass  # Error already logged by _parse_outer_json
                return None

            inner_json_data = MechImageFetcher._parse_inner_json_from_result(result_val)
            if not inner_json_data:
                logging.error(
                    f"Failed to parse inner JSON for prompt '{prompt_text_for_filename}', call {call_number}."
                )
                return None

            return self._save_image_from_artifacts(
                inner_json_data, prompt_text_for_filename, call_number
            )

        except Exception as e_unexpected:  # pylint: disable=broad-except
            logging.error(
                f"An unexpected error occurred while processing direct output for '{prompt_text_for_filename}', call {call_number}: {e_unexpected}\\n{traceback.format_exc()}"
            )
        return None


def _handle_mechx_process_output(  # noqa: DAR101, DAR201
    stdout_data: str,
    stderr_data: str,
    base_context_msg: str,
    return_code: Optional[int] = None,
    timed_out: bool = False,
) -> Optional[str]:
    """
    Helper to process stdout/stderr from mechx, check for signal, and log.

    _save_stdout_to_log should be called by the caller of this function.

    Args:
        stdout_data: The standard output from the mechx process. # noqa: DAR101
        stderr_data: The standard error from the mechx process. # noqa: DAR101
        base_context_msg: Base message for logging context. # noqa: DAR101
        return_code: The return code of the mechx process. # noqa: DAR101
        timed_out: Boolean indicating if the process timed out. # noqa: DAR101

    Returns: # noqa: DAR201
        Optional[str]: The stdout data if the 'Data arrived' signal is found and
                       stdout is not empty, otherwise None.
    """
    combined_output = stdout_data + "\\n" + stderr_data
    data_arrived_signal_found = "Data arrived:" in combined_output
    stdout_data_to_return = None

    context_msg = base_context_msg + (" (on timeout)" if timed_out else "")
    rc_msg = (
        f"(rc={return_code})"
        if return_code is not None
        else "(no rc - timeout before completion)"
    )

    if data_arrived_signal_found:
        logging.info(
            f"'Data arrived' signal found in MechX output {rc_msg} {context_msg}."
        )
        if return_code is not None and return_code != 0:
            logging.warning(
                f"MechX command {context_msg} completed with non-zero exit code {return_code} but 'Data arrived' signal was present."
            )
            if stderr_data:
                logging.warning(
                    f"MechX STDERR {context_msg} (signal found, non-zero rc): {stderr_data.strip()[:1000]}"
                )
        logging.info(
            f"Waiting 15 seconds after 'Data arrived' signal detection {context_msg}"
        )
        time.sleep(15)

        if stdout_data:
            logging.info(
                f"Proceeding to return stdout {context_msg} for direct parsing attempt after 15s wait."
            )
            stdout_data_to_return = stdout_data
        else:
            logging.error(
                f"'Data arrived' signal found {context_msg}, but stdout is empty."
            )
            if stderr_data:
                logging.info(
                    f"MechX STDERR {context_msg} (stdout empty, signal found): {stderr_data.strip()[:1000]}"
                )
    else:
        logging.error(
            f"Could not find 'Data arrived' signal in MechX output {rc_msg} {context_msg}."
        )
        if not stdout_data:
            logging.info(f"MechX STDOUT {context_msg} (no signal): <empty>")
        if stderr_data:
            logging.info(
                f"MechX STDERR {context_msg} (no signal): {stderr_data.strip()[:1000]}"
            )
        else:
            logging.info(f"MechX STDERR {context_msg} (no signal): <empty>")

    return stdout_data_to_return


def run_mechx_command(  # pylint: disable=too-many-locals
    command_args: List[str],
    output_dir: str,
    original_prompt_text: str,
    call_number: int,
) -> Optional[str]:
    """Runs the mechx command, saves its stdout to a log file, and returns stdout if 'Data arrived' signal is found."""

    # Sanitize prompt for log filename
    prompt_text_for_log_filename = re.sub(
        r"[^a-zA-Z0-9_-]", "", original_prompt_text.replace(" ", "_")
    )[:25]
    if not prompt_text_for_log_filename:  # Ensure not empty
        prompt_text_for_log_filename = "log"

    logging.info(
        f"Executing: {' '.join(command_args)} for prompt '{original_prompt_text}', call {call_number}"
    )
    stdout_data_to_return = None
    try:  # nosec
        process = subprocess.run(
            command_args, capture_output=True, text=True, check=False, timeout=900
        )
        stdout_data = process.stdout or ""
        stderr_data = process.stderr or ""
        return_code = process.returncode

        _save_stdout_to_log(
            stdout_data, output_dir, prompt_text_for_log_filename, call_number
        )
        base_ctx_msg = f"for prompt '{original_prompt_text}', call {call_number}"
        stdout_data_to_return = _handle_mechx_process_output(
            stdout_data, stderr_data, base_ctx_msg, return_code, timed_out=False
        )

    except subprocess.TimeoutExpired as e_timeout:
        logging.error(
            f"MechX command for prompt '{original_prompt_text}', call {call_number} timed out after {e_timeout.timeout} seconds."
        )
        timeout_stdout_data = (
            e_timeout.stdout.decode("utf-8", errors="replace")
            if e_timeout.stdout and isinstance(e_timeout.stdout, bytes)
            else (e_timeout.stdout or "")
        )
        timeout_stderr_data = (
            e_timeout.stderr.decode("utf-8", errors="replace")
            if e_timeout.stderr and isinstance(e_timeout.stderr, bytes)
            else (e_timeout.stderr or "")
        )

        _save_stdout_to_log(
            timeout_stdout_data,
            output_dir,
            prompt_text_for_log_filename,
            call_number,
            is_timeout_log=True,
        )
        base_ctx_msg_timeout = (
            f"for prompt '{original_prompt_text}', call {call_number}"
        )
        stdout_data_to_return = _handle_mechx_process_output(
            timeout_stdout_data,
            timeout_stderr_data,
            base_ctx_msg_timeout,
            timed_out=True,  # return_code is implicitly None here
        )

    except FileNotFoundError:
        logging.error(
            f"`mechx` command not found for prompt '{original_prompt_text}', call {call_number}. Ensure it's installed and in PATH."
        )
    except Exception as e_general_run:  # pylint: disable=broad-except
        logging.error(
            f"Error running MechX command for prompt '{original_prompt_text}', call {call_number}: {e_general_run}\\n{traceback.format_exc()}"
        )

    return stdout_data_to_return


def main_process(
    num_calls: int, prompt_text: str
):  # noqa: DAR101 (already present, see Args for specific noqas)
    """
    Main process to run MechX commands and save resulting images.

    Iterates a specified number of times, calling the MechX tool with a given
    prompt. It then attempts to process the output to extract and save images.
    Finally, it logs a summary of the operations.

    Args:
        num_calls: The number of times to call the MechX tool. # noqa: DAR101
        prompt_text: The prompt text to use for the MechX tool. # noqa: DAR101
    """
    fetcher = MechImageFetcher()
    saved_count = 0

    base_command_args = [
        "mechx",
        "interact",
        prompt_text,
        "--chain-config",
        "base",
        "--priority-mech",
        "0xe535D7AcDEeD905dddcb5443f41980436833cA2B",
        "--tool",
        "stabilityai-stable-diffusion-v1-6",
    ]

    for i in range(num_calls):
        logging.info(f"\\n--- Call {i + 1}/{num_calls} for prompt: '{prompt_text}' ---")
        mech_output_str = run_mechx_command(
            base_command_args, fetcher.output_dir, prompt_text, i + 1
        )
        if mech_output_str:
            if fetcher.process_direct_mech_output(mech_output_str, prompt_text, i + 1):
                saved_count += 1
        else:
            logging.error(
                f"Skipping image processing for call {i+1} (prompt: '{prompt_text}') due to missing or invalid MechX output."
            )

    logging.info("\\n--- Process Complete ---")
    logging.info(f"Total images saved: {saved_count}/{num_calls}")
    logging.info(f"Check the '{fetcher.output_dir}' directory for images and logs.")


if __name__ == "__main__":
    try:  # nosec
        subprocess.run(
            ["mechx", "--version"], capture_output=True, check=True, timeout=10
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logging.error(
            f"`mechx` command not found or non-functional: {e}. Please ensure it's installed and in PATH."
        )
        sys.exit(1)

    NUMBER_OF_CALLS = 2
    PROMPT = "samsung s25 ultra vs iphone 16 pro max"
    main_process(NUMBER_OF_CALLS, PROMPT)
