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


import base64
import json
import logging
import os
import re
import subprocess
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional


try:
    import requests
except ImportError:
    logging.error(
        "The 'requests' library is not installed. Please install it: pip install requests"
    )
    exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MechImageFetcher:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_direct_mech_output(
        self, mech_stdout_str: str, prompt_text_for_filename: str
    ) -> Optional[str]:
        """Parses direct stdout from mechx, decodes base64 image, and saves it."""
        DATA_MARKER = "- Data from agent:"
        json_to_parse = None

        try:
            logging.info(
                f"Processing direct MechX output for prompt: '{prompt_text_for_filename}'"
            )

            marker_index = mech_stdout_str.find(DATA_MARKER)
            if marker_index != -1:
                # Start searching for JSON after the marker
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
                    return None
            else:
                # Fallback: try to find the first '{' and last '}' in the whole output if marker is missing
                # This is less reliable and might grab unintended JSON if present in logs.
                logging.warning(
                    f"'{DATA_MARKER}' not found. Attempting fallback JSON extraction (less reliable)."
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
                        f"Fallback JSON extraction failed: Could not find valid JSON braces {{...}} in the entire output. Output preview: {mech_stdout_str[:200]}"
                    )
                    return None

            if json_to_parse is None:  # Should be caught by above logic
                logging.error("Failed to extract a JSON string from MechX output.")
                return None

            try:
                outer_json = json.loads(json_to_parse)
            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON decoding error for extracted MechX output: {e}. Extracted string was: {json_to_parse[:300]}"
                )
                return None

            result_val = outer_json.get("result")
            if (
                result_val is None
            ):  # Allow empty string or other falsy values if they are valid json for inner_json
                logging.error(
                    f"'result' field is missing in MechX output. Keys: {list(outer_json.keys())}"
                )
                return None

            inner_json: Optional[Dict[str, Any]] = None
            if isinstance(result_val, str):
                if (
                    not result_val.strip()
                ):  # Handle case where result is an empty string
                    logging.error(
                        "'result' field is an empty string, cannot parse as JSON."
                    )
                    return None
                try:
                    inner_json = json.loads(result_val)
                except json.JSONDecodeError as e:
                    logging.error(
                        f"JSON decoding error for 'result' field string: {e}. Result string was: {result_val[:200]}"
                    )
                    return None
            elif isinstance(result_val, dict):
                inner_json = result_val
            else:
                logging.error(
                    f"'result' field is not a string or a dict. Got: {type(result_val)}"
                )
                return None

            if inner_json is None:  # Should be caught by earlier checks
                logging.error("Failed to obtain inner JSON from 'result' field.")
                return None

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

            image_base64 = artifacts[0]["base64"]
            image_data = base64.b64decode(image_base64)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
            # Sanitize prompt text for filename
            prompt_part = re.sub(
                r"[^a-zA-Z0-9_-]", "", prompt_text_for_filename.replace(" ", "_")
            )[:25]
            if not prompt_part:  # Ensure prompt_part is not empty after sanitization
                prompt_part = "image"
            file_name = f"{prompt_part}_{timestamp}.png"
            image_path = os.path.join(self.output_dir, file_name)

            with open(image_path, "wb") as f:
                f.write(image_data)
            logging.info(f"Successfully saved image: {image_path}")
            return image_path

        except json.JSONDecodeError as e:  # Should be caught above, but as a fallback
            logging.error(
                f"JSON decoding error during processing: {e}. Input: {mech_stdout_str[:200]}"
            )
        except (TypeError, KeyError, IndexError, base64.binascii.Error) as e:
            logging.error(f"Data processing or decoding error: {e}")
        except IOError as e:
            logging.error(f"File saving error: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while processing direct output: {e}\n{traceback.format_exc()}"
            )
        return None


def run_mechx_command(command_args: List[str]) -> Optional[str]:
    """Runs the mechx command and returns its stdout if 'Data arrived' signal is found and stdout is not empty."""
    logging.info(f"Executing: {' '.join(command_args)}")
    try:
        process = subprocess.run(
            command_args, capture_output=True, text=True, check=False, timeout=300
        )

        stdout_data = process.stdout or ""
        stderr_data = process.stderr or ""
        combined_output = stdout_data + "\n" + stderr_data

        data_arrived_signal_found = "Data arrived:" in combined_output

        if data_arrived_signal_found:
            logging.info(
                f"'Data arrived' signal found in MechX output (rc={process.returncode})."
            )
            if process.returncode != 0:
                logging.warning(
                    f"MechX command completed with non-zero exit code {process.returncode} but 'Data arrived' signal was present."
                )
                if stdout_data:
                    logging.warning(
                        f"MechX STDOUT (signal found, non-zero rc): {stdout_data.strip()[:1000]}"
                    )
                if stderr_data:
                    logging.warning(
                        f"MechX STDERR (signal found, non-zero rc): {stderr_data.strip()[:1000]}"
                    )

            logging.info(
                "Waiting 15 seconds after 'Data arrived' signal detection as requested..."
            )
            time.sleep(15)

            if stdout_data:
                logging.info(
                    "Proceeding to return stdout for direct parsing attempt after 15s wait."
                )
                return stdout_data
            else:  # Data arrived signal found, but stdout is empty
                logging.error("'Data arrived' signal found, but stdout is empty.")
                if stderr_data:
                    logging.info(
                        f"MechX STDERR (stdout empty, signal found): {stderr_data.strip()[:1000]}"
                    )
                return None
        else:  # Data arrived signal NOT found
            logging.error(
                f"Could not find 'Data arrived' signal in MechX output (rc={process.returncode})."
            )
            if stdout_data:
                logging.info(f"MechX STDOUT (no signal): {stdout_data.strip()[:1000]}")
            else:
                logging.info("MechX STDOUT (no signal): <empty>")
            if stderr_data:
                logging.info(f"MechX STDERR (no signal): {stderr_data.strip()[:1000]}")
            else:
                logging.info("MechX STDERR (no signal): <empty>")
            return None

    except subprocess.TimeoutExpired as e:
        logging.error(f"MechX command timed out after {e.timeout} seconds.")
        timeout_stdout_data = (
            e.stdout.decode("utf-8", errors="replace")
            if e.stdout and isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        timeout_stderr_data = (
            e.stderr.decode("utf-8", errors="replace")
            if e.stderr and isinstance(e.stderr, bytes)
            else (e.stderr or "")
        )
        combined_timeout_output = timeout_stdout_data + "\n" + timeout_stderr_data

        data_arrived_signal_found_timeout = "Data arrived:" in combined_timeout_output

        if data_arrived_signal_found_timeout:
            logging.info("'Data arrived' signal found in MechX output after timeout.")

            logging.info(
                "Waiting 15 seconds after 'Data arrived' signal detection (in timeout path) as requested..."
            )
            time.sleep(15)

            if timeout_stdout_data:
                logging.info(
                    "Proceeding to return stdout captured before timeout for direct parsing attempt after 15s wait."
                )
                return timeout_stdout_data
            else:  # Signal found, but timeout_stdout is empty
                logging.error(
                    "'Data arrived' signal found in timeout output, but captured stdout is empty."
                )
                if timeout_stderr_data:
                    logging.info(
                        f"MechX STDERR (on timeout, stdout empty, signal found): {timeout_stderr_data.strip()[:1000]}"
                    )
                return None
        else:  # Data arrived signal NOT found in timeout output
            logging.error(
                "Could not find 'Data arrived' signal in MechX output after timeout."
            )
            if timeout_stdout_data:
                logging.info(
                    f"MechX STDOUT (on timeout, no signal): {timeout_stdout_data.strip()[:1000]}"
                )
            else:
                logging.info(
                    "MechX STDOUT (on timeout, no signal): <empty or not captured>"
                )
            if timeout_stderr_data:
                logging.info(
                    f"MechX STDERR (on timeout, no signal): {timeout_stderr_data.strip()[:1000]}"
                )
            else:
                logging.info(
                    "MechX STDERR (on timeout, no signal): <empty or not captured>"
                )
            return None
    except FileNotFoundError:
        logging.error("`mechx` command not found. Ensure it's installed and in PATH.")
        return None
    except Exception as e:
        logging.error(f"Error running MechX command: {e}\n{traceback.format_exc()}")
        return None


def main_process(num_calls: int, prompt_text: str):
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
        logging.info(f"\n--- Call {i + 1}/{num_calls} for prompt: '{prompt_text}' ---")
        mech_output_str = run_mechx_command(base_command_args)
        if mech_output_str:
            if fetcher.process_direct_mech_output(mech_output_str, prompt_text):
                saved_count += 1
        else:
            logging.error(
                f"Skipping image processing for call {i+1} due to missing or invalid MechX output."
            )

    logging.info(f"\n--- Process Complete ---")
    logging.info(f"Total images saved: {saved_count}/{num_calls}")
    logging.info(f"Check the '{fetcher.output_dir}' directory for images.")


if __name__ == "__main__":
    try:
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
        exit(1)

    NUMBER_OF_CALLS = 5
    PROMPT = "solana logo"  # Fixed prompt as requested
    main_process(NUMBER_OF_CALLS, PROMPT)
