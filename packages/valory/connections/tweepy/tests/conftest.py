# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2026 Valory AG
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

"""Conftest for tweepy connection tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the project root is on sys.path so `packages.*` imports resolve
project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The local `tweepy` package directory can shadow the installed `tweepy` library
# during pytest collection. Pre-load the real tweepy modules to avoid this.
import importlib  # noqa: E402  # pylint: disable=wrong-import-position

_real_tweepy = importlib.import_module("tweepy")
if not hasattr(_real_tweepy, "tweet"):
    # If the real tweepy module got shadowed, provide a mock
    sys.modules.setdefault("tweepy.tweet", MagicMock())
