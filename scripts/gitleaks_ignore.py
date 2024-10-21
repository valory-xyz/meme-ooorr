#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 Valory AG
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

"""Add ignores to gitleaksignore"""


import json


with open("leak_report", "r", encoding="utf-8") as report_file:
    report = json.load(report_file)
    fps = [leak["Fingerprint"] + "\n" for leak in report]

with open(".gitleaksignore", "a", encoding="utf-8") as ignore_file:
    ignore_file.writelines(fps)
