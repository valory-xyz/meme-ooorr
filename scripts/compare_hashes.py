#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2026 Valory AG
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

"""Compare package hashes between source-of-truth repos and this repo."""

import json
from pathlib import Path

# Source repos (dev packages become this repo's third_party packages)
SOURCE_REPOS = [
    Path("/home/lockhart/work/valory/repos/open-autonomy/packages/packages.json"),
    Path("/home/lockhart/work/valory/repos/open-aea/packages/packages.json"),
    Path("/home/lockhart/work/valory/repos/mech-interact/packages/packages.json"),
    Path("/home/lockhart/work/valory/repos/funds-manager/packages/packages.json"),
    Path("/home/lockhart/work/valory/genai/packages/packages.json"),
    Path("/home/lockhart/work/valory/repos/kv-store/packages/packages.json"),
]

# Target: this repo's packages.json
TARGET_PACKAGES_JSON = Path("packages/packages.json")


def main() -> None:
    """Compare hashes."""
    # Merge all source dev + third_party packages
    source_all = {}
    for repo_path in SOURCE_REPOS:
        if not repo_path.exists():
            print(f"WARNING: {repo_path} not found, skipping")
            continue
        with open(repo_path, encoding="utf-8") as f:
            source = json.load(f)
        source_all.update(source.get("third_party", {}))
        source_all.update(source.get("dev", {}))

    with open(TARGET_PACKAGES_JSON, encoding="utf-8") as f:
        target = json.load(f)

    target_third = target.get("third_party", {})

    mismatches = []
    missing = []
    for pkg, target_hash in target_third.items():
        if pkg in source_all:
            source_hash = source_all[pkg]
            if target_hash != source_hash:
                mismatches.append((pkg, target_hash, source_hash))
        else:
            missing.append(pkg)

    if not mismatches and not missing:
        print("All hashes match!")
        return

    if mismatches:
        print(f"Found {len(mismatches)} mismatched hashes:\n")
        for pkg, _, new in mismatches:
            print(f'  "{pkg}": "{new}",')
        print(f"\nReplace these in {TARGET_PACKAGES_JSON}")

    if missing:
        print(f"\n{len(missing)} packages not found in any source repo:")
        for pkg in missing:
            print(f"  {pkg}")


if __name__ == "__main__":
    main()
