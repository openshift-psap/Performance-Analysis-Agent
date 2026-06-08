"""Red Flags working memory — ephemeral markdown file manager.

Red flags are cross-config observations that may affect other configurations
of the same model or all models. They live in markdown files named
`red_flags_{version}.md` and are organized by model section.

Lifecycle:
  1. Created by the first config analysis in a version run
  2. Appended to after each config analysis
  3. Read by subsequent config analyses (model + ALL MODELS sections)
  4. Folded into persistent facts during consolidation
  5. Archived after consolidation completes
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_DEFAULT_RED_FLAGS_DIR = Path("red_flags")
ALL_MODELS_SECTION = "ALL MODELS"


class RedFlagsManager:
    """Manage red_flags_{version}.md files."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_RED_FLAGS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, version: str) -> Path:
        return self.base_dir / f"red_flags_{version}.md"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_file(self, version: str) -> str:
        """Read the entire red flags file for a version. Returns '' if absent."""
        path = self._file_path(version)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def parse_sections(self, version: str) -> dict[str, list[str]]:
        """Parse the red flags file into {section_name: [flag_lines]}.

        Section names are the model names (e.g. 'synthcorp/Titan-72B')
        or 'ALL MODELS'.
        """
        content = self.read_file(version)
        if not content:
            return {}

        sections: dict[str, list[str]] = {}
        current_section: Optional[str] = None

        for line in content.splitlines():
            # Match section headers: ### model_name or ### ALL MODELS
            header_match = re.match(r"^###\s+(.+)$", line.strip())
            if header_match:
                current_section = header_match.group(1).strip()
                sections.setdefault(current_section, [])
                continue

            # Skip the top-level header (## Red Flags - ...)
            if line.strip().startswith("## Red Flags"):
                continue

            # Collect flag lines under the current section
            if current_section and line.strip().startswith("- "):
                sections[current_section].append(line.strip())
            elif current_section and line.strip() and not line.strip().startswith("#"):
                # Continuation line of a multi-line flag
                if sections[current_section]:
                    sections[current_section][-1] += "\n  " + line.strip()

        return sections

    def get_relevant_flags(self, version: str, model_name: str) -> str:
        """Get red flags relevant to a specific model.

        Returns flags from the model's own section + the ALL MODELS section,
        formatted as a string.
        """
        sections = self.parse_sections(version)
        if not sections:
            return ""

        relevant_lines: list[str] = []

        # Model-specific flags
        model_flags = sections.get(model_name, [])
        if model_flags:
            relevant_lines.extend(model_flags)

        # ALL MODELS flags
        all_flags = sections.get(ALL_MODELS_SECTION, [])
        if all_flags:
            relevant_lines.extend(all_flags)

        return "\n".join(relevant_lines)

    # ------------------------------------------------------------------
    # Write / Append
    # ------------------------------------------------------------------

    def create_file(self, version: str) -> Path:
        """Create an empty red flags file for a version (idempotent)."""
        path = self._file_path(version)
        if not path.exists():
            path.write_text(
                f"## Red Flags - {version}\n\n",
                encoding="utf-8",
            )
            logger.info(f"Created red flags file: {path}")
        return path

    def append_flags(
        self,
        version: str,
        model_name: str,
        flags: list[str],
    ) -> None:
        """Append red flag entries under a model section.

        Creates the file and section header if they don't exist.
        Deduplicates flags within the same section.
        """
        if not flags:
            return

        path = self.create_file(version)
        content = path.read_text(encoding="utf-8")
        sections = self.parse_sections(version)

        existing_flags = set(sections.get(model_name, []))
        new_flags = [f for f in flags if f"- {f}" not in existing_flags and f not in existing_flags]

        if not new_flags:
            logger.info(f"No new red flags to add for {model_name} in {version}")
            return

        section_header = f"### {model_name}"
        formatted_flags = "\n".join(f"- {f}" for f in new_flags)

        if section_header in content:
            # Find the section and append after the last flag in it
            lines = content.splitlines()
            insert_idx = None
            in_section = False
            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    in_section = True
                    insert_idx = i + 1
                    continue
                if in_section:
                    if line.strip().startswith("###") or line.strip().startswith("## "):
                        break
                    if line.strip():
                        insert_idx = i + 1

            if insert_idx is not None:
                lines.insert(insert_idx, formatted_flags)
                content = "\n".join(lines) + "\n"
        else:
            # Add new section at the end (but before ALL MODELS if it exists)
            if model_name == ALL_MODELS_SECTION:
                content += f"\n{section_header}\n{formatted_flags}\n"
            else:
                all_models_marker = f"### {ALL_MODELS_SECTION}"
                if all_models_marker in content:
                    content = content.replace(
                        all_models_marker,
                        f"{section_header}\n{formatted_flags}\n\n{all_models_marker}",
                    )
                else:
                    content += f"\n{section_header}\n{formatted_flags}\n"

        path.write_text(content, encoding="utf-8")
        logger.info(
            f"Appended {len(new_flags)} red flags for {model_name} in {version}"
        )

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def archive(self, version: str) -> Optional[Path]:
        """Archive the red flags file after consolidation.

        Moves the file to red_flags/archive/red_flags_{version}_{timestamp}.md
        """
        path = self._file_path(version)
        if not path.exists():
            return None

        archive_dir = self.base_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"red_flags_{version}_{timestamp}.md"
        shutil.move(str(path), str(archive_path))
        logger.info(f"Archived red flags: {path} -> {archive_path}")
        return archive_path

    def list_versions(self) -> list[str]:
        """List all versions that have active red flags files."""
        versions = []
        for f in self.base_dir.glob("red_flags_*.md"):
            # Extract version from filename: red_flags_BENCH-2.5.md -> BENCH-2.5
            match = re.match(r"red_flags_(.+)\.md$", f.name)
            if match:
                versions.append(match.group(1))
        return sorted(versions)
