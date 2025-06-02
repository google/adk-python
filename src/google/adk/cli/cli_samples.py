# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import click
from pathlib import Path
import tempfile
import os
import shutil
import requests
import zipfile
import io
import re  # For extracting description
from typing import Optional, Dict, List, TypedDict

import logging

logger = logging.getLogger("google_adk." + __name__)


# --- Configuration ---
class SampleSourceInfo(TypedDict):
    name: str  # A friendly name for the source (e.g., "Official ADK Samples")
    zip_url: str
    samples_path_in_repo: str  # e.g., "adk-python-main/contributing/samples"


DEFAULT_SAMPLE_SOURCES: List[SampleSourceInfo] = [
    {
        "name": "Official ADK Samples",
        "zip_url": "https://github.com/google/adk-python/archive/refs/heads/main.zip",
        "samples_path_in_repo": "adk-python-main/contributing/samples",
    },
    
    # Add more sources here in the future if needed
    # {
    #     "name": "Community Samples (Example)",
    #     "zip_url": "https://github.com/some-community/adk-samples/archive/refs/heads/main.zip",
    #     "samples_path_in_repo": "adk-samples-main/samples",
    # }
]

ADK_CACHE_SUBDIR_NAME = ".adk"
SAMPLES_CONTAINER_DIR_NAME = "adk-samples" # Directory where samples will be prepared


def _get_samples_cache_dir(user_path_str: Optional[str] = None) -> Path:
    if user_path_str:
        samples_cache_dir = Path(user_path_str)
        click.echo(f"Using custom cache directory: {samples_cache_dir}")
    else:
        home_dir = Path.home()
        adk_app_dir = home_dir / ADK_CACHE_SUBDIR_NAME
        cache_base_dir = adk_app_dir / "cache"
        samples_cache_dir = cache_base_dir / "samples"
    try:
        samples_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        click.secho(
            f"Error: Could not create cache directory at '{samples_cache_dir}'. ({e})",
            fg="red",
        )
        raise
    return samples_cache_dir


def _download_and_extract_source(
    source_info: SampleSourceInfo,
    source_cache_dir: Path,  # Specific cache dir for this source
):
    """Downloads and extracts samples from a single source into its dedicated cache subdir."""
    click.echo(
        f"Downloading samples from '{source_info['name']}' ({source_info['zip_url']})..."
    )
    try:
        with requests.Session() as session:
            response = session.get(source_info["zip_url"], stream=True)
            response.raise_for_status()

            click.echo(
                f"Extracting samples from '{source_info['name']}' to local cache..."
            )
            if source_cache_dir.exists():  # Clear only this source's cache subdir
                for item in source_cache_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            source_cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for member_info in zf.infolist():
                    if (
                        member_info.filename.startswith(
                            source_info["samples_path_in_repo"] + "/"
                        )
                        and not member_info.is_dir()
                    ):
                        relative_path_in_zip = Path(member_info.filename)
                        try:
                            path_parts_after_samples_root = relative_path_in_zip.parts[
                                len(Path(source_info["samples_path_in_repo"]).parts) :
                            ]
                            if not path_parts_after_samples_root:
                                continue
                            target_path = source_cache_dir.joinpath(
                                *path_parts_after_samples_root
                            )
                        except IndexError:
                            logger.warning(
                                f"Could not determine target path for {member_info.filename}, skipping."
                            )
                            continue
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with (
                            zf.open(member_info) as source_file,
                            open(target_path, "wb") as target_file,
                        ):
                            shutil.copyfileobj(source_file, target_file)
        click.secho(
            f"Samples from '{source_info['name']}' cached successfully.", fg="green"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading from {source_info['name']}: {e}")
        click.secho(
            f"Error: Could not download samples from {source_info['name']}. ({e})",
            fg="red",
        )
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Error processing zip from {source_info['name']}: {e}")
        click.secho(
            f"Error: Archive from {source_info['name']} was corrupted. ({e})", fg="red"
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error with source {source_info['name']}: {e}")
        click.secho(
            f"An unexpected error occurred with {source_info['name']}: {e}", fg="red"
        )
        raise


def _is_source_cache_populated(source_cache_dir: Path) -> bool:
    """Checks if a specific source's cache subdirectory seems populated."""
    return source_cache_dir.exists() and any(source_cache_dir.iterdir())


def _ensure_all_samples_cached(
    user_cache_path_str: Optional[str] = None,
    sources: List[SampleSourceInfo] = DEFAULT_SAMPLE_SOURCES,
) -> Optional[Path]:
    """
    Ensures samples from all sources are cached.
    Each source will be in a subdirectory of the main cache_dir.
    Returns the main cache_dir path or None if any source fails.
    """
    main_cache_dir = _get_samples_cache_dir(user_cache_path_str)

    for source_info in sources:
        # Create a slug for the source name to use as a directory name
        # e.g., "Official ADK Samples" -> "official_adk_samples"
        source_slug = re.sub(r"\s+", "_", source_info["name"].lower())
        source_slug = re.sub(
            r"[^a-z0-9_]", "", source_slug
        )  # Keep only alphanumeric and underscore
        source_specific_cache_dir = main_cache_dir / source_slug

        if not _is_source_cache_populated(source_specific_cache_dir):
            click.echo(f"Cache for '{source_info['name']}' is missing or incomplete.")
            try:
                _download_and_extract_source(source_info, source_specific_cache_dir)
            except Exception:
                # Error already printed by _download_and_extract_source
                return None  # Indicate failure for this source
    return main_cache_dir


def _extract_description_from_agent_py(agent_py_path: Path) -> str:
    """
    Extracts the value assigned to a 'description' variable in an agent.py file.
    Handles single-line, multi-line strings, and assignments with parentheses.
    """
    try:
        with open(agent_py_path, "r", encoding="utf-8") as f:
            content = f.read()
            patterns_to_try = [
                # Triple-quoted strings (multi-line)
                r"""description\s*=\s*\(?\s*\"\"\"(.*?)\"\"\"""",
                r"""description\s*=\s*\(?\s*'''(.*?)'''""",
                # Single-quoted strings (can be multi-line if within parentheses)
                r"""description\s*=\s*\(?\s*['"](.*?)['"]""",
            ]

            description_text = None
            for pattern in patterns_to_try:
                # re.DOTALL makes . match newlines as well, crucial for multi-line strings
                # re.MULTILINE is not strictly needed here as we are searching the whole content
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    description_text = match.group(1).strip()
                    break  # Found a match, use it

            if description_text:
                # Take the first line of a multi-line description for the table
                first_line = description_text.splitlines()[0].strip()
                return first_line
            else:
                logger.info(f"No 'description' assignment found in {agent_py_path}")

    except FileNotFoundError:
        logger.warning(f"Agent file not found: {agent_py_path}")
    except Exception as e:
        # Log the original exception message and the specific file
        logger.warning(
            f"Could not read or parse description from {agent_py_path}: {type(e).__name__} - {e}"
        )
    return "N/A - Description not available"


def _get_available_samples_from_cache(
    main_cache_dir: Path, sources: List[SampleSourceInfo] = DEFAULT_SAMPLE_SOURCES
) -> Dict[
    str, Dict[str, any]
]:  # Returns dict: {sample_name: {path: Path, description: str, source: str}}
    """
    Scans all source subdirectories in the main_cache_dir.
    Returns a dictionary of unique sample names to their details.
    Handles potential name clashes by preferring the first source in the list.
    """
    available_samples: Dict[str, Dict[str, any]] = {}
    processed_sample_names = set()

    for source_info in sources:
        source_slug = re.sub(r"\s+", "_", source_info["name"].lower())
        source_slug = re.sub(r"[^a-z0-9_]", "", source_slug)
        source_specific_cache_dir = main_cache_dir / source_slug

        if not source_specific_cache_dir.is_dir():
            continue

        for item in source_specific_cache_dir.iterdir():
            if item.is_dir() and item.name not in processed_sample_names:
                agent_py = item / "agent.py"
                init_py = item / "__init__.py"
                if agent_py.is_file() and init_py.is_file():
                    description = _extract_description_from_agent_py(agent_py)
                    available_samples[item.name] = {
                        "path": item,
                        "description": description,
                        "source_name": source_info["name"],
                    }
                    processed_sample_names.add(item.name)
                else:
                    logger.info(
                        f"Skipping '{item.name}' from '{source_info['name']}', missing key files."
                    )
    return available_samples


def _print_samples_table(samples_data: Dict[str, Dict[str, any]]):
    """Prints samples in a formatted table-like structure."""
    if not samples_data:
        click.secho("No samples found in the local cache.", fg="yellow")
        return

    click.echo("\nAvailable ADK Samples:")
    click.echo(
        "+-----+--------------------------------+--------------------------------------------------+"
    )
    click.echo(
        "| No. | Sample Name                    | Description                                      |"
    )
    click.echo(
        "+-----+--------------------------------+--------------------------------------------------+"
    )

    sorted_sample_names = sorted(list(samples_data.keys()))
    max_name_len = 28  # Max length for sample name column
    max_desc_len = 48  # Max length for description column

    for i, name in enumerate(sorted_sample_names):
        data = samples_data[name]
        desc = data["description"]
        # Truncate if too long
        display_name = (
            (name[: max_name_len - 3] + "...") if len(name) > max_name_len else name
        )
        display_desc = (
            (desc[: max_desc_len - 3] + "...") if len(desc) > max_desc_len else desc
        )

        click.echo(
            f"| {i + 1:<3} | {display_name:<{max_name_len}} | {display_desc:<{max_desc_len}} |"
        )
    click.echo(
        "+-----+--------------------------------+--------------------------------------------------+\n"
    )


def _list_samples_interactive(
    available_samples_in_cache: Dict[str, Dict[str, any]],
) -> Optional[str]:
    _print_samples_table(available_samples_in_cache)
    if not available_samples_in_cache:
        return None

    sample_names = sorted(list(available_samples_in_cache.keys()))
    while True:
        try:
            choice_str = click.prompt(
                "Enter the number of the sample to prepare (or 'q' to quit)", type=str
            )
            if choice_str.lower() == "q":
                return None
            choice = int(choice_str)
            if 1 <= choice <= len(sample_names):
                return sample_names[choice - 1]
            else:
                click.secho("Invalid choice.", fg="red")
        except ValueError:
            click.secho("Invalid input. Please enter a number or 'q'.", fg="red")


def _get_specific_sample_details_from_cache(
    sample_name: str, available_samples_in_cache: Dict[str, Dict[str, any]]
) -> Optional[Dict[str, any]]:
    return available_samples_in_cache.get(sample_name)


def _copy_sample_files(
    source_path_in_cache: Path,
    destination_container_path: Path,
    sample_name: str,
) -> Path:
    destination_sample_dir = destination_container_path / sample_name
    if destination_sample_dir.exists():
        if any(destination_sample_dir.iterdir()):
            if not click.confirm(
                f"Sample directory '{destination_sample_dir.name}' already exists inside "
                f"'{destination_container_path.name}' and is not empty. Overwrite?",
                default=False,
            ):
                click.echo("Preparation aborted by user.")
                raise SystemExit(0)
            else:
                shutil.rmtree(destination_sample_dir)

    destination_sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copytree(
            source_path_in_cache, destination_sample_dir, dirs_exist_ok=True
        )
    except Exception as e:
        logger.error(f"Error copying sample files for '{sample_name}': {e}")
        click.secho(f"Error copying sample files: {e}", fg="red")
        raise
    return destination_sample_dir  # Path to the specific sample's dir (e.g. .../adk-samples/hello_world)


def _create_env_file(sample_dir_path: Path):
    env_file_path = sample_dir_path / ".env"
    if env_file_path.exists():
        if not click.confirm(
            f"'.env' file already exists in '{sample_dir_path.name}'. Overwrite?",
            default=True,
        ):
            click.echo("Skipping .env file creation.")
            return

    api_key = click.prompt(
        "Please enter your Gemini API key", hide_input=True, type=str
    )
    api_key_line = (
        f"GOOGLE_API_KEY={api_key}\n"
        if api_key
        else "# GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE\n"
    )

    try:
        with open(env_file_path, "w") as f:
            f.write(api_key_line)
            f.write("GOOGLE_GENAI_USE_VERTEXAI=0\n")
        click.secho("'.env' file created successfully.", fg="green")
    except IOError as e:
        logger.error(f"Error writing .env file to {env_file_path}: {e}")
        click.secho(f"Error: Could not write .env file. ({e})", fg="red")


def run_samples_command(
    sample_name_arg: Optional[str],
    user_cache_path_str: Optional[str] = None,
    user_output_base_path_str: Optional[str] = None,  # New parameter
):
    click.echo(
        "\n+---------------------------------------------------------------------------+\n"
        "| Welcome to ADK Samples! This tool helps you get started with examples.    |\n"
        "+---------------------------------------------------------------------------+"
    )
    click.echo("Checking for local ADK samples cache...")
    try:
        main_cache_dir = _ensure_all_samples_cached(user_cache_path_str)
        if not main_cache_dir:
            return
    except Exception:
        return

    available_samples_details = _get_available_samples_from_cache(main_cache_dir)
    if not available_samples_details:
        # Error message logic as before, potentially mentioning sources
        click.secho("No valid samples found after checking all sources.", fg="red")
        return

    chosen_sample_name: Optional[str] = None
    chosen_sample_details: Optional[Dict[str, any]] = None

    if sample_name_arg:
        chosen_sample_details = _get_specific_sample_details_from_cache(
            sample_name_arg, available_samples_details
        )
        if not chosen_sample_details:
            click.secho(f"Error: Sample '{sample_name_arg}' not found.", fg="red")
            _print_samples_table(available_samples_details)  # Show available ones
            return
        chosen_sample_name = sample_name_arg
    else:
        chosen_sample_name = _list_samples_interactive(available_samples_details)
        if not chosen_sample_name:
            click.echo("No sample selected. Exiting.")
            return
        chosen_sample_details = available_samples_details[chosen_sample_name]

    if chosen_sample_name and chosen_sample_details:
        source_path_in_cache = chosen_sample_details["path"]

        if user_output_base_path_str:
            output_base_path = Path(user_output_base_path_str)
        else:
            output_base_path = Path.cwd()

        output_container_path = output_base_path / SAMPLES_CONTAINER_DIR_NAME
        output_container_path.mkdir(parents=True, exist_ok=True)

        click.echo(
            f"\nPreparing sample '{chosen_sample_name}' from source '{chosen_sample_details['source_name']}'\n"
            f"Target location: {output_container_path / chosen_sample_name}"
        )
        try:
            prepared_sample_path = _copy_sample_files(
                source_path_in_cache, output_container_path, chosen_sample_name
            )
            _create_env_file(prepared_sample_path)

            click.secho(
                f"\nSample '{chosen_sample_name}' prepared successfully in '{prepared_sample_path}'.",
                fg="green",
            )
            click.echo("\nNext steps:")
            current_working_dir = Path.cwd()
            is_output_relative = False
            try:
                # Try to get a relative path
                relative_container_path_str = str(
                    output_container_path.relative_to(current_working_dir)
                )
                is_output_relative = True
            except ValueError:
                # If not relative, use the absolute path
                relative_container_path_str = str(output_container_path)

            click.echo(f"  1. Navigate to the ADK samples container directory:")
            if is_output_relative and not relative_container_path_str.startswith(".."):
                # Only suggest 'cd' if it's a direct subdirectory or the same directory
                click.echo(f"     cd {relative_container_path_str}")
            else:
                # Otherwise, always show the full path
                click.echo(
                    f'     cd "{output_container_path}"'
                )  # Use quotes for paths with spaces

            click.echo(
                f"  2. Inside this directory, you will find '{chosen_sample_name}/'."
            )
            click.echo(
                f"  3. To run a sample (e.g., '{chosen_sample_name}') from within "
                f"'{SAMPLES_CONTAINER_DIR_NAME}':"
            )  # Refer to the container dir name
            click.echo(f"     adk web")
            click.echo(
                f"     (or 'adk run {chosen_sample_name}', 'adk api_server {chosen_sample_name}')"
            )

            click.echo(
                "\nNote: Some samples might have additional setup. Check their README.md if available."
            )
            click.echo(
                "+---------------------------------------------------------------------------+"
            )
        except SystemExit:
            pass
        except Exception as e:
            click.secho(f"An unexpected error occurred: {e}", fg="red")
