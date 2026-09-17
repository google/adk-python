# Copyright 2026 Google LLC
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

"""Tests for packing live media frames into a single ZIP artifact."""

from __future__ import annotations

import io
import json
import zipfile

from google.adk.live._media_frames import build_manifest
from google.adk.live._media_frames import DEFAULT_MIME_TYPE
from google.adk.live._media_frames import extract_frame
from google.adk.live._media_frames import extract_preview_frame
from google.adk.live._media_frames import FRAMES_DIRECTORY
from google.adk.live._media_frames import MEDIA_ZIP_MIME_TYPE
from google.adk.live._media_frames import MediaFrame
from google.adk.live._media_frames import METADATA_FILENAME
from google.adk.live._media_frames import pack_media_frames
from google.adk.live._media_frames import read_manifest
from google.adk.live._media_frames import summarize_manifest
from google.adk.live._media_frames import unpack_media_frames
from google.genai import types
import pydantic
import pytest


def make_frames(
    count: int = 3,
    *,
    mime_type: str = 'image/jpeg',
    start: float = 100.0,
    step: float = 0.2,
) -> list[MediaFrame]:
  """Builds a frame batch whose bytes identify their index."""
  return [
      MediaFrame(
          blob=types.Blob(
              data=f'frame-{index}-payload'.encode(), mime_type=mime_type
          ),
          timestamp=start + index * step,
      )
      for index in range(count)
  ]


class TestMediaFrame:
  """The model is the contract between the cache manager and the packer."""

  def test_unknown_fields_are_rejected(self) -> None:
    # `extra="forbid"` is what stops a typo'd field name from being silently
    # dropped on its way into the archive.
    with pytest.raises(pydantic.ValidationError):
      MediaFrame(
          blob=types.Blob(data=b'x', mime_type='image/jpeg'),
          timestamp=0.0,
          **{'not_a_field': 1},  # type: ignore[arg-type]
      )

  def test_accepts_the_declared_fields(self) -> None:
    frame = MediaFrame(
        blob=types.Blob(data=b'x', mime_type='image/png'), timestamp=1.5
    )
    assert frame.blob.mime_type == 'image/png'
    assert frame.timestamp == 1.5


class TestRoundTrip:
  """pack and unpack have to be exact inverses, or stored video is lossy."""

  def test_unpack_returns_the_frames_that_were_packed(self) -> None:
    frames = make_frames(5)

    archive_bytes, _ = pack_media_frames(frames)
    restored, _ = unpack_media_frames(archive_bytes)

    assert restored == frames

  def test_round_trip_preserves_mixed_mime_types(self) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'jpeg-bytes', mime_type='image/jpeg'),
            timestamp=1.0,
        ),
        MediaFrame(
            blob=types.Blob(data=b'png-bytes', mime_type='image/png'),
            timestamp=2.0,
        ),
        MediaFrame(
            blob=types.Blob(data=b'webp-bytes', mime_type='image/webp'),
            timestamp=3.0,
        ),
    ]

    archive_bytes, _ = pack_media_frames(frames)
    restored, _ = unpack_media_frames(archive_bytes)

    assert [frame.blob.mime_type for frame in restored] == [
        'image/jpeg',
        'image/png',
        'image/webp',
    ]
    assert restored == frames

  def test_round_trip_preserves_a_single_frame(self) -> None:
    frames = make_frames(1)

    archive_bytes, _ = pack_media_frames(frames)
    restored, _ = unpack_media_frames(archive_bytes)

    assert restored == frames

  def test_round_trip_preserves_frame_order_for_a_long_sequence(self) -> None:
    """Members are named with a zero-padded index precisely so that frame 10
    does not sort before frame 2."""
    frames = make_frames(25, step=0.04)

    archive_bytes, _ = pack_media_frames(frames)
    restored, _ = unpack_media_frames(archive_bytes)

    assert [frame.blob.data for frame in restored] == [
        frame.blob.data for frame in frames
    ]

  def test_timestamps_round_trip_to_millisecond_precision(self) -> None:
    """Capture times are stored as whole-millisecond offsets from the start,
    so a round trip quantises them. Recording the limit here keeps a caller
    from assuming microsecond fidelity it will not get."""
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'a', mime_type='image/jpeg'),
            timestamp=100.00049,
        ),
        MediaFrame(
            blob=types.Blob(data=b'b', mime_type='image/jpeg'),
            timestamp=100.30051,
        ),
    ]

    archive_bytes, _ = pack_media_frames(frames)
    restored, _ = unpack_media_frames(archive_bytes)

    assert restored[0].timestamp == pytest.approx(100.0, abs=1e-3)
    assert restored[1].timestamp == pytest.approx(100.3, abs=1e-3)


class TestArchiveLayout:
  """The on-disk shape is a contract with any reader that is not this module."""

  def test_archive_holds_the_frames_and_the_manifest(self) -> None:
    frames = make_frames(3)

    archive_bytes, _ = pack_media_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      assert sorted(archive.namelist()) == [
          f'{FRAMES_DIRECTORY}/frame_0000.jpeg',
          f'{FRAMES_DIRECTORY}/frame_0001.jpeg',
          f'{FRAMES_DIRECTORY}/frame_0002.jpeg',
          METADATA_FILENAME,
      ]
      for index, frame in enumerate(frames):
        member = f'{FRAMES_DIRECTORY}/frame_{index:04d}.jpeg'
        assert archive.read(member) == frame.blob.data

  def test_frames_are_stored_uncompressed(self) -> None:
    """Frames arrive already JPEG/PNG encoded, so deflate would spend CPU per
    frame for no size win, and storing keeps members individually readable."""
    archive_bytes, _ = pack_media_frames(make_frames(3))

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      for info in archive.infolist():
        assert info.compress_type == zipfile.ZIP_STORED

  def test_manifest_member_is_valid_json(self) -> None:
    archive_bytes, manifest = pack_media_frames(make_frames(2))

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      written = json.loads(archive.read(METADATA_FILENAME).decode('utf-8'))

    assert written == manifest

  @pytest.mark.parametrize(
      'mime_type,expected_extension',
      [
          ('image/jpeg', 'jpeg'),
          # jpg and jpeg are the same encoding; collapsing them keeps one
          # sequence from mixing two names for one format.
          ('image/jpg', 'jpeg'),
          ('image/png', 'png'),
          ('image/webp', 'webp'),
          ('image/gif', 'gif'),
          ('image/svg+xml', 'svg'),
          ('video/mp4', 'mp4'),
          ('IMAGE/PNG', 'png'),
          ('image/png;codecs=foo', 'png'),
          ('image/png ', 'png'),
          # Anything that is not a recognized MIME type falls back rather than
          # failing the batch: an odd mime type is not worth losing video over.
          ('notamimetype', 'jpeg'),
          ('image/', 'jpeg'),
          ('', 'jpeg'),
          (None, 'jpeg'),
      ],
  )
  def test_member_extension_follows_the_frame_mime_type(
      self, mime_type: str | None, expected_extension: str
  ) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'x', mime_type=mime_type), timestamp=0.0
        )
    ]

    archive_bytes, _ = pack_media_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      assert archive.namelist()[0] == (
          f'{FRAMES_DIRECTORY}/frame_0000.{expected_extension}'
      )

  @pytest.mark.parametrize(
      'malicious_mime',
      [
          'image/../../etc/passwd',
          'image/..\\..\\etc\\passwd',
          'image/..',
      ],
  )
  def test_a_mime_type_carrying_separators_cannot_escape_the_frames_directory(
      self, malicious_mime: str
  ) -> None:
    """The mime type arrives from a live model stream, so it is untrusted
    input that ends up in a member name."""
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'x', mime_type=malicious_mime),
            timestamp=0.0,
        )
    ]

    archive_bytes, _ = pack_media_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      member = archive.namelist()[0]
    assert member == f'{FRAMES_DIRECTORY}/frame_0000.jpeg'
    assert '..' not in member
    assert '\\' not in member
    assert member.count('/') == 1


class TestManifest:
  """The manifest is what a reader consults before downloading anything."""

  def test_manifest_describes_the_sequence(self) -> None:
    frames = make_frames(5, start=10.0, step=0.25)

    manifest = build_manifest(frames)

    assert manifest['type'] == 'video_frame_sequence'
    assert manifest['frameCount'] == 5
    assert manifest['startTimestampMs'] == 10000
    assert manifest['endTimestampMs'] == 11000
    assert manifest['durationMs'] == 1000
    assert manifest['estimatedFps'] == 4.0

  def test_manifest_describes_every_frame(self) -> None:
    frames = make_frames(4, start=10.0, step=0.25)

    manifest = build_manifest(frames)

    assert [entry['frameIndex'] for entry in manifest['frames']] == [0, 1, 2, 3]
    assert [entry['offsetMs'] for entry in manifest['frames']] == [
        0,
        250,
        500,
        750,
    ]
    assert [entry['fileName'] for entry in manifest['frames']] == [
        f'{FRAMES_DIRECTORY}/frame_{index:04d}.jpeg' for index in range(4)
    ]
    assert [entry['mimeType'] for entry in manifest['frames']] == [
        'image/jpeg'
    ] * 4
    assert [entry['sizeBytes'] for entry in manifest['frames']] == [
        len(frame.blob.data or b'') for frame in frames
    ]

  def test_single_frame_reports_no_frame_rate(self) -> None:
    """One frame spans no interval, so a rate would be a division by zero
    dressed up as data."""
    manifest = build_manifest(make_frames(1))

    assert manifest['frameCount'] == 1
    assert manifest['durationMs'] == 0
    assert manifest['estimatedFps'] == 0.0

  def test_frames_sharing_one_timestamp_report_no_frame_rate(self) -> None:
    """Frames captured inside the same clock tick are legitimate and must not
    divide by a zero interval."""
    manifest = build_manifest(make_frames(3, step=0.0))

    assert manifest['durationMs'] == 0
    assert manifest['estimatedFps'] == 0.0
    assert [entry['offsetMs'] for entry in manifest['frames']] == [0, 0, 0]

  def test_frame_rate_counts_intervals_not_frames(self) -> None:
    """Five frames over one second is four intervals, so 4fps, not 5."""
    manifest = build_manifest(make_frames(5, start=0.0, step=0.25))

    assert manifest['estimatedFps'] == 4.0

  def test_frame_without_a_mime_type_is_recorded_as_the_default(self) -> None:
    frames = [
        MediaFrame(blob=types.Blob(data=b'x', mime_type=None), timestamp=0.0)
    ]

    manifest = build_manifest(frames)

    assert manifest['frames'][0]['mimeType'] == DEFAULT_MIME_TYPE

  def test_mime_types_are_recorded_lowercased(self) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'x', mime_type='IMAGE/PNG'), timestamp=0.0
        )
    ]

    manifest = build_manifest(frames)

    assert manifest['frames'][0]['mimeType'] == 'image/png'

  def test_empty_frame_list_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='empty frame list'):
      build_manifest([])

  def test_decreasing_timestamps_are_rejected(self) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'a', mime_type='image/jpeg'), timestamp=10.0
        ),
        MediaFrame(
            blob=types.Blob(data=b'b', mime_type='image/jpeg'), timestamp=9.5
        ),
    ]

    with pytest.raises(ValueError, match='must be non-decreasing'):
      build_manifest(frames)

  def test_custom_metadata_is_merged_in(self) -> None:
    manifest = build_manifest(
        make_frames(2), custom_metadata={'source': 'webcam'}
    )

    assert manifest['source'] == 'webcam'
    assert manifest['frameCount'] == 2

  def test_custom_metadata_cannot_override_computed_keys(self) -> None:
    """Computed structural keys (`startTimestampMs`, `frameCount`, `frames`,
    etc.) take precedence over `custom_metadata` so unpacked timestamps and
    members cannot be shifted or corrupted."""
    frames = make_frames(2, start=100.0, step=0.2)

    archive_bytes, manifest = pack_media_frames(
        frames,
        custom_metadata={
            'source': 'webcam',
            'frameCount': 999,
            'startTimestampMs': 0,
        },
    )

    assert manifest['source'] == 'webcam'
    assert manifest['frameCount'] == 2
    assert manifest['startTimestampMs'] == 100000
    restored, _ = unpack_media_frames(archive_bytes)
    assert restored == frames
    assert extract_frame(archive_bytes, 1) == frames[1]

  def test_overridden_frames_key_cannot_misname_or_break_unpack(self) -> None:
    """`custom_metadata` cannot override `frames`, so the archive always
    records and unpacks the packed members."""
    frames = make_frames(2)

    archive_bytes, manifest = pack_media_frames(
        frames, custom_metadata={'frames': [{'fileName': 'evil/path.jpeg'}]}
    )

    assert [entry['fileName'] for entry in manifest['frames']] == [
        f'{FRAMES_DIRECTORY}/frame_0000.jpeg',
        f'{FRAMES_DIRECTORY}/frame_0001.jpeg',
    ]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      names = archive.namelist()
    assert f'{FRAMES_DIRECTORY}/frame_0000.jpeg' in names
    assert f'{FRAMES_DIRECTORY}/frame_0001.jpeg' in names
    assert 'evil/path.jpeg' not in names
    restored, unpacked_manifest = unpack_media_frames(archive_bytes)
    assert restored == frames
    assert unpacked_manifest == manifest
    assert extract_frame(archive_bytes, 0) == frames[0]


class TestSummarizeManifest:
  """What travels in artifact metadata has to stay small and bounded."""

  def test_summary_drops_the_per_frame_index(self) -> None:
    manifest = build_manifest(make_frames(3))

    summary = summarize_manifest(manifest)

    assert 'frames' not in summary
    assert summary['frameCount'] == 3
    assert summary['type'] == 'video_frame_sequence'
    assert summary['durationMs'] == manifest['durationMs']
    assert summary['estimatedFps'] == manifest['estimatedFps']

  def test_summary_does_not_mutate_the_manifest(self) -> None:
    manifest = build_manifest(make_frames(3))

    summarize_manifest(manifest)

    assert len(manifest['frames']) == 3

  def test_summary_size_does_not_grow_with_the_frame_count(self) -> None:
    """Several artifact backends keep custom_metadata in object metadata with
    a few-kilobyte cap, which a per-frame index would blow through within a
    minute of video."""
    short_summary = summarize_manifest(build_manifest(make_frames(2)))
    long_summary = summarize_manifest(build_manifest(make_frames(600)))

    assert short_summary.keys() == long_summary.keys()
    assert len(json.dumps(long_summary).encode()) < 1024


class TestPackValidation:
  """Nothing half-formed should reach the archive."""

  def test_empty_frame_list_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='empty frame list'):
      pack_media_frames([])

  def test_decreasing_timestamps_are_rejected(self) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=b'a', mime_type='image/jpeg'), timestamp=5.0
        ),
        MediaFrame(
            blob=types.Blob(data=b'b', mime_type='image/jpeg'), timestamp=4.9
        ),
    ]

    with pytest.raises(ValueError, match='must be non-decreasing'):
      pack_media_frames(frames)

  def test_frame_with_no_bytes_is_rejected(self) -> None:
    frames = make_frames(3)
    frames[1] = MediaFrame(
        blob=types.Blob(data=b'', mime_type='image/jpeg'), timestamp=1.0
    )

    with pytest.raises(ValueError, match='Frame 1 must contain non-empty'):
      pack_media_frames(frames)

  def test_frame_with_none_bytes_is_rejected(self) -> None:
    frames = [
        MediaFrame(
            blob=types.Blob(data=None, mime_type='image/jpeg'), timestamp=1.0
        )
    ]

    with pytest.raises(ValueError, match='Frame 0 must contain non-empty'):
      pack_media_frames(frames)


class TestUnpackValidation:
  """A corrupt archive must say so rather than return partial video."""

  def test_non_zip_input_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='not a valid ZIP'):
      unpack_media_frames(b'this is not a zip file')

  def test_archive_without_a_manifest_is_rejected(self) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_0000.jpeg', b'x')

    with pytest.raises(ValueError, match=f'missing {METADATA_FILENAME}'):
      unpack_media_frames(buffer.getvalue())

  def test_manifest_naming_an_absent_member_is_rejected(self) -> None:
    """Silently skipping the missing frame would hand back a sequence shorter
    than its own manifest claims."""
    frames = make_frames(3)
    archive_bytes, manifest = pack_media_frames(frames)

    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(rebuilt, mode='w') as target:
        for name in source.namelist():
          if name.endswith('frame_0001.jpeg'):
            continue
          target.writestr(name, source.read(name))

    with pytest.raises(ValueError, match='missing frame member'):
      unpack_media_frames(rebuilt.getvalue())

  def test_non_dict_manifest_raises_value_error_across_all_readers(
      self,
  ) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_0000.jpeg', b'x')
      archive.writestr(METADATA_FILENAME, json.dumps([1, 2, 3]))
    corrupt_bytes = buffer.getvalue()

    with pytest.raises(ValueError, match='must contain a JSON object'):
      unpack_media_frames(corrupt_bytes)
    with pytest.raises(ValueError, match='must contain a JSON object'):
      read_manifest(corrupt_bytes)
    with pytest.raises(ValueError, match='must contain a JSON object'):
      extract_frame(corrupt_bytes, 0)

  def test_bad_member_crc_raises_value_error_across_all_readers(self) -> None:
    """`ZipFile(...)` only validates the central directory; a corrupt member
    payload raises `BadZipFile` later inside `archive.read()`, which must still
    surface as the documented `ValueError`."""
    archive_bytes, _ = pack_media_frames(make_frames(2))
    corrupted = bytearray(archive_bytes)
    payload_offset = corrupted.index(b'frame-0-payload')
    corrupted[payload_offset] ^= 0xFF
    corrupt_frame_bytes = bytes(corrupted)

    with pytest.raises(ValueError, match='not a valid ZIP'):
      unpack_media_frames(corrupt_frame_bytes)
    with pytest.raises(ValueError, match='not a valid ZIP'):
      extract_frame(corrupt_frame_bytes, 0)

    corrupted_manifest = bytearray(archive_bytes)
    manifest_offset = corrupted_manifest.index(b'video_frame_sequence')
    corrupted_manifest[manifest_offset] ^= 0xFF
    corrupt_manifest_bytes = bytes(corrupted_manifest)

    with pytest.raises(ValueError, match='not a valid ZIP'):
      read_manifest(corrupt_manifest_bytes)


class TestReadManifest:
  """Reading the shape of a sequence should not cost the whole sequence."""

  def test_read_manifest_matches_the_packed_manifest(self) -> None:
    archive_bytes, manifest = pack_media_frames(make_frames(4))

    assert read_manifest(archive_bytes) == manifest

  def test_non_zip_input_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='not a valid ZIP'):
      read_manifest(b'not a zip')

  def test_archive_without_a_manifest_is_rejected(self) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('unrelated.txt', b'x')

    with pytest.raises(ValueError, match=f'missing {METADATA_FILENAME}'):
      read_manifest(buffer.getvalue())


class TestExtractFrame:
  """Random access is the reason for ZIP over TAR."""

  def test_extract_returns_the_requested_frame(self) -> None:
    frames = make_frames(5)
    archive_bytes, _ = pack_media_frames(frames)

    assert extract_frame(archive_bytes, 2) == frames[2]

  def test_preview_is_the_first_frame(self) -> None:
    frames = make_frames(4)
    archive_bytes, _ = pack_media_frames(frames)

    assert extract_preview_frame(archive_bytes) == frames[0]

  def test_extract_reads_only_the_requested_member(self) -> None:
    """A reader wanting a thumbnail should not have to hold every frame in
    memory; the central directory is what makes that possible. An archive
    stripped of every other frame still yields frame 0."""
    frames = make_frames(5)
    archive_bytes, _ = pack_media_frames(frames)

    trimmed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(trimmed, mode='w') as target:
        target.writestr(METADATA_FILENAME, source.read(METADATA_FILENAME))
        first = f'{FRAMES_DIRECTORY}/frame_0000.jpeg'
        target.writestr(first, source.read(first))

    assert extract_frame(trimmed.getvalue(), 0) == frames[0]

  @pytest.mark.parametrize('index', [-1, 3, 99])
  def test_out_of_range_index_is_rejected(self, index: int) -> None:
    archive_bytes, _ = pack_media_frames(make_frames(3))

    with pytest.raises(ValueError, match='no frame at index'):
      extract_frame(archive_bytes, index)

  def test_non_zip_input_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='not a valid ZIP'):
      extract_frame(b'not a zip', 0)

  def test_manifest_naming_an_absent_member_is_rejected(self) -> None:
    frames = make_frames(2)
    archive_bytes, _ = pack_media_frames(frames)

    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(rebuilt, mode='w') as target:
        target.writestr(METADATA_FILENAME, source.read(METADATA_FILENAME))

    with pytest.raises(ValueError, match='missing frame member'):
      extract_frame(rebuilt.getvalue(), 0)

  def test_archive_without_a_manifest_is_rejected(self) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_0000.jpeg', b'x')

    with pytest.raises(ValueError, match=f'missing {METADATA_FILENAME}'):
      extract_frame(buffer.getvalue(), 0)

  def test_five_digit_frame_index_preserves_manifest_order(self) -> None:
    """Wide indices (`frame_10000.jpeg` after `frame_9999.jpeg`) are resolved
    directly from the `frames` list in `metadata.json` rather than by
    lexicographic member sorting."""
    manifest = {
        'type': 'video_frame_sequence',
        'frameCount': 3,
        'startTimestampMs': 1000,
        'endTimestampMs': 3000,
        'durationMs': 2000,
        'estimatedFps': 1.0,
        'frames': [
            {
                'frameIndex': 0,
                'offsetMs': 0,
                'fileName': f'{FRAMES_DIRECTORY}/frame_0000.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
            {
                'frameIndex': 9999,
                'offsetMs': 1000,
                'fileName': f'{FRAMES_DIRECTORY}/frame_9999.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
            {
                'frameIndex': 10000,
                'offsetMs': 2000,
                'fileName': f'{FRAMES_DIRECTORY}/frame_10000.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_10000.jpeg', b'f2')
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_9999.jpeg', b'f1')
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_0000.jpeg', b'f0')
      archive.writestr(METADATA_FILENAME, json.dumps(manifest))
    archive_bytes = buffer.getvalue()

    assert [extract_frame(archive_bytes, i).blob.data for i in range(3)] == [
        b'f0',
        b'f1',
        b'f2',
    ]

  def test_non_frame_files_inside_frames_directory_are_ignored(self) -> None:
    frames = make_frames(1)
    archive_bytes, _ = pack_media_frames(frames)

    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(rebuilt, mode='w') as target:
        target.writestr(f'{FRAMES_DIRECTORY}/notes.txt', b'stray note')
        for member_name in source.namelist():
          target.writestr(member_name, source.read(member_name))
    archive_with_stray = rebuilt.getvalue()

    assert extract_frame(archive_with_stray, 0) == frames[0]
    with pytest.raises(ValueError, match='no frame at index'):
      extract_frame(archive_with_stray, 1)

  @pytest.mark.parametrize(
      'bad_manifest,expected_match',
      [
          ({'frames': 'not-a-list'}, 'frames field must be a list'),
          ({'frames': ['not-a-dict']}, 'frame entries must be objects'),
          (
              {
                  'startTimestampMs': 'bad',
                  'frames': [{'fileName': 'frames/frame_0000.jpeg'}],
              },
              'startTimestampMs must be a finite number',
          ),
          (
              {
                  'frames': [{
                      'fileName': 'frames/frame_0000.jpeg',
                      'offsetMs': 'bad',
                  }]
              },
              'offsetMs must be a finite number',
          ),
      ],
  )
  def test_shared_manifest_and_entry_validation_across_readers(
      self,
      bad_manifest: dict[str, object],
      expected_match: str,
  ) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr(f'{FRAMES_DIRECTORY}/frame_0000.jpeg', b'x')
      archive.writestr(METADATA_FILENAME, json.dumps(bad_manifest))
    corrupt_bytes = buffer.getvalue()

    with pytest.raises(ValueError, match=expected_match):
      unpack_media_frames(corrupt_bytes)
    with pytest.raises(ValueError, match=expected_match):
      extract_frame(corrupt_bytes, 0)


def test_zip_mime_type_is_the_one_stored_on_the_artifact() -> None:
  """The declared type has to describe what `load_artifact` hands back, which
  is the archive rather than any frame inside it."""
  assert MEDIA_ZIP_MIME_TYPE == 'application/zip'
