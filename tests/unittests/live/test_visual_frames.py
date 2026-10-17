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

"""Tests for packing live visual frames into a single ZIP artifact."""

from __future__ import annotations

import io
import json
from typing import Any
import zipfile

from google.adk.live._visual_frames import pack_visual_frames
from google.adk.live._visual_frames import unpack_visual_frames
from google.adk.live._visual_frames import VisualFrame
from google.genai import types
import pydantic
import pytest


def make_frames(
    count: int = 3,
    *,
    mime_type: str = 'image/jpeg',
    start: float = 100.0,
    step: float = 0.2,
) -> list[VisualFrame]:
  """Builds a frame batch whose bytes identify their index."""
  return [
      VisualFrame(
          blob=types.Blob(
              data=f'frame-{index}-payload'.encode(), mime_type=mime_type
          ),
          timestamp=start + index * step,
      )
      for index in range(count)
  ]


def written_manifest(archive_bytes: bytes) -> dict[str, Any]:
  """Returns the ``metadata.json`` written into an archive."""
  with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    return json.loads(archive.read('metadata.json').decode('utf-8'))


class TestVisualFrame:
  """The model is the contract between the cache manager and the packer."""

  def test_unknown_fields_are_rejected(self) -> None:
    # `extra="forbid"` is what stops a typo'd field name from being silently
    # dropped on its way into the archive.
    with pytest.raises(pydantic.ValidationError):
      VisualFrame(
          blob=types.Blob(data=b'x', mime_type='image/jpeg'),
          timestamp=0.0,
          **{'not_a_field': 1},  # type: ignore[arg-type]
      )

  def test_accepts_the_declared_fields(self) -> None:
    frame = VisualFrame(
        blob=types.Blob(data=b'x', mime_type='image/png'), timestamp=1.5
    )
    assert frame.blob.mime_type == 'image/png'
    assert frame.timestamp == 1.5

  @pytest.mark.parametrize(
      'timestamp', [float('nan'), float('inf'), float('-inf')]
  )
  def test_non_finite_timestamps_are_rejected(self, timestamp: float) -> None:
    """A NaN or infinite receive time fails when the frame is created."""
    with pytest.raises(pydantic.ValidationError):
      VisualFrame(
          blob=types.Blob(data=b'x', mime_type='image/jpeg'),
          timestamp=timestamp,
      )


class TestRoundTrip:
  """pack and unpack have to be exact inverses, or stored video is lossy."""

  def test_unpack_returns_the_frames_that_were_packed(self) -> None:
    frames = make_frames(5)

    archive_bytes, _ = pack_visual_frames(frames)
    restored, _ = unpack_visual_frames(archive_bytes)

    assert restored == frames

  def test_round_trip_preserves_mixed_mime_types(self) -> None:
    frames = [
        VisualFrame(
            blob=types.Blob(data=b'jpeg-bytes', mime_type='image/jpeg'),
            timestamp=1.0,
        ),
        VisualFrame(
            blob=types.Blob(data=b'png-bytes', mime_type='image/png'),
            timestamp=2.0,
        ),
        VisualFrame(
            blob=types.Blob(data=b'webp-bytes', mime_type='image/webp'),
            timestamp=3.0,
        ),
    ]

    archive_bytes, _ = pack_visual_frames(frames)
    restored, _ = unpack_visual_frames(archive_bytes)

    assert [frame.blob.mime_type for frame in restored] == [
        'image/jpeg',
        'image/png',
        'image/webp',
    ]
    assert restored == frames

  def test_round_trip_preserves_a_single_frame(self) -> None:
    frames = make_frames(1)

    archive_bytes, _ = pack_visual_frames(frames)
    restored, _ = unpack_visual_frames(archive_bytes)

    assert restored == frames

  def test_round_trip_preserves_frame_order_for_a_long_sequence(self) -> None:
    """Frame order survives a sequence of more than ten frames.

    Members are named with a zero-padded index precisely so that frame 10
    does not sort before frame 2.
    """
    frames = make_frames(25, step=0.04)

    archive_bytes, _ = pack_visual_frames(frames)
    restored, _ = unpack_visual_frames(archive_bytes)

    assert [frame.blob.data for frame in restored] == [
        frame.blob.data for frame in frames
    ]

  def test_timestamps_round_trip_to_millisecond_precision(self) -> None:
    """Times come back as the nearest whole millisecond.

    Times are stored as whole milliseconds, so a round trip quantises them.
    Recording the limit here keeps a caller from assuming microsecond fidelity
    it will not get.
    """
    frames = [
        VisualFrame(
            blob=types.Blob(data=b'a', mime_type='image/jpeg'),
            timestamp=100.00049,
        ),
        VisualFrame(
            blob=types.Blob(data=b'b', mime_type='image/jpeg'),
            timestamp=100.30051,
        ),
    ]

    archive_bytes, _ = pack_visual_frames(frames)
    restored, _ = unpack_visual_frames(archive_bytes)

    assert [frame.timestamp for frame in restored] == [100.0, 100.301]


class TestArchiveLayout:
  """The on-disk shape is a contract with any reader that is not this module."""

  def test_archive_holds_the_frames_and_the_manifest(self) -> None:
    frames = make_frames(3)

    archive_bytes, _ = pack_visual_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      assert sorted(archive.namelist()) == [
          'frames/frame_0000.jpeg',
          'frames/frame_0001.jpeg',
          'frames/frame_0002.jpeg',
          'metadata.json',
      ]
      for index, frame in enumerate(frames):
        member = f'frames/frame_{index:04d}.jpeg'
        assert archive.read(member) == frame.blob.data

  def test_frames_are_stored_uncompressed(self) -> None:
    """Every member is stored rather than deflated.

    Frames arrive already JPEG or PNG encoded, so deflating them would spend
    CPU per frame for no size win.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(3))

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      for info in archive.infolist():
        assert info.compress_type == zipfile.ZIP_STORED

  def test_manifest_member_is_valid_json(self) -> None:
    archive_bytes, _ = pack_visual_frames(make_frames(2))

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      written = json.loads(archive.read('metadata.json').decode('utf-8'))

    assert written['type'] == 'video_frame_sequence'
    assert len(written['frames']) == 2

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
        VisualFrame(
            blob=types.Blob(data=b'x', mime_type=mime_type), timestamp=0.0
        )
    ]

    archive_bytes, _ = pack_visual_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      assert archive.namelist()[0] == f'frames/frame_0000.{expected_extension}'

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
    """A MIME type with path separators cannot move a member out of frames/.

    The MIME type arrives from a live model stream, so it is untrusted input
    that ends up in a member name.
    """
    frames = [
        VisualFrame(
            blob=types.Blob(data=b'x', mime_type=malicious_mime),
            timestamp=0.0,
        )
    ]

    archive_bytes, _ = pack_visual_frames(frames)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
      member = archive.namelist()[0]
    assert member == 'frames/frame_0000.jpeg'
    assert '..' not in member
    assert '\\' not in member
    assert member.count('/') == 1


class TestManifest:
  """The manifest is what a reader consults before downloading anything."""

  def test_manifest_records_its_format_version(self) -> None:
    """A reader checks the version to know it understands the archive."""
    archive_bytes, _ = pack_visual_frames(make_frames(2))

    assert written_manifest(archive_bytes)['formatVersion'] == 1

  def test_manifest_describes_the_sequence(self) -> None:
    frames = make_frames(5, start=10.0, step=0.25)

    archive_bytes, _ = pack_visual_frames(frames)

    manifest = written_manifest(archive_bytes)
    assert manifest['type'] == 'video_frame_sequence'
    assert manifest['frameCount'] == 5
    assert manifest['startTimestampMs'] == 10000
    assert manifest['endTimestampMs'] == 11000
    assert manifest['durationMs'] == 1000
    assert manifest['estimatedFps'] == 4.0

  def test_manifest_describes_every_frame(self) -> None:
    frames = make_frames(4, start=10.0, step=0.25)

    archive_bytes, _ = pack_visual_frames(frames)

    manifest = written_manifest(archive_bytes)
    assert [entry['frameIndex'] for entry in manifest['frames']] == [0, 1, 2, 3]
    assert [entry['offsetMs'] for entry in manifest['frames']] == [
        0,
        250,
        500,
        750,
    ]
    assert [entry['fileName'] for entry in manifest['frames']] == [
        f'frames/frame_{index:04d}.jpeg' for index in range(4)
    ]
    assert [entry['mimeType'] for entry in manifest['frames']] == [
        'image/jpeg'
    ] * 4
    assert [entry['sizeBytes'] for entry in manifest['frames']] == [
        len(frame.blob.data or b'') for frame in frames
    ]

  def test_millisecond_fields_agree_with_each_other(self) -> None:
    """The millisecond fields come from the same rounded times, so they agree.

    Frames at 0.4ms and 1.6ms round to 0ms and 2ms, so the duration and the
    last offset are 2ms as well.
    """
    frames = make_frames(2, start=0.0004, step=0.0012)

    archive_bytes, _ = pack_visual_frames(frames)

    manifest = written_manifest(archive_bytes)
    assert manifest['startTimestampMs'] == 0
    assert manifest['endTimestampMs'] == 2
    assert manifest['durationMs'] == 2
    assert [entry['offsetMs'] for entry in manifest['frames']] == [0, 2]

  def test_single_frame_reports_no_frame_rate(self) -> None:
    """A single frame reports a frame rate of 0.0.

    One frame spans no interval, so a rate would be a division by zero
    dressed up as data.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(1))

    manifest = written_manifest(archive_bytes)
    assert manifest['frameCount'] == 1
    assert manifest['durationMs'] == 0
    assert manifest['estimatedFps'] == 0.0

  def test_frames_sharing_one_timestamp_report_no_frame_rate(self) -> None:
    """Frames that share one timestamp report a frame rate of 0.0.

    Frames captured inside the same clock tick are legitimate and must not
    divide by a zero interval.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(3, step=0.0))

    manifest = written_manifest(archive_bytes)
    assert manifest['durationMs'] == 0
    assert manifest['estimatedFps'] == 0.0
    assert [entry['offsetMs'] for entry in manifest['frames']] == [0, 0, 0]

  def test_frame_rate_counts_intervals_not_frames(self) -> None:
    """Five frames over one second is four intervals, so 4fps, not 5."""
    archive_bytes, _ = pack_visual_frames(make_frames(5, start=0.0, step=0.25))

    assert written_manifest(archive_bytes)['estimatedFps'] == 4.0

  def test_frame_without_a_mime_type_is_recorded_as_jpeg(self) -> None:
    frames = [
        VisualFrame(blob=types.Blob(data=b'x', mime_type=None), timestamp=0.0)
    ]

    archive_bytes, _ = pack_visual_frames(frames)

    manifest = written_manifest(archive_bytes)
    assert manifest['frames'][0]['mimeType'] == 'image/jpeg'

  def test_mime_types_are_recorded_lowercased(self) -> None:
    frames = [
        VisualFrame(
            blob=types.Blob(data=b'x', mime_type='IMAGE/PNG'), timestamp=0.0
        )
    ]

    archive_bytes, _ = pack_visual_frames(frames)

    manifest = written_manifest(archive_bytes)
    assert manifest['frames'][0]['mimeType'] == 'image/png'


class TestArtifactMetadata:
  """What travels in artifact metadata has to stay small and bounded."""

  def test_metadata_is_the_manifest_without_its_frame_index(self) -> None:
    archive_bytes, metadata = pack_visual_frames(make_frames(3))

    manifest = written_manifest(archive_bytes)
    assert 'frames' not in metadata
    assert metadata == {
        key: value for key, value in manifest.items() if key != 'frames'
    }

  def test_metadata_size_does_not_grow_with_the_frame_count(self) -> None:
    """The metadata stays the same size however many frames there are.

    Several artifact backends keep custom_metadata in object metadata with a
    few-kilobyte cap, which a per-frame index would blow through within a
    minute of video.
    """
    _, short_metadata = pack_visual_frames(make_frames(2))
    _, long_metadata = pack_visual_frames(make_frames(600))

    assert short_metadata.keys() == long_metadata.keys()
    assert len(json.dumps(long_metadata).encode()) < 1024


class TestPackValidation:
  """Nothing half-formed should reach the archive."""

  def test_empty_frame_list_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='empty frame list'):
      pack_visual_frames([])

  def test_decreasing_timestamps_are_rejected(self) -> None:
    frames = [
        VisualFrame(
            blob=types.Blob(data=b'a', mime_type='image/jpeg'), timestamp=5.0
        ),
        VisualFrame(
            blob=types.Blob(data=b'b', mime_type='image/jpeg'), timestamp=4.9
        ),
    ]

    with pytest.raises(ValueError, match='must be non-decreasing'):
      pack_visual_frames(frames)

  def test_frame_with_no_bytes_is_rejected(self) -> None:
    frames = make_frames(3)
    frames[1] = VisualFrame(
        blob=types.Blob(data=b'', mime_type='image/jpeg'), timestamp=1.0
    )

    with pytest.raises(ValueError, match='Frame 1 must contain non-empty'):
      pack_visual_frames(frames)

  def test_frame_with_none_bytes_is_rejected(self) -> None:
    frames = [
        VisualFrame(
            blob=types.Blob(data=None, mime_type='image/jpeg'), timestamp=1.0
        )
    ]

    with pytest.raises(ValueError, match='Frame 0 must contain non-empty'):
      pack_visual_frames(frames)


class TestUnpackFollowsTheManifest:
  """unpack returns what metadata.json lists, in the order it lists them."""

  def test_five_digit_frame_index_preserves_manifest_order(self) -> None:
    """Frames come back in the order `metadata.json` lists them.

    Wide indices (`frame_10000.jpeg` after `frame_9999.jpeg`) are resolved
    directly from the `frames` list in `metadata.json` rather than by
    lexicographic member sorting.
    """
    manifest = {
        'formatVersion': 1,
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
                'fileName': 'frames/frame_0000.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
            {
                'frameIndex': 9999,
                'offsetMs': 1000,
                'fileName': 'frames/frame_9999.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
            {
                'frameIndex': 10000,
                'offsetMs': 2000,
                'fileName': 'frames/frame_10000.jpeg',
                'mimeType': 'image/jpeg',
                'sizeBytes': 2,
            },
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('frames/frame_10000.jpeg', b'f2')
      archive.writestr('frames/frame_9999.jpeg', b'f1')
      archive.writestr('frames/frame_0000.jpeg', b'f0')
      archive.writestr('metadata.json', json.dumps(manifest))

    restored, _ = unpack_visual_frames(buffer.getvalue())

    assert [frame.blob.data for frame in restored] == [b'f0', b'f1', b'f2']

  def test_non_frame_files_inside_frames_directory_are_ignored(self) -> None:
    frames = make_frames(1)
    archive_bytes, _ = pack_visual_frames(frames)
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(rebuilt, mode='w') as target:
        target.writestr('frames/notes.txt', b'stray note')
        for member_name in source.namelist():
          target.writestr(member_name, source.read(member_name))

    restored, _ = unpack_visual_frames(rebuilt.getvalue())

    assert restored == frames


class TestUnpackValidation:
  """A corrupt archive must say so rather than return partial video."""

  def test_non_zip_input_is_rejected(self) -> None:
    with pytest.raises(ValueError, match='not a valid ZIP'):
      unpack_visual_frames(b'this is not a zip file')

  def test_archive_without_a_manifest_is_rejected(self) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('frames/frame_0000.jpeg', b'x')

    with pytest.raises(ValueError, match='missing metadata.json'):
      unpack_visual_frames(buffer.getvalue())

  def test_manifest_naming_an_absent_member_is_rejected(self) -> None:
    """A manifest that lists a member the archive lacks is rejected.

    Silently skipping the missing frame would hand back a sequence shorter
    than its own manifest claims.
    """
    frames = make_frames(3)
    archive_bytes, _ = pack_visual_frames(frames)

    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      with zipfile.ZipFile(rebuilt, mode='w') as target:
        for name in source.namelist():
          if name.endswith('frame_0001.jpeg'):
            continue
          target.writestr(name, source.read(name))

    with pytest.raises(ValueError, match='missing frame member'):
      unpack_visual_frames(rebuilt.getvalue())

  def test_non_dict_manifest_is_rejected(self) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('frames/frame_0000.jpeg', b'x')
      archive.writestr('metadata.json', json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match='must contain a JSON object'):
      unpack_visual_frames(buffer.getvalue())

  @pytest.mark.parametrize(
      'corrupted_bytes',
      [
          pytest.param(b'frame-0-payload', id='frame_member'),
          pytest.param(b'video_frame_sequence', id='manifest_member'),
      ],
  )
  def test_bad_member_crc_is_rejected(self, corrupted_bytes: bytes) -> None:
    """A member whose bytes fail their CRC check is rejected.

    `ZipFile(...)` only validates the central directory; a corrupt member
    payload raises `BadZipFile` later inside `archive.read()`, which must still
    surface as the documented `ValueError`.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(2))
    corrupted = bytearray(archive_bytes)
    corrupted[corrupted.index(corrupted_bytes)] ^= 0xFF

    with pytest.raises(ValueError, match='not a valid ZIP'):
      unpack_visual_frames(bytes(corrupted))

  @pytest.mark.parametrize(
      'bad_manifest,expected_match',
      [
          (
              {'formatVersion': 1, 'frames': 'not-a-list'},
              'frames field must be a list',
          ),
          (
              {'formatVersion': 1, 'frames': ['not-a-dict']},
              'frame entries must be objects',
          ),
          (
              {
                  'formatVersion': 1,
                  'startTimestampMs': 'bad',
                  'frames': [{'fileName': 'frames/frame_0000.jpeg'}],
              },
              'startTimestampMs must be a finite number',
          ),
          (
              {
                  'formatVersion': 1,
                  'frames': [{
                      'fileName': 'frames/frame_0000.jpeg',
                      'offsetMs': 'bad',
                  }],
              },
              'offsetMs must be a finite number',
          ),
      ],
  )
  def test_malformed_manifest_fields_are_rejected(
      self,
      bad_manifest: dict[str, object],
      expected_match: str,
  ) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('frames/frame_0000.jpeg', b'x')
      archive.writestr('metadata.json', json.dumps(bad_manifest))

    with pytest.raises(ValueError, match=expected_match):
      unpack_visual_frames(buffer.getvalue())

  @pytest.mark.parametrize(
      'manifest_bytes',
      [
          pytest.param(b'{not json', id='invalid_json'),
          pytest.param(b'\xff\xfe\x00', id='invalid_utf8'),
      ],
  )
  def test_manifest_that_is_not_json_is_rejected(
      self, manifest_bytes: bytes
  ) -> None:
    """A `metadata.json` that is not JSON gets an error naming the manifest."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('metadata.json', manifest_bytes)

    with pytest.raises(ValueError, match='metadata.json is not valid JSON'):
      unpack_visual_frames(buffer.getvalue())

  @pytest.mark.parametrize(
      'format_version',
      [
          pytest.param(None, id='missing'),
          pytest.param(2, id='newer'),
          pytest.param('1', id='string'),
          pytest.param(True, id='boolean'),
      ],
  )
  def test_unsupported_format_version_is_rejected(
      self, format_version: object
  ) -> None:
    """An archive in a format version this reader does not know is rejected.

    Failing clearly keeps an older reader from misreading a newer archive.
    """
    manifest: dict[str, object] = {'frames': []}
    if format_version is not None:
      manifest['formatVersion'] = format_version
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('metadata.json', json.dumps(manifest))

    with pytest.raises(ValueError, match='format version'):
      unpack_visual_frames(buffer.getvalue())

  @pytest.mark.parametrize(
      'compressed_member',
      [
          pytest.param('frames/frame_0000.jpeg', id='frame_member'),
          pytest.param('metadata.json', id='manifest_member'),
          pytest.param('notes.txt', id='unlisted_member'),
      ],
  )
  def test_compressed_member_is_rejected(self, compressed_member: str) -> None:
    """A compressed member is refused before anything is read.

    Decompressing it could take far more memory than the archive occupies, or
    fail with an error other than the documented `ValueError`.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(2))
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
      members = {name: source.read(name) for name in source.namelist()}
    members['notes.txt'] = b'a note the manifest does not list'
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(rebuilt, mode='w') as target:
      for name, data in members.items():
        target.writestr(
            name,
            data,
            compress_type=(
                zipfile.ZIP_DEFLATED
                if name == compressed_member
                else zipfile.ZIP_STORED
            ),
        )

    with pytest.raises(ValueError, match='must be stored uncompressed'):
      unpack_visual_frames(rebuilt.getvalue())

  @pytest.mark.parametrize(
      'flag',
      [
          pytest.param(0x01, id='encrypted'),
          pytest.param(0x20, id='compressed_patched_data'),
          pytest.param(0x40, id='strong_encryption'),
      ],
  )
  def test_member_flagged_encrypted_or_patched_is_rejected(
      self, flag: int
  ) -> None:
    """A member `zipfile` cannot read raises ValueError up front.

    `zipfile` would raise `RuntimeError` for an encrypted member and
    `NotImplementedError` for compressed patched data or strong encryption.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(2))
    marked = bytearray(archive_bytes)
    # The member's central directory entry is the last occurrence of its name,
    # and holds the general-purpose flags 38 bytes before the name.
    flags_offset = marked.rindex(b'frames/frame_0001.jpeg') - 38
    marked[flags_offset] |= flag

    with pytest.raises(ValueError, match='uncompressed and unencrypted'):
      unpack_visual_frames(bytes(marked))

  def test_member_needing_newer_zip_version_is_rejected(self) -> None:
    """A member needing a newer ZIP version raises ValueError.

    `zipfile` raises `NotImplementedError` for it while opening the archive,
    before any member can be checked.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(2))
    marked = bytearray(archive_bytes)
    # The member's central directory entry holds the version needed to extract
    # 40 bytes before the name; zipfile reads up to version 6.3.
    version_offset = marked.rindex(b'frames/frame_0001.jpeg') - 40
    marked[version_offset] = 64

    with pytest.raises(ValueError, match='not a valid ZIP'):
      unpack_visual_frames(bytes(marked))

  def test_frame_member_listed_twice_is_rejected(self) -> None:
    """A frame member listed twice is refused rather than read twice.

    Reading it once per listing would let a small archive return one large
    frame many times over.
    """
    archive_bytes, _ = pack_visual_frames(make_frames(1))
    manifest = written_manifest(archive_bytes)
    manifest['frames'].append(manifest['frames'][0])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w') as archive:
      archive.writestr('frames/frame_0000.jpeg', b'frame-0-payload')
      archive.writestr('metadata.json', json.dumps(manifest))

    with pytest.raises(ValueError, match='more than once'):
      unpack_visual_frames(buffer.getvalue())
