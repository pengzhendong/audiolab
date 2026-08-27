# Copyright (c) 2025 Zhendong Peng (pzd17@tsinghua.org.cn)
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

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from tempfile import TemporaryFile
from threading import Condition, Thread

import av
from numpy.typing import DTypeLike

from audiolab._processor import AudioProcessor, build_graph_filters, validate_transforms
from audiolab.av import build_filter_chain
from audiolab.av.format import get_dtype
from audiolab.av.frame import split_audio_frame
from audiolab.av.graph import Graph
from audiolab.av.typing import DecodedChunk, FilterSpec


def _looks_like_mp3(signature: bytes | bytearray) -> bool:
    if signature[:3] == b"ID3":
        return True
    if len(signature) < 3 or signature[0] != 0xFF or signature[1] & 0xE0 != 0xE0:
        return False
    version = signature[1] >> 3 & 0x03
    layer = signature[1] >> 1 & 0x03
    bit_rate = signature[2] >> 4 & 0x0F
    sample_rate = signature[2] >> 2 & 0x03
    return version != 1 and layer != 0 and bit_rate not in (0, 15) and sample_rate != 3


class _StreamingInput:
    """A non-seekable byte source whose readable window is controlled by ``pull``."""

    def __init__(self):
        self._condition = Condition()
        self._buffer = bytearray()
        self._buffer_offset = 0
        self._bytes_read = 0
        self._bytes_written = 0
        self._read_limit = 0
        self._waiting = False
        self._finished = False
        self._cancelled = False
        self._worker_done = False
        self._signature = bytearray()
        self._signature_checked = False
        self._seekable_archive = None
        self._trim_to_duration = False

    def push(self, data: bytes) -> None:
        with self._condition:
            self._archive_if_seekable_container(data)
            self._buffer.extend(data)
            self._bytes_written += len(data)

    def _archive_if_seekable_container(self, data: bytes) -> None:
        if self._seekable_archive is not None:
            self._seekable_archive.write(data)
            return
        if self._signature_checked:
            return

        needed = 12 - len(self._signature)
        self._signature.extend(data[:needed])
        if len(self._signature) < 12:
            return
        self._signature_checked = True
        is_mp4 = self._signature[4:8] == b"ftyp"
        is_mp3 = _looks_like_mp3(self._signature)
        if is_mp4 or is_mp3:
            self._seekable_archive = TemporaryFile(mode="w+b")  # noqa: SIM115
            self._seekable_archive.write(self._signature)
            self._seekable_archive.write(data[needed:])
            self._trim_to_duration = is_mp3
        self._signature.clear()

    def release(self, final: bool = False) -> int:
        with self._condition:
            self._read_limit = self._bytes_written
            self._finished = final
            self._condition.notify_all()
            return self._read_limit

    def read(self, size: int = -1) -> bytes:
        with self._condition:
            while self._bytes_read >= self._read_limit and not self._finished and not self._cancelled:
                self._waiting = True
                self._condition.notify_all()
                self._condition.wait()
                self._waiting = False

            if self._cancelled or self._bytes_read >= self._read_limit:
                return b""
            available = self._read_limit - self._bytes_read
            length = available if size < 0 else min(size, available)
            start = self._buffer_offset
            end = start + length
            data = bytes(self._buffer[start:end])
            self._buffer_offset = end
            self._bytes_read += length
            self._compact_buffer()
            return data

    def _compact_buffer(self) -> None:
        if self._buffer_offset == len(self._buffer):
            self._buffer = bytearray()
            self._buffer_offset = 0
        elif self._buffer_offset >= 1024 * 1024 and self._buffer_offset >= len(self._buffer) // 2:
            del self._buffer[: self._buffer_offset]
            self._buffer_offset = 0

    def wait_until_processed(self, target: int) -> None:
        with self._condition:
            self._condition.wait_for(lambda: (self._waiting and self._bytes_read >= target) or self._worker_done)

    def worker_done(self) -> None:
        with self._condition:
            self._worker_done = True
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._finished = True
            self._discard_buffer()
            self._condition.notify_all()

    def discard_buffer(self) -> None:
        with self._condition:
            self._discard_buffer()

    def _discard_buffer(self) -> None:
        self._buffer = bytearray()
        self._buffer_offset = 0
        self._bytes_read = self._bytes_written

    def seekable_archive(self):
        with self._condition:
            if self._seekable_archive is None:
                return None
            self._seekable_archive.seek(0)
            return self._seekable_archive

    def expected_audio_samples(self) -> int | None:
        """Read the finalized archive metadata used to remove codec end padding."""
        with self._condition:
            archive = self._seekable_archive
            if archive is None or not self._finished or not self._trim_to_duration:
                return None
            archive.flush()
            archive.seek(0)
            try:
                with av.open(archive, metadata_encoding="latin1") as container:
                    stream = container.streams.audio[0]
                    if stream.duration is None:
                        return None
                    return round(stream.duration * stream.time_base * stream.sample_rate)
            except (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
                return None
            finally:
                archive.seek(0, 2)

    def close_archive(self) -> None:
        with self._condition:
            archive = self._seekable_archive
            self._seekable_archive = None
        if archive is not None:
            archive.close()

    @property
    def buffered_bytes(self) -> int:
        with self._condition:
            return self._bytes_written - self._bytes_read

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    @property
    def finished(self) -> bool:
        with self._condition:
            return self._finished

    @property
    def trims_trailing_padding(self) -> bool:
        with self._condition:
            return self._trim_to_duration


@dataclass(frozen=True)
class _ProcessorConfig:
    filters: list[FilterSpec] | None
    dtype: DTypeLike | None
    sample_rate: int | None
    to_mono: bool
    speed: float
    pitch_shift: float
    frame_size: int


class _DecoderState:
    def __init__(self, config: _ProcessorConfig):
        self.source = _StreamingInput()
        self.outputs: SimpleQueue[DecodedChunk] = SimpleQueue()
        self.config = config
        self.graph: Graph | None = None
        self.error: BaseException | None = None
        self.failed = False
        self.awaiting_archive = False
        self.output_count = 0


class StreamReader:
    """Incrementally decode encoded audio bytes.

    The decoder stays alive across calls, so encoded bytes are consumed once and
    released as soon as FFmpeg has read them. Call ``pull(partial=True)`` once
    when no more bytes will be pushed. Common output transforms are configured
    with dtype, sample rate, channel, speed, and pitch arguments; callers never
    need to select a processing engine or an FFmpeg sample format.
    """

    def __init__(
        self,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        frame_size: int = 1024,
    ):
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        validate_transforms(speed, pitch_shift)

        raw_filters = None if filters is None else list(filters)
        self.filters = (
            build_filter_chain(
                raw_filters,
                dtype=dtype,
                is_planar=False,
                sample_format=None,
                sample_rate=sample_rate,
                to_mono=to_mono,
            )
            if raw_filters
            else None
        )
        self.frame_size = frame_size
        self._config = _ProcessorConfig(
            raw_filters,
            dtype,
            sample_rate,
            to_mono,
            speed,
            pitch_shift,
            frame_size,
        )
        self._state = _DecoderState(self._config)
        self._thread: Thread | None = None
        self._new_bytes = 0
        self._finalized = False
        self._closed = False

    def push(self, data: bytes) -> None:
        if self._state.failed:
            raise RuntimeError("Cannot push data after the decoder has failed; call reset() first")
        if self._finalized or self._closed:
            raise RuntimeError("Cannot push data after the stream has been finalized or closed")
        self._state.source.push(data)
        self._new_bytes += len(data)

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        if self._closed:
            return
        if self._finalized or (self._new_bytes == 0 and not partial):
            yield from self._drain_outputs()
            self._raise_decoder_error()
            return

        self._new_bytes = 0
        target = self._state.source.release(final=partial)
        if self._state.awaiting_archive:
            self._state.source.discard_buffer()
        self._start_decoder()
        if partial:
            self._finalized = True
            while self._thread is not None:
                self._thread.join()
                if not self._state.awaiting_archive or self._state.failed:
                    break
                self._start_decoder()
        else:
            self._state.source.wait_until_processed(target)

        yield from self._drain_outputs()
        self._raise_decoder_error()

    def _start_decoder(self) -> None:
        if self._thread is not None and (
            self._thread.is_alive() or not (self._state.awaiting_archive and self._state.source.finished)
        ):
            return
        self._thread = Thread(target=self._decode, args=(self._state,), name="audiolab-stream-reader", daemon=True)
        self._thread.start()

    @staticmethod
    def _decode(state: _DecoderState) -> None:
        expected_errors = (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError)
        decode_error: BaseException | None = None
        try:
            if state.awaiting_archive:
                state.awaiting_archive = False
                archive = state.source.seekable_archive()
                if archive is None:
                    raise RuntimeError("Seekable stream archive is unavailable")
                try:
                    StreamReader._decode_source(state, archive)
                except expected_errors as error:
                    state.failed = True
                    StreamReader._store_error(state, error)
                return

            try:
                StreamReader._decode_source(state, state.source)
            except expected_errors as error:
                decode_error = error

            archive = state.source.seekable_archive()
            if not state.source.finished and state.output_count == 0 and archive is not None:
                state.awaiting_archive = True
                state.source.discard_buffer()
                return
            if state.source.finished and not state.source.cancelled and state.output_count == 0 and archive is not None:
                try:
                    StreamReader._decode_source(state, archive)
                    decode_error = None
                except expected_errors as error:
                    decode_error = error

            if not state.source.finished and not state.source.cancelled:
                state.failed = True
                StreamReader._store_error(
                    state,
                    decode_error or RuntimeError("Decoder stopped before the stream was finalized"),
                )
            elif state.output_count == 0 and decode_error is not None:
                state.failed = True
                StreamReader._store_error(state, decode_error)
        except BaseException as error:
            state.failed = True
            StreamReader._store_error(state, error)
        finally:
            state.graph = None
            if state.failed:
                state.source.cancel()
            if not state.awaiting_archive:
                state.source.close_archive()
            state.source.worker_done()

    @staticmethod
    def _store_error(state: _DecoderState, error: BaseException) -> None:
        error = error.with_traceback(None)
        error.__cause__ = None
        error.__context__ = None
        state.error = error

    def _raise_decoder_error(self) -> None:
        error = self._state.error
        self._state.error = None
        if error is not None:
            raise error.with_traceback(None)

    @staticmethod
    def _decode_source(state: _DecoderState, source) -> None:
        with av.open(source, metadata_encoding="latin1") as container:
            stream = container.streams.audio[0]
            config = state.config
            filters = None
            output_dtype = config.dtype
            if config.filters:
                filters = build_graph_filters(
                    config.filters,
                    input_sample_rate=stream.sample_rate,
                    speed=config.speed,
                    pitch_shift=config.pitch_shift,
                    dtype=config.dtype,
                    is_planar=False,
                    sample_format=None,
                    output_sample_rate=config.sample_rate,
                    to_mono=config.to_mono,
                )
            processor = AudioProcessor(
                input_sample_rate=stream.sample_rate,
                input_dtype=get_dtype(stream.format),
                input_sample_format=stream.format,
                input_layout=stream.layout.name,
                input_time_base=stream.time_base,
                channels=stream.channels,
                filters=filters,
                dtype=output_dtype,
                output_sample_rate=config.sample_rate,
                to_mono=config.to_mono,
                speed=config.speed,
                pitch_shift=config.pitch_shift,
                frame_size=config.frame_size,
            )
            state.graph = processor.graph
            try:
                trim_trailing_padding = source is state.source and state.source.trims_trailing_padding
                pending_frames = []
                processed_samples = 0
                for packet in container.demux(stream):
                    decoded_frames = packet.decode()
                    if trim_trailing_padding and decoded_frames:
                        for audio_frame in pending_frames:
                            processed_samples += audio_frame.samples
                            processor.push(audio_frame)
                            StreamReader._queue_chunks(state, processor.pull())
                        pending_frames = decoded_frames
                        continue
                    for audio_frame in decoded_frames:
                        processor.push(audio_frame)
                        StreamReader._queue_chunks(state, processor.pull())
                if trim_trailing_padding:
                    expected_samples = state.source.expected_audio_samples()
                    pending_samples = sum(audio_frame.samples for audio_frame in pending_frames)
                    if expected_samples is None or not (
                        processed_samples <= expected_samples <= processed_samples + pending_samples
                    ):
                        remaining_samples = None
                    else:
                        remaining_samples = expected_samples - processed_samples
                    for audio_frame in pending_frames:
                        if remaining_samples is not None:
                            if remaining_samples == 0:
                                break
                            audio_frame, _ = split_audio_frame(audio_frame, remaining_samples)
                            if audio_frame is None:
                                break
                            remaining_samples -= audio_frame.samples
                        processor.push(audio_frame)
                        StreamReader._queue_chunks(state, processor.pull())
                if not state.source.cancelled:
                    StreamReader._queue_chunks(state, processor.pull(partial=True))
            finally:
                processor.close()

    @staticmethod
    def _queue_chunks(state: _DecoderState, chunks: Iterator[DecodedChunk]) -> None:
        for chunk in chunks:
            state.outputs.put(chunk)
            state.output_count += 1

    def _drain_outputs(self) -> Iterator[DecodedChunk]:
        while True:
            try:
                yield self._state.outputs.get_nowait()
            except Empty:
                return

    @property
    def buffered_bytes(self) -> int:
        return self._state.source.buffered_bytes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.source.cancel()
        if self._thread is not None:
            self._thread.join()
        self._state.source.close_archive()
        for _ in self._drain_outputs():
            pass
        self._state.error = None
        self._state.graph = None
        self._thread = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def reset(self) -> None:
        self.close()
        self._state = _DecoderState(self._config)
        self._thread = None
        self._new_bytes = 0
        self._finalized = False
        self._closed = False
