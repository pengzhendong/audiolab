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
from queue import Empty, SimpleQueue
from tempfile import TemporaryFile
from threading import Condition, Thread

import av
from numpy.typing import DTypeLike

from audiolab.av import build_filter_chain
from audiolab.av.graph import Graph
from audiolab.av.typing import AudioFormatLike, DecodedChunk, FilterSpec


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
        if self._signature[4:8] == b"ftyp":
            self._seekable_archive = TemporaryFile(mode="w+b")  # noqa: SIM115
            self._seekable_archive.write(self._signature)
            self._seekable_archive.write(data[needed:])
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
            self._buffer = bytearray()
            self._buffer_offset = 0
            self._bytes_read = self._bytes_written
            self._condition.notify_all()

    def seekable_archive(self):
        with self._condition:
            if self._seekable_archive is None:
                return None
            self._seekable_archive.seek(0)
            return self._seekable_archive

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


class _DecoderState:
    def __init__(self, filters: list[FilterSpec] | None, frame_size: int):
        self.source = _StreamingInput()
        self.outputs: SimpleQueue[DecodedChunk] = SimpleQueue()
        self.filters = filters
        self.frame_size = frame_size
        self.graph: Graph | None = None
        self.error: BaseException | None = None
        self.output_count = 0


class StreamReader:
    """Incrementally decode encoded audio bytes.

    The decoder stays alive across calls, so encoded bytes are consumed once and
    released as soon as FFmpeg has read them. Call ``pull(partial=True)`` once
    when no more bytes will be pushed.
    """

    def __init__(
        self,
        filters: list[FilterSpec] | None = None,
        dtype: DTypeLike | None = None,
        is_planar: bool = False,
        sample_format: AudioFormatLike | None = None,
        sample_rate: int | None = None,
        to_mono: bool = False,
        frame_size: int = 1024,
    ):
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.filters = build_filter_chain(
            filters,
            dtype=dtype,
            is_planar=is_planar,
            sample_format=sample_format,
            sample_rate=sample_rate,
            to_mono=to_mono,
        )
        self.frame_size = frame_size
        self._state = _DecoderState(self.filters, frame_size)
        self._thread: Thread | None = None
        self._new_bytes = 0
        self._finalized = False
        self._closed = False

    def push(self, data: bytes) -> None:
        if self._finalized or self._closed:
            raise RuntimeError("Cannot push data after the stream has been finalized or closed")
        self._state.source.push(data)
        self._new_bytes += len(data)

    def pull(self, partial: bool = False) -> Iterator[DecodedChunk]:
        if self._closed:
            return
        if self._finalized or (self._new_bytes == 0 and not partial):
            yield from self._drain_outputs()
            if self._state.error is not None:
                raise self._state.error
            return

        self._new_bytes = 0
        target = self._state.source.release(final=partial)
        self._start_decoder()
        if partial:
            self._finalized = True
            self._thread.join()
        else:
            self._state.source.wait_until_processed(target)

        yield from self._drain_outputs()
        if self._state.error is not None:
            raise self._state.error

    def _start_decoder(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._decode, args=(self._state,), name="audiolab-stream-reader", daemon=True)
        self._thread.start()

    @staticmethod
    def _decode(state: _DecoderState) -> None:
        try:
            with contextlib.suppress(av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
                StreamReader._decode_source(state, state.source)
            archive = state.source.seekable_archive()
            if not state.source.cancelled and state.output_count == 0 and archive is not None:
                StreamReader._decode_source(state, archive)
        except (av.EOFError, av.InvalidDataError, av.OSError, av.PermissionError):
            pass
        except BaseException as error:
            state.error = error
        finally:
            state.source.close_archive()
            state.source.worker_done()

    @staticmethod
    def _decode_source(state: _DecoderState, source) -> None:
        with av.open(source, metadata_encoding="latin1") as container:
            stream = container.streams.audio[0]
            graph = Graph(stream, filters=state.filters, frame_size=state.frame_size)
            state.graph = graph
            for packet in container.demux(stream):
                for audio_frame in packet.decode():
                    graph.push(audio_frame)
                    for chunk in graph.pull():
                        state.outputs.put(chunk)
                        state.output_count += 1
            if not state.source.cancelled:
                for chunk in graph.pull(partial=True):
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

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def reset(self) -> None:
        self.close()
        self._state = _DecoderState(self.filters, self.frame_size)
        self._thread = None
        self._new_bytes = 0
        self._finalized = False
        self._closed = False
