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

from typing import Any

import click

import audiolab
from audiolab.reader.info import Info


@click.command()
@click.argument("audio-files", type=click.File(mode="rb"), nargs=-1)
@click.option("-S", "--stream-id", type=int, default=0)
@click.option("-f", "--force-decoding", is_flag=True, help="Force decoding the audio file to get the duration")
@click.option("-t", "--show-file-type", is_flag=True, help="Show detected file-type")
@click.option("-r", "--show-sample-rate", is_flag=True, help="Show sample-rate")
@click.option("-c", "--show-channels", is_flag=True, help="Show number of channels")
@click.option("-s", "--show-samples", is_flag=True, help="Show number of samples (N/A if unavailable)")
@click.option(
    "-d", "--show-duration-hms", is_flag=True, help="Show duration in hours, minutes and seconds (N/A if unavailable)"
)
@click.option("-D", "--show-duration-seconds", is_flag=True, help="Show duration in seconds (N/A if unavailable)")
@click.option(
    "-b", "--show-bits-per-sample", is_flag=True, help="Show number of bits per sample (N/A if not applicable)"
)
@click.option(
    "-B", "--show-bitrate", is_flag=True, help="Show the bitrate averaged over the whole file (N/A if unavailable)"
)
@click.option("-p", "--show-precision", is_flag=True, help="Show estimated sample precision in bits")
@click.option("-e", "--show-encoding", is_flag=True, help="Show the name of the audio encoding")
@click.option("-a", "--show-comments", is_flag=True, help="Show file comments (annotations) if available")
def main(
    audio_files: Any,
    stream_id: int = 0,
    force_decoding: bool = False,
    show_file_type: bool = False,
    show_sample_rate: bool = False,
    show_channels: bool = False,
    show_samples: bool = False,
    show_duration_hms: bool = False,
    show_duration_seconds: bool = False,
    show_bits_per_sample: bool = False,
    show_bitrate: bool = False,
    show_precision: bool = False,
    show_encoding: bool = False,
    show_comments: bool = False,
):
    """
    Print the information of audio files.

    Args:
        audio_files: The audio files, paths to audio files, or stdin.
        stream_id: The index of the stream to load.
        force_decoding: Whether to force decoding the audio file to get the duration.
        show_file_type: Show detected file-type.
        show_sample_rate: Show sample-rate.
        show_channels: Show number of channels.
        show_samples: Show number of samples (0 if unavailable).
        show_duration_hms: Show duration in hours, minutes and seconds (0 if unavailable).
        show_duration_seconds: Show duration in seconds (0 if unavailable).
        show_bits_per_sample: Show number of bits per sample (0 if not applicable).
        show_bitrate: Show the bitrate averaged over the whole file (0 if unavailable).
        show_precision: Show estimated sample precision in bits.
        show_encoding: Show the name of the audio encoding.
        show_comments: Show file comments (annotations) if available.
    """
    # If no files are provided, use stdin
    if not audio_files:
        # Create a file object for stdin
        stdin_file = click.File(mode="rb").convert("-", None, None)
        audio_files = [stdin_file]

    # Initialize total duration and show_any flag
    total_duration = 0.0
    show_any = any(
        [
            show_file_type,
            show_sample_rate,
            show_channels,
            show_samples,
            show_duration_hms,
            show_duration_seconds,
            show_bits_per_sample,
            show_bitrate,
            show_precision,
            show_encoding,
            show_comments,
        ]
    )

    # Process each audio file
    for audio_file in audio_files:
        info = audiolab.info(audio_file, stream_id, force_decoding)
        # If no specific options are selected, show all information (default behavior)
        if not show_any:
            print(info)
            # Accumulate total duration
            total_duration += info.duration or 0.0

        # Display information based on selected options
        if show_file_type:
            print(info.container.format.name)
        if show_sample_rate:
            print(info.sample_rate)
        if show_channels:
            print(info.channels)
        if show_samples:
            print(info.num_samples or 0)
        if show_duration_hms:
            print(Info.format_duration(info.duration))
        if show_duration_seconds:
            print(info.duration or 0)
        if show_bits_per_sample:
            print(info.precision)
        if show_bitrate:
            print(Info.format_bit_rate(info.bit_rate))
        if show_precision:
            print(info.precision)
        if show_encoding:
            print(info.codec.long_name)
        if show_comments:
            if info.metadata:
                for key, value in info.metadata.items():
                    print(f"{key}: {value}")

    # Print total duration if any files were processed and any options were selected
    if len(audio_files) > 1 and not show_any:
        print(f"\nTotal duration of {len(audio_files)} files: {Info.format_duration(total_duration)}")
