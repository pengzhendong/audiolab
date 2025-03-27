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

from typing import Dict, Literal, Optional, Tuple, Union

from .utils import format_options, parse_frequency, parse_threshold


def asetrate(sample_rate: Union[str, int] = 44100) -> Tuple[str, str]:
    """
    Change the sample rate without altering the data.

    Args:
        sample_rate: set the sample rate (from 1 to INT_MAX) (default 44100)

    $ ffmpeg -h filter=asetrate
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#asetrate
    """
    sample_rate = parse_frequency(sample_rate)
    assert sample_rate >= 1
    return "asetrate", str(sample_rate)


def aresample(sample_rate: Union[str, int] = 0, resampler_options: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """
    Resample audio data.

    Args:
        sample_rate: (from 0 to INT_MAX) (default 0)

    $ ffmpeg -h filter=aresample
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#aresample
    - https://ffmpeg.org/ffmpeg-resampler.html#Resampler-Options
    """
    sample_rate = parse_frequency(sample_rate)
    assert sample_rate >= 0
    options = [str(sample_rate)]
    options.extend(list(resampler_options.items()) if resampler_options else [])
    return "aresample", format_options(options)


def atempo(tempo: float = 1) -> Tuple[str, str]:
    """
    Adjust audio tempo.

    Args:
        tempo: set tempo scale factor (from 0.5 to 100) (default 1)

    $ ffmpeg -h filter=atempo
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#atempo
    """
    assert 0.5 <= tempo <= 100.0
    return "atempo", str(tempo)


def volume(
    volume: Union[float, str] = "1.0",
    precision: Literal[0, "fixed", 1, "float", 2, "double"] = "float",
    eval: Literal[0, "once", 1, "frame"] = "once",
    replaygain: Literal[0, "drop", 1, "ignore", 2, "track", 3, "album"] = "drop",
    replaygain_preamp: float = 0,
    replaygain_noclip: bool = True,
) -> Tuple[str, str]:
    """
    Change input volume.

    Args:
        volume: set volume adjustment expression (default "1.0")
        precision: select mathematical precision (from 0 to 2) (default float)
            fixed: select 8-bit fixed-point
            float: select 32-bit floating-point
            double: select 64-bit floating-point
        eval: specify when to evaluate expressions (from 0 to 1) (default once)
            once: eval volume expression once
            frame: eval volume expression per-frame
        replaygain: Apply replaygain side data when present (from 0 to 3) (default drop)
            drop: replaygain side data is dropped
            ignore: replaygain side data is ignored
            track: track gain is preferred
            album: album gain is preferred
        replaygain_preamp: Apply replaygain pre-amplification (from -15 to 15) (default 0)
        replaygain_noclip: Apply replaygain clipping prevention (default true)

    $ ffmpeg -h filter=volume
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#volume
    """
    assert -15 <= replaygain_preamp <= 15
    return "volume", format_options(
        {
            "volume": volume,
            "precision": precision,
            "eval": eval,
            "replaygain": replaygain,
            "replaygain_preamp": replaygain_preamp,
            "replaygain_noclip": int(replaygain_noclip),
        }
    )


def highpass(
    frequency: Union[str, float] = 3000,
    width_type: Literal[1, "h", 2, "q", 3, "o", 4, "s", 5, "k"] = "q",
    width: float = 0.707,
    poles: int = 2,
    mix: float = 1.0,
    channels: str = "all",
    normalize: bool = False,
    transform: Literal[0, "di", 1, "dii", 2, "tdi", 3, "tdii", 4, "latt", 5, "svf", 6, "zdf"] = "di",
    precision: Literal[-1, "auto", 0, "s16", 1, "s32", 2, "f32", 3, "f64"] = "auto",
    blocksize: int = 0,
):
    """
    Apply a high-pass filter with 3dB point frequency.

    Args:
        frequency: set frequency (from 0 to 999999) (default 3000)
        width_type: set filter-width type (from 1 to 5) (default q)
            h: Hz
            q: Q-Factor
            o: octave
            s: slope
            k: kHz
        width: set width (from 0 to 99999) (default 0.707)
        poles: set number of poles (from 1 to 2) (default 2)
        mix: set mix (from 0 to 1) (default 1)
        channels: set channels to filter (default "all")
        normalize: normalize coefficients (default false)
        transform: set transform type (from 0 to 6) (default di)
            di: direct form I
            dii: direct form II
            tdi: transposed direct form I
            tdii: transposed direct form II
            latt: lattice-ladder form
            svf: state variable filter form
            zdf: zero-delay filter form
        precision: set filtering precision (from -1 to 3) (default auto)
            auto: automatic
            s16: signed 16-bit
            s32: signed 32-bit
            f32: floating-point single
            f64: floating-point double
        blocksize: set the block size (from 0 to 32768) (default 0)

    $ ffmpeg -h filter=highpass
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#highpass
    """
    frequency = parse_frequency(frequency)
    assert 0 <= frequency <= 999999
    assert 0 <= width <= 99999
    assert 1 <= poles <= 2
    assert 0 <= mix <= 1
    assert 0 <= blocksize <= 32768
    return "highpass", format_options(
        {
            "frequency": frequency,
            "width_type": width_type,
            "width": width,
            "poles": poles,
            "mix": mix,
            "channels": channels,
            "normalize": normalize,
            "transform": transform,
            "precision": precision,
            "blocksize": blocksize,
        }
    )


def lowpass(
    frequency: Union[str, float] = 500,
    width_type: Literal[1, "h", 2, "q", 3, "o", 4, "s", 5, "k"] = "q",
    width: float = 0.707,
    poles: int = 2,
    mix: float = 1.0,
    channels: str = "all",
    normalize: bool = False,
    transform: Literal[0, "di", 1, "dii", 2, "tdi", 3, "tdii", 4, "latt", 5, "svf", 6, "zdf"] = "di",
    precision: Literal[-1, "auto", 0, "s16", 1, "s32", 2, "f32", 3, "f64"] = "auto",
    blocksize: int = 0,
) -> Tuple[str, str]:
    """
    Apply a low-pass filter with 3dB point frequency.

    Args:
        frequency: set frequency (from 0 to 999999) (default 500)
        width_type: set filter-width type (from 1 to 5) (default q)
            h: Hz
            q: Q-Factor
            o: octave
            s: slope
            k: kHz
        width: set width (from 0 to 99999) (default 0.707)
        poles: set number of poles (from 1 to 2) (default 2)
        mix: set mix (from 0 to 1) (default 1)
        channels: set channels to filter (default "all")
        normalize: normalize coefficients (default false)
        transform: set transform type (from 0 to 6) (default di)
            di: direct form I
            dii: direct form II
            tdi: transposed direct form I
            tdii: transposed direct form II
            latt: lattice-ladder form
            svf: state variable filter form
            zdf: zero-delay filter form
        precision: set filtering precision (from -1 to 3) (default auto)
            auto: automatic
            s16: signed 16-bit
            s32: signed 32-bit
            f32: floating-point single
            f64: floating-point double
        blocksize: set the block size (from 0 to 32768) (default 0)

    $ ffmpeg -h filter=lowpass
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#lowpass
    """
    frequency = parse_frequency(frequency)
    assert 0 <= frequency <= 999999
    assert 0 <= width <= 99999
    assert 1 <= poles <= 2
    assert 0 <= mix <= 1
    assert 0 <= blocksize <= 32768
    return "lowpass", format_options(
        {
            "frequency": frequency,
            "width_type": width_type,
            "width": width,
            "poles": poles,
            "mix": mix,
            "channels": channels,
            "normalize": normalize,
            "transform": transform,
            "precision": precision,
            "blocksize": blocksize,
        }
    )


def afftdn(
    noise_reduction: float = 12,
    noise_floor: float = -50,
    noise_type: Literal["white", "vinyl", "shellac", "custom"] = "white",
    band_noise: Optional[str] = None,
    residual_floor: float = -38,
    track_noise: bool = False,
    track_residual: bool = False,
    output_mode: Literal[0, "input", 1, "output", 2, "noise"] = "output",
    adaptivity: float = 0.5,
    floor_offset: float = 1,
    noise_link: Literal[0, "none", 1, "min", 2, "max", 3, "average"] = "min",
    band_multiplier: float = 1.25,
    sample_noise: Literal[0, "none", 1, "start", "begin", 2, "stop", "end"] = "none",
    gain_smooth: int = 0,
) -> Tuple[str, str]:
    """
    Denoise audio samples using FFT.

    Args:
        noise_reduction: set the noise reduction (from 0.01 to 97) (default 12)
        noise_floor: set the noise floor (from -80 to -20) (default -50)
        noise_type: set the noise type (from 0 to 3) (default white)
            white: white noise
            vinyl: vinyl noise
            shellac: shellac noise
            custom: custom noise
        band_noise: set the custom bands noise
        residual_floor: set the residual floor (from -80 to -20) (default -38)
        track_noise: track noise (default false)
        track_residual: track residual (default false)
        output_mode: set output mode (from 0 to 2) (default output)
            input: input
            output: output
            noise: noise
        adaptivity: set adaptivity factor (from 0 to 1) (default 0.5)
        floor_offset: set noise floor offset factor (from -2 to 2) (default 1)
        noise_link: set the noise floor link (from 0 to 3) (default min)
            none: none
            min: min
            max: max
            average: average
        band_multiplier: set band multiplier (from 0.2 to 5) (default 1.25)
        sample_noise: set sample noise mode (from 0 to 2) (default none)
            none: none
            start: start
            begin: start
            stop: stop
            end: stop
        gain_smooth: set gain smooth radius (from 0 to 50) (default 0)

    $ ffmpeg -h filter=afftdn
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#afftdn
    """
    assert 0.01 <= noise_reduction <= 97
    assert -80 <= noise_floor <= -20
    assert -80 <= residual_floor <= -20
    assert 0 <= adaptivity <= 1
    assert -2 <= floor_offset <= 2
    assert 0.2 <= band_multiplier <= 5
    assert 0 <= gain_smooth <= 50
    options = {
        "noise_reduction": noise_reduction,
        "noise_floor": noise_floor,
        "noise_type": noise_type,
        "residual_floor": residual_floor,
        "track_noise": int(track_noise),
        "track_residual": int(track_residual),
        "output_mode": output_mode,
        "adaptivity": adaptivity,
        "floor_offset": floor_offset,
        "noise_link": noise_link,
        "band_multiplier": band_multiplier,
        "sample_noise": sample_noise,
        "gain_smooth": gain_smooth,
    }
    if noise_type == "custom":
        assert band_noise is not None
        options["band_noise"] = band_noise
    return "afftdn", format_options(options)


def pan(layout: str, outdef: str) -> Tuple[str, str]:
    """
    Remix channels with coefficients (panning).

    Args:
        layout: output channel layout or number of channels
        outdef: output channel specification, of the form: "out_name=[gain*]in_name[(+-)[gain*]in_name...]"
            out_name: output channel to define, either a channel name (FL, FR, etc.) or a channel number (c0, c1, etc.)
            gain: multiplicative coefficient for the channel, 1 leaving the volume unchanged
            in_name: input channel to use, see out_name for details; it is not possible to mix named and numbered input channels
    If the '=' in a channel specification is replaced by '<', then the gains for that specification will be renormalized so that the total is 1, thus avoiding clipping noise.

    $ ffmpeg -h filter=pan
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#pan
    - https://trac.ffmpeg.org/wiki/AudioChannelManipulation
    """
    return "pan", f"{layout}|{outdef}"


def loudnorm(
    i: float = -24,
    lra: float = 7,
    tp: float = -2,
    measured_i: float = 0,
    measured_lra: float = 0,
    measured_tp: float = 99,
    measured_thresh: float = -70,
    offset: float = 0,
    linear: bool = True,
    dual_mono: bool = False,
) -> Tuple[str, str]:
    """
    EBU R128 loudness normalization.

    Args:
        i: set integrated loudness target (from -70 to -5) (default -24)
        lra: set loudness range target (from 1 to 50) (default 7)
        tp: set maximum true peak (from -9 to 0) (default -2)
        measured_i: measured IL of input file (from -99 to 0) (default 0)
        measured_lra: measured LRA of input file (from 0 to 99) (default 0)
        measured_tp: measured true peak of input file (from -99 to 99) (default 99)
        measured_thresh: measured threshold of input file (from -99 to 0) (default -70)
        offset: set offset gain (from -99 to 99) (default 0)
        linear: normalize linearly if possible (default true)
        dual_mono: treat mono input as dual-mono (default false)

    $ ffmpeg -h filter=loudnorm
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#loudnorm
    """
    assert -70 <= i <= -5
    assert 1 <= lra <= 50
    assert -9 <= tp <= 0
    assert -99 <= measured_i <= 0
    assert 0 <= measured_lra <= 99
    assert -99 <= measured_tp <= 99
    assert -99 <= measured_thresh <= 0
    assert -99 <= offset <= 99
    return "loudnorm", format_options(
        {
            "i": i,
            "lra": lra,
            "tp": tp,
            "measured_i": measured_i,
            "measured_lra": measured_lra,
            "measured_tp": measured_tp,
            "measured_thresh": measured_thresh,
            "offset": offset,
            "linear": int(linear),
            "dual_mono": int(dual_mono),
        }
    )


def silenceremove(
    start_periods: int = 0,
    start_duration: Union[float, str] = 0,
    start_threshold: Union[float, str] = 0,
    start_silence: Union[float, str] = 0,
    start_mode: Literal[0, "any", 1, "all"] = "any",
    stop_periods: int = 0,
    stop_duration: Union[float, str] = 0,
    stop_threshold: Union[float, str] = 0,
    stop_silence: Union[float, str] = 0,
    stop_mode: Literal[0, "any", 1, "all"] = "any",
    detection: Literal[0, "avg", 1, "rms", 2, "peak", 3, "median", 4, "ptp", 5, "dev"] = "rms",
    window: float = 0.02,
    timestamp: Literal[0, "write", 1, "copy"] = "write",
) -> Tuple[str, str]:
    """
    Remove silence.

    Args:
        start_periods: set periods of silence parts to skip from start (from 0 to 9000) (default 0)
        start_duration: set start duration of non-silence part (default 0)
        start_threshold: set threshold for start silence detection (from 0 to DBL_MAX) (default 0)
        start_silence: set start duration of silence part to keep (default 0)
        start_mode: set which channel will trigger trimming from start (from 0 to 1) (default any)
            any: any
            all: all
        stop_periods: set periods of silence parts to skip from end (from -9000 to 9000) (default 0)
        stop_duration: set stop duration of silence part (default 0)
        stop_threshold: set threshold for stop silence detection (from 0 to DBL_MAX) (default 0)
        stop_silence: set stop duration of silence part to keep (default 0)
        stop_mode: set which channel will trigger trimming from end (from 0 to 1) (default all)
            any: any
            all: all
        detection: set how silence is detected (from 0 to 5) (default rms)
            avg: use mean absolute values of samples
            rms: use root mean squared values of samples
            peak: use max absolute values of samples
            median: use median of absolute values of samples
            ptp: use absolute of max peak to min peak difference
            dev: use standard deviation from values of samples
        window: set duration of window for silence detection (default 0.02)
        timestamp: set how every output frame timestamp is processed (from 0 to 1) (default write)
            write: full timestamps rewrite, keep only the start time
            copy: non-dropped frames are left with same timestamp

    $ ffmpeg -h filter=silenceremove
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#silenceremove
    - https://blog.tubumu.com/2021/12/07/ffmpeg-command-silenceremove
    """
    start_threshold = parse_threshold(start_threshold)
    stop_threshold = parse_threshold(stop_threshold)
    assert 0 <= start_periods <= 9000
    assert 0 <= start_threshold
    assert -9000 <= stop_periods <= 9000
    assert 0 <= stop_threshold
    assert 0 <= window <= 1
    return "silenceremove", format_options(
        {
            "start_periods": start_periods,
            "start_duration": start_duration,
            "start_threshold": start_threshold,
            "start_silence": start_silence,
            "start_mode": start_mode,
            "stop_periods": stop_periods,
            "stop_duration": stop_duration,
            "stop_threshold": stop_threshold,
            "stop_silence": stop_silence,
            "stop_mode": stop_mode,
            "detection": detection,
            "window": window,
            "timestamp": timestamp,
        }
    )


def aformat(
    sample_fmts: Optional[str] = None,
    sample_rates: Optional[str] = None,
    channel_layouts: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Convert the input audio to one of the specified formats.

    Args:
        sample_fmts: '|'-separated list of sample formats.
        sample_rates: '|'-separated list of sample rates.
        channel_layouts: '|'-separated list of channel layouts.

    $ ffmpeg -h filter=aformat
    Also see:
    - https://ffmpeg.org/ffmpeg-filters.html#aformat

    $ ffmpeg -sample_fmts
    $ ffmpeg -encoders | grep "^ A"
    $ ffmpeg -h encoder=aac | grep "Supported sample rates"
    $ ffmpeg -layouts
    """
    options = {}
    if sample_fmts is not None:
        options["sample_fmts"] = sample_fmts
    if sample_rates is not None:
        sample_rates = [parse_frequency(rate) for rate in str(sample_rates).split("|")]
        options["sample_rates"] = "|".join([str(rate) for rate in sample_rates])
    if channel_layouts is not None:
        options["channel_layouts"] = channel_layouts
    return "aformat", format_options(options)
