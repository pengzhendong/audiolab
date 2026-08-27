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

from importlib import import_module

_EXPORTS = {
    "AudioDecoder": ("audiolab.av.codec", "AudioDecoder"),
    "AudioEncoder": ("audiolab.av.codec", "AudioEncoder"),
    "AudioFormat": ("audiolab.av.format", "AudioFormat"),
    "AudioLayout": ("audiolab.av.layout", "AudioLayout"),
    "ContainerFormat": ("audiolab.av.container", "ContainerFormat"),
    "FilterSpec": ("audiolab.av.typing", "FilterSpec"),
    "Graph": ("audiolab.av.graph", "Graph"),
    "aformat": ("audiolab.av.processing", "aformat"),
    "audio_decoders": ("audiolab.av.codec", "audio_decoders"),
    "audio_encoders": ("audiolab.av.codec", "audio_encoders"),
    "audio_formats": ("audiolab.av.format", "audio_formats"),
    "audio_layouts": ("audiolab.av.layout", "audio_layouts"),
    "build_filter_chain": ("audiolab.av.processing", "build_filter_chain"),
    "clip": ("audiolab.av.frame", "clip"),
    "codec_aliases": ("audiolab.av.codec", "codec_aliases"),
    "container_formats": ("audiolab.av.container", "container_formats"),
    "extension_formats": ("audiolab.av.container", "extension_formats"),
    "filter": ("audiolab.av.filter", None),
    "from_ndarray": ("audiolab.av.frame", "from_ndarray"),
    "get_codecs": ("audiolab.av.format", "get_codecs"),
    "get_dtype": ("audiolab.av.format", "get_dtype"),
    "get_format": ("audiolab.av.format", "get_format"),
    "split_audio_frame": ("audiolab.av.frame", "split_audio_frame"),
    "standard_channel_layouts": ("audiolab.av.layout", "standard_channel_layouts"),
    "to_ndarray": ("audiolab.av.frame", "to_ndarray"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


__all__ = [
    "AudioDecoder",
    "AudioEncoder",
    "AudioFormat",
    "AudioLayout",
    "ContainerFormat",
    "FilterSpec",
    "Graph",
    "aformat",
    "audio_decoders",
    "audio_encoders",
    "audio_formats",
    "audio_layouts",
    "build_filter_chain",
    "clip",
    "codec_aliases",
    "container_formats",
    "extension_formats",
    "filter",
    "from_ndarray",
    "get_codecs",
    "get_dtype",
    "get_format",
    "split_audio_frame",
    "standard_channel_layouts",
    "to_ndarray",
]
