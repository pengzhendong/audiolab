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

from typing import Any, Callable, Dict, List

from av import filter
from av.option import OptionType

from audiolab.av.utils import get_template


class FilterManager:
    """Manages audio filter functions with lazy initialization."""

    def __init__(self):
        self._filter_data: Dict[str, Dict[str, Any]] = {}
        self._initialized: bool = False

    def _generate_filter_data(self) -> None:
        """Generate filter data by querying available filters."""
        for name in filter.filters_available:
            options = []
            _filter = filter.Filter(name)
            if _filter.options is not None:
                for opt in _filter.options:
                    try:
                        opt_type = opt.type
                    except ValueError:
                        opt_type = OptionType.STRING
                    options.append(
                        {
                            "name": opt.name,
                            "type": opt_type,
                            "default": opt.default,
                            "help": opt.help if opt.name != "temp" else "set temperature °C",
                        }
                    )
            self._filter_data[name] = {
                "name": _filter.name,
                "description": _filter.description,
                "options": options,
            }

    def _create_filter_function(self, name: str):
        """Create a filter function for the given name."""

        def filter_func(args=None, **kwargs):
            return (
                name,
                None if args is None else str(args),
                {k: str(v) for k, v in kwargs.items()},
            )

        filter_func.__name__ = name
        return filter_func

    def _initialize_filters(self) -> None:
        """Initialize filter functions in globals namespace."""
        if self._initialized:
            return

        # Generate filter data on first initialization
        self._generate_filter_data()

        for name in filter.filters_available:
            # Create filter function
            filter_func = self._create_filter_function(name)

            # Set docstring
            data = self._filter_data[name]
            filter_func.__doc__ = get_template("filter").render(
                name=data["name"], description=data["description"], options=data["options"]
            )

            # Add to globals
            globals()[name] = filter_func

        self._initialized = True

    def __getattr__(self, name: str) -> Callable:
        """Dynamically create and return filter function when accessed."""
        if name in filter.filters_available:
            # Initialize all filters on first access
            self._initialize_filters()
            return globals()[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def filters(self) -> List[str]:
        """Get list of available filter names."""
        # Load filter names without initializing functions
        return filter.filters_available


# Create singleton instance
_filter_manager = FilterManager()

# Expose filters property
filters = _filter_manager.filters


# Expose individual filter functions through dynamic attribute access
def __getattr__(name: str) -> Callable:
    """Module-level dynamic attribute access for filter functions."""
    return getattr(_filter_manager, name)
