from pathlib import Path
from typing import Type

import pandas as pd

from .base import BaseLogParser
from .nor140_overview_xlsx import Nor140OverviewXlsxParser


class ParserRegistry:
    """
    Registry for log parsers.

    Parsers are checked in registration order.
    The first parser whose ``can_parse`` returns True is used.
    """

    def __init__(self) -> None:
        self._parsers: list[Type[BaseLogParser]] = []

    def register(self, parser_cls: Type[BaseLogParser]) -> None:
        if not issubclass(parser_cls, BaseLogParser):
            raise TypeError("Only BaseLogParser subclasses can be registered.")
        if parser_cls not in self._parsers:
            self._parsers.append(parser_cls)

    def unregister(self, parser_cls: Type[BaseLogParser]) -> None:
        if parser_cls in self._parsers:
            self._parsers.remove(parser_cls)

    def clear(self) -> None:
        self._parsers.clear()

    def get_parser_class(self, path: str | Path) -> Type[BaseLogParser]:
        for parser_cls in self._parsers:
            if parser_cls.can_parse(path):
                return parser_cls
        raise ValueError(f"No registered parser could handle file: {path}")

    def get_parser(self, path: str | Path) -> BaseLogParser:
        parser_cls = self.get_parser_class(path)
        return parser_cls()

    def parse(self, path: str | Path) -> pd.DataFrame:
        parser = self.get_parser(path)
        return parser.parse(path)

    @property
    def parsers(self) -> tuple[Type[BaseLogParser], ...]:
        return tuple(self._parsers)


default_registry = ParserRegistry()
default_registry.register(Nor140OverviewXlsxParser)