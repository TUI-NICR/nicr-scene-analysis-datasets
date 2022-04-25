# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


class ExtrinsicCameraParametersNormalized(dict):
    """Enables a custom pytorch collate function ignore this dict."""
    pass


class IntrinsicCameraParametersNormalized(dict):
    """Enables a custom pytorch collate function ignore this dict."""
    pass


class OrientationDict(dict):
    """Enables a custom pytorch collate function ignore the orientations."""
    pass


class SampleIdentifier(tuple):
    """Enables a custom pytorch collate function ignore the identifier."""
    pass


@dataclass(frozen=True)
class _LabelBase:
    class_name: str


@dataclass(frozen=True)
class SemanticLabel(_LabelBase):
    is_thing: Union[bool, None]
    use_orientations: Union[bool, None]
    color: Tuple[int]


@dataclass(frozen=True)
class SceneLabel(_LabelBase):
    # maybe add color for scene labels
    pass


class _LabelListBase:
    def __init__(
        self,
        label_list: Tuple[_LabelBase] = ()
    ) -> None:
        self.label_list = list(label_list)
        # a copy of a the class names list for faster name to idx lookup
        self._class_names = ()
        self._update_internal_lists()
        # for iterator
        self._idx = 0

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.label_list[idx]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = self[self._idx]
            self._idx += 1
            return el
        except IndexError:
            self._idx = 0
            raise StopIteration     # done iterating

    def add_label(self, label: _LabelBase):
        self.label_list.append(label)
        self._update_internal_lists()

    def _update_internal_lists(self):
        self._class_names = tuple(item.class_name for item in self.label_list)

    def _name_to_idx(self, name: str) -> int:
        return self._class_names.index(name)

    def index(self, value: Union[_LabelBase, str]) -> int:
        if isinstance(value, _LabelBase):
            return self.label_list.index(value)
        else:
            return self._name_to_idx(value)

    @property
    def class_names(self) -> Tuple[str]:
        return self._class_names


class SemanticLabelList(_LabelListBase):
    @property
    def colors(self) -> Tuple[Tuple[int]]:
        return tuple(item.color for item in self.label_list)

    @property
    def colors_array(self) -> np.ndarray:
        return np.array(self.colors, dtype=np.uint8)

    @property
    def classes_is_thing(self) -> Tuple[bool]:
        return tuple(item.is_thing for item in self.label_list)

    @property
    def classes_use_orientations(self) -> Tuple[bool]:
        return [item.use_orientations for item in self.label_list]


class SceneLabelList(_LabelListBase):
    pass
