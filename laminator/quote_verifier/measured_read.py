#!/usr/bin/env python

# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import io

class MeasuredBytesIO(io.BytesIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(initial)
        super().__init__(initial, *args, **kwargs)

class MeasuredStringIO(io.StringIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(bytes(initial, 'utf8'))
        super().__init__(initial, *args, **kwargs)

def open_measured(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha512()

    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        raise IOError("Measured open can't be writable")

    if 'b' in mode:
        with open(path, mode) as fh:
            data = fh.read()
            return MeasuredBytesIO(data, hasher)
    else:
        with open(path, mode) as fh:
            data = fh.read()
            return MeasuredStringIO(data, hasher)