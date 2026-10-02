# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_curator.stages.text.filters.heuristic import (
    BulletsFilter,
    EllipsisFilter,
    LongWordFilter,
    MeanWordLengthFilter,
    PunctuationFilter,
    SymbolsToWordsFilter,
    WordsWithoutAlphabetsFilter,
)


@pytest.mark.parametrize("text", ["", " ", "\n\n", "\xa0"])
def test_empty_and_whitespace_documents_do_not_raise(text: str) -> None:
    """Empty documents must return a defined score instead of raising.

    Regression for NVIDIA-NeMo/Curator#2458.
    """
    filters = [
        SymbolsToWordsFilter(),
        BulletsFilter(),
        LongWordFilter(),
        MeanWordLengthFilter(),
        PunctuationFilter(),
        EllipsisFilter(),
        WordsWithoutAlphabetsFilter(),
    ]
    for document_filter in filters:
        score = document_filter.score_document(text)
        assert isinstance(score, float)
        assert score == score  # not NaN
