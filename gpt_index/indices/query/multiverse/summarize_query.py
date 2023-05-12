import logging
from typing import Any, Dict, Optional, cast

from langchain.input import print_text

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder
from gpt_index.indices.utils import extract_numbers_given_response, get_sorted_node_list
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
)
from gpt_index.prompts.prompts import TreeSelectMultiplePrompt, TreeSelectPrompt
from gpt_index.response.schema import Response

logger = logging.getLogger(__name__)


class GPTMultiverseIndexSummarizeQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Multiverse Index summarize query.

    This class traverses the index graph, summarizing each subgraph to 
    answer the query.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    """

