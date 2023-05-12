"""Summarize query."""

import logging
from typing import List, Optional

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseMode


class GPTMultiversePathIndexQuery(BaseGPTIndexQuery[IndexGraph]):
    """GPT Multiverse Index query.

    Get the path from the root to the checked out node.

    """

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logging.info(f"> Starting query: {query_bundle.query_str}")
        node_path = self._get_checked_out_path(self._index_struct.root_nodes)
        return node_path

    def _get_checked_out_path(self, nodes: List[Node], path: List[Node] = None) -> List[Node]:
        """Get path from root to leaf via checked out nodes."""
        path = path or []
        for node in nodes.values():
            if node.node_info.get('checked_out', False):
                path.append(node)
                return self._get_checked_out_path(self._index_struct.get_children(node), path)
        return path

