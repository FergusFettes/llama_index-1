# from dataclasses import dataclass, field
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type, List, Union

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.multiverse.path_query import GPTMultiversePathIndexQuery
from gpt_index.indices.query.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.readers.schema.base import Document
from gpt_index.schema import BaseDocument
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate

import numpy as np

from dotenv import load_dotenv

load_dotenv()


LATEST_SUMMARY: str = "No key information yet. Please extract the key information from the new messages."

# @dataclass
# class SummarizeModel:
#     llm = OpenAI(temperature=0, max_tokens=1000, model="gpt-3.5-turbo", n=1)
#     prompt_str = (
#         "A conversation is ongoing. The key information is continuously being summarized and updated for the last few messages. If there is no change to the key information, it will be returned as is. The key information can be reformatted when new information arrives.\n\n"
#         "Key Information:\n"
#         "{existing_summary}\n\n"
#         "New Messages:\n"
#         "{new_messages}\n\n"
#         "Updated and Reformatted Key Information:\n"
#     )
#     latest_summary: str = LATEST_SUMMARY
#     new_messages: str = ""
#
#     def __post_init__(self):
#         self.prompter = PromptTemplate(template=self.prompt_str, input_variables=["existing_summary", "new_messages"])
#
#     def simple_gen(self):
#         responses = self.llm.generate([self.prompt()])
#         return responses.generations[0][0].text
#
#     def get_info(self, path: List[Node]) -> None:
#         self.new_messages = ""
#         for node in path:
#             if node.node_info.get("summary", None) is not None:
#                 self.latest_summary = node.node_info["summary"]
#             if node.node_info.get("summary", None) is None:
#                 self.new_messages += node.text
#                 self.new_messages += "\n"
#
#     def prompt(self):
#         return self.prompter.format(existing_summary=self.latest_summary, new_messages=self.new_messages)
#
#     def summarize(self, path: List[Node]) -> str:
#         self.get_info(path)
#         self.latest_summary = self.simple_gen()
#         path[-1].node_info["summary"] = self.latest_summary
#         return self.latest_summary


class GPTMultiverseIndex(BaseGPTIndex[IndexGraph]):
    """Multiverse Index.

    The multiverse index is a tree-structured index, which starts with a single
    root node and branches into multiple nodes. During construction, the tree
    is built one node at a time similar to a list index.

    There are a few different query types (see :ref:`Ref-Query`).
    The main option is to traverse down the tree, summarizing the different branches.
    This can be used to summarize discourse in forums, twitter and other
    online media that are structured as a tree.
    """

    index_struct_cls = IndexGraph
    tags: Dict[str, Node] = field(default_factory=dict)
    summary: Optional[str] = None
    name: Optional[str] = None
    generate_embeddings: bool = False
    generate_summaries: bool = False
    cache_size: int = 4
    # summarizer: SummarizeModel = SummarizeModel()
    latest_summary: str = "None."

    def __init__(self, name=None, summary=None, generate_embeddings=False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.summary = summary
        self.generate_embeddings = generate_embeddings

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTMultiversePathIndexQuery,
            QueryMode.EMBEDDING: GPTTreeIndexEmbeddingQuery,
            # QueryMode.DEFAULT: GPTMultiverseIndexGenerateQuery,   <- mutli generate every branch
            # QueryMode.SUMMARISE: GPTMultiverseIndexSummariseQuery,
        }

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexGraph:
        """Build the index from documents.

        Args:
            documents (List[BaseDocument]): A list of documents.

        Returns:
            IndexGraph: The created graph index.
        """
        index_struct = IndexGraph()
        start_index = 0
        for i, d in enumerate(documents):
            node = self._get_nodes_from_document(d, start_index + i)
            index_struct.add_root_node(node)
        return index_struct

    def _update_index_registry_and_docstore(self) -> None:
        """Update index registry and docstore."""
        super()._update_index_registry_and_docstore()
        if len(self._index_struct.root_nodes) == 1 and len(self._index_struct.all_nodes) == 1:
            self.checkout_path(self._index_struct.root_nodes[0])
        self.tags = {}

    def _get_nodes_from_document(
        self,
        document: BaseDocument,
        start_idx: int = 0,
    ) -> List[Node]:
        if not document.embedding and self.generate_embeddings:
            document.embedding = self._embed_model.get_text_embedding(document.text)
        return Node(
            text=document.text,
            index=start_idx,
            ref_doc_id=document.get_doc_id(),
            embedding=document.embedding,
            extra_info=document.extra_info if self._include_extra_info else None,
            node_info={},
        )

    def checkout_path(self, node: Node) -> None:
        """Checkout a path in the index.
        The tree is traversed from the chosen node to the root node,
        and every node is labelled as "checked_out".

        Args:
            node (Node): The node to checkout.

        Returns:
            None
        """
        self.clear_checkout()
        node.node_info["checked_out"] = True
        while node.index not in self.index_struct.root_nodes.keys():
            node = self.index_struct.get_parent(node)
            if node is None:
                raise ValueError("Node has no parent and is not a root node. Graph is corrupt.")
            node.node_info["checked_out"] = True

    def checkout(self, identifier: Union[int, str]) -> None:
        """Checkout a node in the index."""
        if identifier in self.tags.keys():
            node = self.index_struct.all_nodes[self.tags[identifier]]
        else:
            node = self.index_struct.get_node(identifier)
        # TODO: do an vector search if node is None

        if node is None:
            return
        self.checkout_path(node)

    def cherry_pick(self, identifiers: List[Union[int, str]]) -> None:
        """Cherry pick a list of nodes in the index."""
        nodes = []
        for identifier in identifiers:
            if identifier in self.tags.keys():
                node = self.index_struct.all_nodes[self.tags[identifier]]
            else:
                node = self.index_struct.get_node(identifier)
            nodes.append(node)
        for node in nodes:
            self.extend(node)

    def clear_checkout(self) -> None:
        """Clear checkout."""
        for node in self.index_struct.all_nodes.values():
            node.node_info.pop("checked_out", None)
        for node in self.index_struct.root_nodes.values():
            node.node_info.pop("checked_out", None)

        self.latest_summary = LATEST_SUMMARY

    def repr(self, node):
        if isinstance(node, int):
            node = self.index_struct.all_nodes[node]
        return self.index_struct._get_repr(node)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        path = self.path
        return "\n".join([node.text for node in path])

    def short_path(self) -> str:
        """Get the context plus the last summary plus the cache."""
        context_str = self.context

        summary_str = f"Conversation Summary:\n{self.latest_summary}\n\n"

        cache_str = "Recent Messages:\n"
        cache = self.path[-self.cache_size: ]
        cache_str += "\n".join([node.text for node in cache])

        return context_str + summary_str + cache_str

    @property
    def path(self) -> List[Node]:
        """Get the current path.

        Returns:
            List[Node]: The current path.
        """
        for root in self.index_struct.root_nodes.keys():
            # The root nodes are not a copy of the instances in all_nodes
            # so they don't know if they are checked out.
            node = self.index_struct.all_nodes[root]
            if node.node_info.get("checked_out", False):
                return self._get_checked_out_path({node.index: node}, [])
        return []

    def _get_checked_out_path(self, nodes: List[Node], path: List[Node]) -> List[Node]:
        """Get path from root to leaf via checked out nodes."""
        for node in nodes.values():
            if node.node_info.get('checked_out', False):
                path.append(node)
                return self._get_checked_out_path(self.index_struct.get_children(node), path)
        return path

    def step(self, direction: str) -> None:
        """
        Move up or down the tree.

        When moving down, the first child node is selected.
        """
        if direction in ["w", "up", "k"]:
            direction = "up"
        elif direction in ["s", "down", "j"]:
            direction = "down"
        elif direction in ["a", "left", "h"]:
            direction = "smaller_sibling"
        elif direction in ["d", "right", "l"]:
            direction = "larger_sibling"

        if direction == "up":
            parent = self.index_struct.get_parent(self.path[-1])
            if not parent:
                return
        elif direction == "down":
            children = self.index_struct.get_children(self.path[-1])
            if not children:
                return
            self.checkout_path(children[min(children.keys())])
        elif direction in ["smaller_sibling", "larger_sibling"]:
            siblings = self.index_struct.get_siblings(self.path[-1], include_self=True)
            self._step_sibling(direction, siblings)

    def _step_sibling(self, direction: str, siblings: Dict[int, Node]) -> None:
        """
        Step to directional sibling, but loop around if at end.
        """
        sib_indexes = sorted(siblings.keys())
        if direction == "smaller_sibling":
            sib_indexes = list(reversed(sib_indexes))

        # If the current node is the lowest or highest index, then we need to
        # loop around to the other end of the list.
        current_index = self.path[-1].index
        position_in_siblings = sib_indexes.index(current_index)
        if position_in_siblings == len(sib_indexes) - 1:
            self.checkout_path(siblings[sib_indexes[0]])
        else:
            self.checkout_path(siblings[sib_indexes[position_in_siblings + 1]])

    def tag(self, tag: str) -> None:
        """Tag the current path."""
        self.tags[tag] = self.path[-1].index

    def _insert(self, document: Optional[BaseDocument] = None, node: Optional[Node] = None, **_: Any) -> None:
        """Insert a document."""
        if node and document:
            raise ValueError("Cannot insert both a node and a document.")
        if document:
            node = self._get_nodes_from_document(document, self.index_struct.size)
        if len(self.path):
            current_node = self.path[-1]
            self.index_struct.insert_under_parent(node, current_node)
        else:
            self.index_struct.add_root_node(node)

        if self.generate_summaries:
            self.generate_summary()

    def add_context(self, context: str, node: Optional[Node] = None) -> None:
        """Add a global context."""
        node = node or self.path[0]
        context = Document(text=context)
        node.node_info["context"] = context.doc_id
        self.docstore.add_documents([context])

    @property
    def context(self):
        context = [self.get_context(node) for node in self.path]
        context = [doc.text for doc in context if doc]
        for text in context:
            if Path(text).exists():
                with open(text, "r") as f:
                    context[context.index(text)] = f.read()
        context_str = ""
        if context:
            context_str = "Context:\n"
            context_str += "\n".join(context)
            context_str += "\n\n"
        return context_str

    def get_context(self, node: Optional[Node] = None) -> Optional[Document]:
        """Get the global context."""
        node = node or self.path[0]
        context_id = node.node_info.get("context", None)
        return self.docstore.get_document(context_id) if context_id else None

    def delete_all_context(self) -> None:
        """Delete all global contexts."""
        for node in self.index_struct.all_nodes.values():
            self.delete_context(node)

    def delete_path_context(self) -> None:
        """Delete all global contexts."""
        for node in self.path:
            self.delete_context(node)

    def delete_context(self, node: Optional[Union[Node, int]] = None) -> None:
        """Delete the global context."""
        if isinstance(node, int):
            node = self.index_struct.all_nodes[node]
        node = node or self.path[0]
        context_id = node.node_info.get("context")
        if context_id is None:
            return
        self.docstore.delete_document(context_id)
        del node.node_info["context"]

    def generate_summary(self, node: Optional[Node] = None, refresh: bool = False) -> None:
        """Generate summary for current path."""
        if len(self.path) < self.cache_size:
            return

        if node is None:
            node = self.path[-1]

        if refresh:
            path = self.path
        else:
            path = self.get_unsummarized_path(node)

        if len(path) <= self.cache_size:
            return

        self.latest_summary = self.summarizer.summarize(path)

    def get_unsummarized_path(self, node: Node, path: Optional[List[Node]] = None) -> List[Node]:
        """Get path to root."""
        path = self._get_unsummarized_path(node, path or [node])
        return list(reversed(path))

    def _get_unsummarized_path(self, node: Node, path: List[Node]) -> List[Node]:
        parent_node = self.index_struct.get_parent(node)
        # If there is no parent, return the path
        if parent_node is None:
            return path
        # Otherwise, append the parent
        path.append(parent_node)
        # If the parent has a summary, return the path
        if parent_node.node_info.get("summary", None) is not None:
            return path
        # Otherwise, recurse
        return self._get_unsummarized_path(parent_node, path)

    def extend(self, document: Union[BaseDocument, Node]) -> None:
        self._insert(document=document)
        self.checkout_path(self.index_struct.last_node)

    def new(self, document: BaseDocument) -> None:
        """Create a new branch."""
        self.clear_checkout()
        self._insert(document=document)
        self.checkout_path(self.index_struct.last_node)

    def embeddings(self) -> None:
        """Generate embeddings for all documents."""
        for node in self.index_struct.all_nodes.values():
            if node.embedding is None:
                node.embedding = self._embed_model.get_text_embedding(node.text)

    def get_node_similarities(self, query: str) -> Dict[int, float]:
        """Get cosine similarity between node and query."""
        embeddings = [node.embedding for node in self.index_struct.all_nodes.values()]
        embeddings = np.array(embeddings)
        query_embedding = np.array(self._embed_model.get_text_embedding(query))
        similarities = self.cosine_similarity(embeddings, query_embedding)
        for node, similarity in zip(self.index_struct.all_nodes.values(), similarities):
            node.node_info["similarity"] = similarity
        return similarities

    def clear_node_similarities(self) -> Dict[int, float]:
        """Clear cosine similarity between node and query."""
        for node in self.index_struct.all_nodes.values():
            node.node_info.pop("similarity", None)

    @staticmethod
    def cosine_similarity(vectors, query_vector):
        norm_vectors = GPTMultiverseIndex.get_norm_vector(vectors)
        norm_query_vector = GPTMultiverseIndex.get_norm_vector(query_vector)
        similarities = np.dot(norm_vectors, norm_query_vector.T)
        return similarities

    @staticmethod
    def get_norm_vector(vector):
        if len(vector.shape) == 1:
            return vector / np.linalg.norm(vector)
        else:
            return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

    @classmethod
    def load_from_dict(
        cls, result_dict: Dict[str, Any], **kwargs: Any
    ) -> "BaseGPTIndex":
        """Load index from dictionary."""
        index = super().load_from_dict(result_dict, **kwargs)
        if "tags" in result_dict:
            index.tags = result_dict["tags"]
        if "summary" in result_dict:
            index.summary = result_dict["summary"]
        if "name" in result_dict:
            index.name = result_dict["name"]
        if "generate_summaries" in result_dict:
            index.generate_summaries = result_dict["generate_summaries"]
        if "cache_size" in result_dict:
            index.cache_size = result_dict["cache_size"]
        if "latest_summary" in result_dict:
            index.latest_summary = result_dict["latest_summary"]
        return index

    def save_to_dict(self, **save_kwargs: Any) -> dict:
        """Save index to dictionary."""
        result_dict = super().save_to_dict(**save_kwargs)
        result_dict["tags"] = self.tags
        result_dict["summary"] = self.summary
        result_dict["name"] = self.name
        result_dict["generate_summaries"] = self.generate_summaries
        result_dict["cache_size"] = self.cache_size
        result_dict["latest_summary"] = self.latest_summary
        return result_dict
