import glob
import random
from typing import Any
from mydoc import MyDoc
from constantscls import Consts
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as plt

class Embedder(Consts):

    GPTembed = "text-embedding-3-small"
    def __init__(self, db_name="chromadb"):
        """
        In call a chroma db is created named as specified.
        Call add_data() to create a collection and add the data stored in _chunks list.
        :param db_name: The name of the database.
        """

        # Load env variables
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Initializations
        self._docs: list[MyDoc] = []
        self._chunks = []
        self._vectorDBs = []
        self.gpt_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=db_name)

        self._embedding_method = None

    ## PRIVATE METHODS
    def _gpt_embedding(self, gpt_model="text-embedding-3-small"):
        """
        Sets the embedding method to use gpt embeddings.
        :param gpt_model: The gpt embedding model.
        :return: The vector representing the given text
        """
        self._embedding_method = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name=gpt_model
            )

    def _vectors_generator(self):
        """
        A generator that returns the metadata of each saved chunk.
        """

        for chunk in self.get_chunks():
            _id = chunk.metadata["_id"]
            title = chunk.metadata["title"]
            text = chunk.page_content
            color = chunk.metadata.get("color", "black") #TODO put the hexadecimal value if needed
            doc_path = chunk.metadata["doc_path"]
            yield _id, title, text, color, doc_path

    def _set_embedding_func(self, model="text-embedding-3-small"):
        """
        Sets the embedding function for the vectorization by changing the self._embedding_method.
        :param model: The model that will be used for embedding.
        :return: None
        """

        # Check if model is an openai model
        openai_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        if model in openai_models:
            self._gpt_embedding(gpt_model=model)
            print(f"Embedding function set successfully! Embedding model is {model}")
        else:
            print(f"Currently the only supported embedding models are {', '.join(openai_models)}")

    def _create_collection(self, collection_name, embedding_model) -> chromadb.Collection:
        """
        Creates an empty collection named as specified. If collection exists raises Exception.
        :param collection_name: The name of the collection
        :param embedding_model: The embedding model.
        :return: The collection created.
        """

        # Set embedding model
        self._set_embedding_func(model=embedding_model)

        # Check if the specified collection name already exists.
        if self._get_collection(collection_name):
            raise Exception("Collection already exists.")

        # Creates and returns the collection
        return self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self._embedding_method,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model
            }
        )

    def _get_collection(self, collection_name) -> Any | None:
        """
        Checks if collection exists. If exists returns it.
        :param collection_name: collection name
        :return: Collection if exists else None.
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except:
            return None
        else:
            self._set_embedding_func(model=collection.metadata["embedding_model"])
            return self.chroma_client.get_collection(name=collection_name, embedding_function=self._embedding_method)

    def _get_collection_error(self, collection_name):
        """
        Checks if collection exists. If exists returns it else raises error.
        :param collection_name: collection name
        :return: Collection if exists.
        """

        col = self._get_collection(collection_name)
        if not col:
            raise Exception("Collection does not exists!!!")
        return col

    def _random_colors_generator(self, num_colors) -> list[str]:
        """
        Generates a list of random colors.
        :param num_colors: The number of colors to generate.
        :return: List of colors in hexadecimal format.
        """
        colors = []
        used_colors = set()
        for _ in range(num_colors):
            while True:
                hex_color = '#' + ''.join([random.choice('0123456789abcdef') for _ in range(6)])
                if hex_color not in used_colors:
                    used_colors.add(hex_color)
                    colors.append(hex_color)
                    break
        return colors

    ## CALLABLE METHODS ##
    def chunk_docs(self, chunking_type=None, color=None):
        """
        Chunks the documents in a specified type.
        :param chunking_type: Constant from Consts class.
        :param color: color to add to the chunks if specified.
        :return: None
        """
        if self.has_loaded_docs():
            for doc in self._docs:
                if doc.has_chunks(): # Chunking has been done
                    self._chunks.extend(doc.get_chunks())
                else:
                    doc.chunk_document(chunking_type=chunking_type, color=color)
                    self._chunks.extend(doc.get_chunks())
        else:
            raise Exception("You need to load the documents first! Use load_docs()")

    def load_docs(self, directory="aiani dedomena/*", chunking_type=None, colors=None) -> None:
        """
        Loads the pdfs in MyDoc parser and saves them in self._docs.
        Also, if chunking type is provided, the documents get chunked in the desired method.
        If specified, a color metadata will be added to the docs.
        :param colors: List of hexadecimal colors to add as metadata in the documents. The length of the list should be equal to the pdfs
        :param chunking_type: The chunking method of the documents.
        :param directory: the path of the directory where the pdfs to load are located, should "smth/*".
        :return: None
        """

        # Load document's paths
        doc_paths = glob.glob(directory) # Load document paths

        # Check for colors
        if colors:
            colors_len = len(colors)
            doc_paths_len = len(doc_paths)
            if colors_len > doc_paths_len:
                raise Exception("You provided more colors than documents.")
            elif colors_len < doc_paths_len:
                raise Exception("You provided less colors than documents length.")
        else:
            colors = self._random_colors_generator(len(doc_paths))
            print(f"Colors generated: {colors}")

        # Load documents
        for i in range(len(doc_paths)):
            color = colors[i]
            doc_path = doc_paths[i]
            self._docs.append(MyDoc(doc_path, chunking_type=chunking_type, color=color)) # Save docs in a list

        if chunking_type:
            self.chunk_docs(chunking_type=chunking_type, color=color)

    def get_docs(self) -> list[MyDoc]:
        """
        Gets the loaded pdfs in MyDoc format
        :return: list[MyDoc]
        """
        return self._docs

    def get_chunks(self, collection_name=None) -> list:
        """
        Gets the loaded chunks from the loaded documents. If a collecion_name is specified the chunks will be from that collection.
        :param collection_name: If specified the chunks of this collection will be fetched else the loaded
        :return: list[chunks]
        """
        if collection_name:
            collection = self._get_collection_error(collection_name)
            return collection.get()

        if self._chunks:
            return self._chunks
        else:
            raise Exception("You need to chunk the loaded documents first!!! Use chunk_docs() method")

    def add_data(self, collection_name, embedding_model="text-embedding-3-small"):
        """
        Adds data to the specified collection. The data are the saved chunks.
        :param embedding_model: The embedding model. Default is 'text-embedding-3-small'.
        :param collection_name: The name of the collection to add the data.
        :return: None
        """
        # If a collection exists get it else create it.
        collection = self._get_collection(collection_name)
        if not collection:
            collection = self._create_collection(collection_name, embedding_model)

        collection.add(
            documents=[text for _, _, text, _, _ in self._vectors_generator()],
            metadatas=[
                {
                    "id": _id,
                    "title": title,
                    "color": color,
                    "doc_path": doc_path
                } for _id, title, text, color, doc_path in self._vectors_generator()
            ],
            ids=[_id for _id, _, _, _, _ in self._vectors_generator()]
        )


    def delete_collections(self, collections_to_delete) -> list[str]:
        """
        Deletes the specified collection.
        :param collections_to_delete: list of collection/s to delete. If 'all' is given all the collections will be deleted
        :return: list of deleted collections
        """

        if not collections_to_delete:
            raise Exception("You need to specify the collection to delete. Give 'all' to delete all collections.")

        existing_collections = self.list_collections()
        deleted_collections = []

        if existing_collections:
            if collections_to_delete == "all":
                for collection in existing_collections:
                    self.chroma_client.delete_collection(collection)
                    deleted_collections.append(collection)
            else:
                for collection in collections_to_delete:
                    if self._get_collection(collection):
                        self.chroma_client.delete_collection(collection)
                        deleted_collections.append(collection)
                    else:
                        print(f"Collection: {collection} does not exist.")
        return deleted_collections


    def search_similar(self, collection_name, *input_text, n_results=20) -> tuple[Any, list[Any]]:
        """
        Searches specified collection for similar text chunks according to given input_text.
        :param n_results: How many results to return.
        :param collection_name: The collection to search to.
        :param input_text: The text chunk/s to search for similar chunks.
        :return: tuple of results where the first index is the text result and the second index is the doc_path of each result.
        """

        query_text = list(input_text)

        # Set embedding function

        # Get collection or raise error if it doesn't exist.
        collection = self._get_collection_error(collection_name)

        # Get results
        results = collection.query(
            query_texts=query_text,
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        return results["documents"][0], [item["doc_path"] for item in results["metadatas"][0]]

    def count(self, collection_name) -> int:
        """
        Counts the number of items in the specified collection.
        :param collection_name: The name of the collection.
        :return: The number of items in the collection
        """
        if self._get_collection(collection_name):
            return self.chroma_client.get_collection(collection_name).count()

    def list_collections(self) -> list[str]:
        """
        Lists all collection names.
        :return: Sequence[CollectionName] - A list of collection names. CollectionName is a string.
        """

        return self.chroma_client.list_collections()



    def has_loaded_docs(self):
        """
        Checks if documents have been loaded or not.
        :return: True if there are any documents False otherwise.
        """
        return True if self._docs else False

    def visualize(self, collection_name: str, dimensions=None) -> None:
        """
        Creates a plot with matplotlib of the embeddings.
        :param collection_name: The name of the collection to visualize.
        :param dimensions: List representing the dimensions to use for the plot. Default is ['2d']. If ['2d', '3d'] is given there will be two plots.
        :return: None
        """

        # Set default dimensions for visualization to 2d
        if dimensions is None:
            dimensions = ["2d"]

        # Extract metadata and embeddings from specified collection
        collection = self._get_collection_error(collection_name)
        results = collection.get(include=["embeddings", "metadatas"])
        ids = results["ids"]
        embeddings = results["embeddings"]
        titles = []
        colors = []
        for i in range(len(results["metadatas"])):
            titles.append(results["metadatas"][i]["title"])
            colors.append(results["metadatas"][i]["color"])

        print(f"ids: {ids}\nembeddings: {embeddings}\ntitles: {titles}\ncolors: {colors}")

        # Reduce dimensionality with tsne package
        original_embeddings = np.array(embeddings)

        # For 2d
        if "2d" in dimensions:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_vectors = tsne.fit_transform(original_embeddings)
            fig = plt.Figure(data=[plt.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode='markers',
                marker=dict(size=5, color=colors, opacity=0.8),
                text=ids,
                hoverinfo='text'
            )])

            fig.update_layout(
                title='2D Chroma Vector Store Visualization',
                scene=dict(xaxis_title='x', yaxis_title='y'),
                width=800,
                height=600,
                margin=dict(r=20, b=10, l=10, t=40)
            )

            fig.show()

        # For 3d
        if "3d" in dimensions:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_vectors = tsne.fit_transform(original_embeddings)
            print(reduced_vectors)
            fig = plt.Figure(data=[plt.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers',
                marker=dict(size=5, color=colors, opacity=0.8),
                text=ids,
                hoverinfo='text'
            )])

            fig.update_layout(
                title='3D Chroma Vector Store Visualization',
                scene=dict(xaxis_title='x', yaxis_title='y'),
                width=800,
                height=600,
                margin=dict(r=20, b=10, l=10, t=40)
            )

            fig.show()

    def collection_exists(self, collection_name):
        """
        Checks if a collection exists.
        :param collection_name: The name of the collection to check.
        :return: True if the collection exists, False otherwise.
        """
        return True if self._get_collection(collection_name) else False


# embedder = Embedder()
# embedder.load_docs(directory="aiani dedomena/to_embed/*", chunking_type=Embedder.ByChar)
# # # print(embedder.get_chunks())
# # # #
# # # #
# embedder.delete_collections("all")
# # # # #
# # # # # # print(embedder.get_chunks())
# embedder.add_data("Mycollection", embedding_model="text-embedding-3-large")
# # # #
# # # # print(embedder.search_similar("Mycollection", "Τι είναι η δεξαμενή?", n_results=3))
# # # #
# embedder.visualize("Mycollection", dimensions=["2d", "3d"])



