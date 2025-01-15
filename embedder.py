import glob
from mydoc import MyDoc
from constantscls import Consts
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

class Embedder(Consts):

    GPTembed = "text-embedding-3-small"
    def __init__(self, db_name="chromadb"):
        """
        In call a chroma db is created named as specified.
        Call vectorize() to create a collection and add the data stored in _chunks list.
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
            yield _id, title, text, color

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


    ## CALLABLE METHODS ##
    def chunk_docs(self, chunking_type=None, color=None):
        """
        Chunks the documents in a specified type.
        :param chunking_type: Constant from Conts class.
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
        Also, if specified chunks the documents in the desired method.
        If specified a color metadata will be added.
        :param colors: List of hexadecimal colors to add as metadata in the chunks.
        :param chunking_type: The chunking method of the documents.
        :param directory: the path of the directory where the pdfs to load are located, should "smth/*".
        :return: None
        """

        # Load documents
        doc_paths = glob.glob(directory) # Load document paths
        for i in range(len(doc_paths)):
            color = colors[i] if colors else None
            doc_path = doc_paths[i]
            self._docs.append(MyDoc(doc_path, chunking_type=chunking_type, color=color)) # Save docs in a list

        # Load Chunks if specified
        if chunking_type:
            self.chunk_docs()

    def get_docs(self) -> list[MyDoc]:
        """
        Gets the loaded pdfs in MyDoc format
        :return: list[MyDoc]
        """
        return self._docs

    def get_chunks(self) -> list:
        """
        Gets the loaded chunks from the loaded documents
        :return: list[chunks]
        """
        if self._chunks:
            return self._chunks
        else:
            raise Exception("You need to chunk the loaded documents first!!! Use chunk_docs() method")

    def vectorize(self, embedding_model="text-embedding-3-small", collection_name="collection"):
        """
        Creates a collection named as specified and embedds the chunked data.
        If the name is the same as an existing collection nothing will happen.
        :param embedding_model: The embedding model to be used. Default is gpt embeddings using text-embedding-3-small model.
        :param collection_name: The name of the collection in the vector db.
        :return: The path of the vector DB created
        """

        # Set embedding model
        self._set_embedding_func(model=embedding_model)

        # Check if the specified collection name already exists if not create it.
        if self.collection_exists(collection_name):
            print("Collection exists!")
            return None


        collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self._embedding_method,
            metadata={
                "hnsw:space": "cosine"
            }
        )

        # Add the data
        collection.add(
            documents=[text for _, _, text, _ in self._vectors_generator()],
            metadatas=[
                {
                    "id": _id,
                    "title": title,
                    "color": color,
                } for _id, title, text, color in self._vectors_generator()
            ],
            ids=[_id for _id, _, _, _ in self._vectors_generator()]
        )

    def add_to_vectordb(self):
        pass

    def delete_collections(self, collections_to_delete=None) -> list[str]:
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
                    if self.collection_exists(collection):
                        self.chroma_client.delete_collection(collection)
                        deleted_collections.append(collection)
                    else:
                        print(f"Collection: {collection} does not exist.")
        return deleted_collections


    def search_vectordb(self):
        pass

    def similarity_check(self):
        pass

    def count(self, collection_name) -> int:
        """
        Returns the number of items in the specified collection.
        :param collection_name: The name of the collection.
        :return: int
        """
        if self.collection_exists(collection_name):
            return self.chroma_client.get_collection(collection_name).count()

    def collection_exists(self, collection_name) -> bool:
        """
        Checks if collection exists.
        :param collection_name:
        :return: True if collection exists False otherwise.
        """
        try:
            self.chroma_client.get_collection(name=collection_name)
        except:
            return False
        else:
            return True

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


embedder = Embedder()
embedder.load_docs(chunking_type=Embedder.ByChar, directory="aiani dedomena/2009-04-22-14-52-16.pdf")
embedder.vectorize(collection_name="fgfg")
print(embedder.count(collection_name="fgfg"))
print(embedder.delete_collections(collections_to_delete="all"))


