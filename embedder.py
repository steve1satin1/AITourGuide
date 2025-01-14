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

    ## PRIVATE METHODS
    def _gpt_embedding(self, gpt_model="text-embedding-3-small"):
        """
        Creates the vector using the specified gpt embedding model.
        :param gpt_model: The gpt embedding model.
        :return: The vector representing the given text
        """
        return embedding_functions.OpenAIEmbeddingFunction(
                api_key="YOUR_API_KEY",
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

    def vectorize(self, ef=None, collection_name="collection"):
        """
        Creates a collection named as specified and adds the chunked data.
        If the name is the same as an existing collection nothing will happen.
        :param ef: The embedding function to be used.
        :param collection_name: The name of the collection in the vector db.
        :return: The path of the vector DB created
        """

        # Check if ef is defined else use gpt embeddings.
        if not ef:
            ef = self._gpt_embedding()

        # Check if the specified collection name already exists if not create it.
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except :
            pass
        else:
            print("collection already exists!")
            return None


        collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=ef,
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

    def delete_vectordb(self):
        pass

    def search_vectordb(self):
        pass

    def similarity_check(self):
        pass

    def has_loaded_docs(self):
        """
        Checks if documents have been loaded or not.
        :return: True if there are any documents False otherwise.
        """
        return True if self._docs else False


embedder = Embedder()
embedder.load_docs(chunking_type=Embedder.ByChar, directory="aiani dedomena/2009-04-22-14-52-16.pdf")
embedder.vectorize(collection_name="nco")


