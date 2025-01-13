import glob
from mydoc import MyDoc
from constantscls import Consts
from openai import OpenAI
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient


class Embedder(Consts):

    GPTembed = "text-embedding-3-small"
    def __init__(self):

        # Load env variables
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Initializations
        self._docs: list[MyDoc] = []
        self._chunks = []
        self._vectorDBs = []
        self.client = OpenAI()

    ## PRIVATE METHODS
    def _get_embedding_gpt(self, text, dimensions=1024):
        """
        Creates the vector using the specified embedding model.
        :param text: The text to embed.
        :param dimensions: The dimensions of the vector
        :return: The vector representing the given text
        """
        return self.client.embeddings.create(input=[text], model=Embedder.GPTembed, dimensions=dimensions).data[0].embedding

    def _vectors_generator(self, dimensions=1024):
        """
        A generator that returns vector representations if the loaded chunks.
        """

        for chunk in self.get_chunks():
            vector = self._get_embedding_gpt(chunk.page_content, dimensions)
            _id = chunk.metadata["_id"]
            title = chunk.metadata["title"]
            text  = chunk.metadata["page_content"]
            color = chunk.metadata.get("color", "black") #TODO put the hexadecimal value if needed
            yield _id, vector, title, text, color

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

    def vectorize(self, vectordb_name=None, collection_name="collection", dimensions=1024):
        """
        Creates the VectoDB named as specified.
        If the name is the same as an existing vectorDB the new one will have an auto increasing number.
        :param dimensions: The dimensions that the vectors will be represented.
        :param collection_name: The name of the collection in the vector db.
        :param vectordb_name: The name of the vector database. Make sure to include .db in the end
        :return: The path of the vector DB created
        """

        if ".db" not in vectordb_name:
            raise Exception("Please include the '.db' at the end of the vector database name!!!")

        ## TODO CREATE THE CODE FOR SAME VECTOR DB NAMES

        client = MilvusClient(vectordb_name) # Create the vectorDB named as...
        client.create_collection(
            collection_name=collection_name,
            dimension=dimensions
        )

        # Prepare the data to for saving in the vector DB
        data = []
        for _id, vector, title, text, color in self._vectors_generator(dimensions):
            data.append(
                {
                    "id": _id,
                    "vector": vector,
                    "title": title,
                    "text": text,
                    "color": color,
                    "subject": title
                }
            )
        res = client.insert(collection_name=collection_name, data=data)

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
embedder.vectorize("vectordb.db")


