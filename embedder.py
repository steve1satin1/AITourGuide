import glob
from mydoc import MyDoc
from constantscls import Consts

class Embedder(Consts):

    def __init__(self):
        self._docs: list[MyDoc] = []
        self._chunks = []
        self._vectorDBs = []

    ## PRIVATE METHODS

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

    def vectorize(self):
        pass

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
embedder.load_docs(chunking_type=Embedder.ByChar)
print(f"Documents:\n\n{embedder.get_docs()}")
print(f"chunks:\n\n{embedder.get_chunks()}")

