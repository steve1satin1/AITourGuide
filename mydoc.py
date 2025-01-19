from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constantscls import Consts

class MyDoc(Consts):

    # CONSTANTS
    EXCLUDE = [". \xa0 ", ".   ","1 / 3 ΣΤΩΪΚΟ ΚΤΗΡΙΟ \xa0 ", "\xa0", "\xa0\xa0\xa0",
               "\xa0\xa0 \xa0\xa0\xa0\xa0\xa0\xa0", "2 / 3 ΣΤΩΪΚΟ ΚΤΗΡΙΟ", "\xa0 \xa0 \xa0", "3 / 3"]

    def __init__(self, filepath: str, chunking_type=None, color=None):
        """
        PDF Parser, can chunk the document's text. Just specify the pdf path on the constructor!
        You can also specify a chunking type so that the chunking could be done immediately.
        If you specify a color it will be added in the chunk metadata.

        :param filepath: the path of the document to load
        :param chunking_type: The chunking type. If not specified the chunking will not happen. You can do it later.
        :param color: If specified the color to add as metadata. If not you can add it later.
        """
        ### INITIALIZATION ###
        self._loader = PyPDFLoader(filepath)
        self._pages = self._loader.load()
        self._text = None
        self._text_splitter = None
        self._chunks = None
        self._title = None

        ### ACTIONS ###
        self._load_text()  # Load text and title

        if chunking_type:
            self.chunk_document(chunking_type=chunking_type, color=color)


    def _chunk_by_char(self, chunk_size=500, chunk_overlap=20, length_function=len) -> None:
        """Chunks the text recursively. The end result is close to chunking by paragraph"""
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            length_function=length_function,
            separators=["\n\xa0", "\n\n", "."]
        )

        chunks: list = self._text_splitter.split_text(self._text)

        self._chunks = self._text_splitter.create_documents(
            chunks,
            metadatas=[
                {
                    "title": self._title,
                    "_id": f"{self._title}-{i}"
                } for i in range(len(chunks))]
        )


    def get_text(self) -> str:
        return self._text

    def get_pages(self) -> list:
        return self._pages

    def get_title(self) -> str:
        return self._title

    def _load_text(self):
        """Extracts the text from the pdf pages"""

        # Basic text extraction from pdf
        text = " ".join([page.page_content for page in self._pages])
        self._title = self._find_title(text)
        text = text.replace("\n\n", " ").replace("\n", " ")
        # Title extraction
        self._text = text

    def _find_title(self, text: str) -> str:
        """
        Finds the title of a given text.
        :param text: The text to find its title.
        :return: The title.
        """
        stoppers = ["\n\n", "\n", "  ", "\xa0"]
        stop_index = [ind for ind in stoppers if text.find(ind)!=-1]

        if stop_index:
            title = text[:text.index(stop_index[0])]
            if len(title) > 20:
                title = text[:6]
        else:
            title = text[:10]
        return title


    def chunk_document(self, chunking_type=Consts.ByChar, color=None) -> None:
        """
        Chunks document depending on the chunking type specified
        :param chunking_type: The chunking method of the document's text.
        :param color: The hexadecimal code of the color metadata of the chunks.
        :return: None
        """
        if chunking_type == MyDoc.ByChar:
            self._chunk_by_char()
            if color:
                self.specify_color(hexadecimal_code=color)
        else:
            Exception("You need to set the chunking type!!!")
        self._clear_chunks()
        self._add_title_to_chunks()

    def get_chunks(self) -> list:
        """
        Gets the chunks. Need to chunk the document first!!
        :return: list[Document]
        """
        if self.has_chunks():
            return self._chunks
        raise Exception("You need to chunk the document first")

    def _clear_chunks(self):
        """Clears the chunks from unwanted characters. Should be called after the chunks have benn created!!!"""
        chunks = []
        for chunk in self._chunks:
            for item in MyDoc.EXCLUDE:
                chunk.page_content = chunk.page_content.replace(item, "")
            chunks.append(chunk)
        self._chunks = chunks

    def specify_color(self, hexadecimal_code=None):
        """Creates a new metadata of the color on the chunks created from the instance"""
        if hexadecimal_code:
            if self._chunks:
                for chunk in self._chunks:
                    chunk.metadata["color"] = hexadecimal_code
            else:
                raise Exception("You need to chunk the document first!")
        else:
            raise Exception("You need to specify the color!!!")

    def has_chunks(self) -> bool:
        """
        True if chunking has been done else False
        :return: bool
        """
        return True if self._chunks else False

    def _add_title_to_chunks(self):
        """Adds the title to the chunks text content"""
        if self._chunks:
            for chunk in self._chunks:
                # Add the title in each chunk
                chunk.page_content = f"{self._title}: {chunk.page_content}"
        else:
            raise Exception("You need to chunk the document first!")

    def __repr__(self):
        return f"Title: {self.get_title()}\n\nText: {self.get_text()}\n\n"



