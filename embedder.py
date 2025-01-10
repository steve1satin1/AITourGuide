import glob
from mydoc import MyDoc

class embedder:

    def __init__(self):
        pass

    def load_docs(self):
        doc_paths = glob.glob("aiani dedomena/*")
        

    def create_vector_store(self):
        pass


s = MyDoc("aiani dedomena/2009-04-22-14-52-16.pdf")
s.chunk_document(chunking_type=MyDoc.ByChar)
s.specify_color("fdfdfdf")
print(s.get_chunks())
