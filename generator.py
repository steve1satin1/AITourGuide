from embedder import Embedder
from openai import OpenAI
from dotenv import load_dotenv
import os

class Generator:

    def __init__(self, embedder: Embedder, collection_name, n_results=3):
        """
        Generator that answers questions using the model used to generate the specified collection.
        Just create an instance of the class and use the generate_answer() method to answer questions.
        :param embedder: The embedder to use for generating the relevant to the user's question context.
        :param collection_name: The name of the collection that stores the relevant chunks to the user's question context.
        :param n_results: The number of relevant chunks to return. Default is 3.
        """
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Initializations
        self._gpt_client = OpenAI()
        self._system_prompt = ("Είσαι ένας ξεναγός του αρχαιολογικού μουσείου Αιανής που βρισκεται στην Κοζάνη (μια μικρή πόλη στην Ελλάδα). "
                               "Στόχος σου είναι να απαντάς στις ερωτήσεις που κάνουν οι επισκέπτες. "
                               "Για κάθε ερώτηση θα σου παρέχεται σχετικά κομμάτια κειμένου τα οποία μπορείς να συμβουλευτείς για να απαντήσεις στην ερώτηση του χρήστη."
                               "Στην περίπτωση που δεν γνωρίζεις την απάντηση στην ερώτηση που έθεσε ο χρήστης πες με ευγενικό τρόπο πως δεν γνωρίζεις την απάντηση και μήπως θέλει να ρωτήσει κάτι άλλο.")
        self._model = "gpt-4o-mini"
        self._conversation = [
            {"role": "system", "content": self._system_prompt}
        ]

        self._embedder = embedder
        self._collection_name = collection_name
        self._n_results = n_results

    ## ====== PRIVATE METHODS ====== ##
    def _prepare_prompt(self, question: str) -> str:
        """
        Creates a prompt for the GPT model.
        The prompt will be added to the self._conversation.
        :param question: The user's question.
        :return: The prompt for the GPT model.
        """

        prompt = ""
        prompt += f"{question}\n\n"
        prompt += "Παρακαλώ συμβουλεύσου τα παρακάτω σχετικά με την ερώτηση κείμενα πριν απαντήσεις: \n\n"
        for chunk in self._embedder.search_similar(self._collection_name, question, n_results=self._n_results):
            prompt += chunk + "\n\n"

        print(prompt)
        return prompt

    def _fetch_conversation(self) -> list[dict[str, str]]:
        """
        Fetches the conversation from the self._conversation. Raise exception if empty.
        :return: List of conversation messages
        """

        if self._is_conversation_empty():
            raise Exception("Conversation is empty. Use _update_conversation() to add messages to the conversation.")
        return self._conversation


    def _get_answering_fn(self) -> callable:
        """
        Gets the answering function based on the model.
        :return: The function to be used for answering.
        """
        gpt_models = ["chatgpt-4o-latest", "gpt-4o-mini", "o1-preview"]
        if self._model in gpt_models:
            return self._gpt_answering_fn
        else:
            raise Exception(f"Currently the only supported models are {', '.join(gpt_models)}")

    def _update_conversation(self, role, content) -> None:
        """
        Updates the conversation with the given role and content.
        :param role: The role of the message.
        :param content: The content of the message.
        :return: None
        """
        self._conversation.append({"role": role, "content": content})

    def _gpt_answering_fn(self):
        """
        Generates an answer using the GPT model.
        The method _update_conversation() should be called before calling this method.
        :return: The generated answer as a generator of strings. Each string represents a chunk of the answer.
        """

        # Change conversation to meet openai standards
        conversation = self._fetch_conversation()
        conversation[0]["role"] = "developer"

        stream = self._gpt_client.chat.completions.create(
            model=self._model,
            messages=conversation,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _is_conversation_empty(self) -> bool:
        """
        Checks if the conversation is empty.
        :return: True if empty else False
        """
        return True if not self._conversation else False

    ## ====== CALLABLE METHODS ====== ##
    def generate_answer(self, question, model) -> None:
        """
        Generates an answer to the given question using the given model.
        :param model: The model to use for generating the answer.
        :param question: The question to answer.
        :return: None
        """
        # Set the model to the specified one
        self._model = model

        # Prepare the prompt
        prompt = self._prepare_prompt(question)

        # Update the conversation list
        self._update_conversation("user", prompt)

        # Get the answering fn based on the given model
        answering_fn = self._get_answering_fn()

        answer = ""
        for chunk in answering_fn():
            print(chunk, end="")
            answer += chunk

        # Save answer to the conversation
        self._update_conversation("assistant", answer)

    def get_conversation(self):
        pass


embedder = Embedder()
embedder.load_docs(directory="aiani dedomena/*", chunking_type=Embedder.ByChar)

if not embedder.collection_exists("Mycollection"):
    embedder.add_data("Mycollection")

gen = Generator(embedder=embedder, collection_name="Mycollection", n_results=5)

while True:
    gen.generate_answer(input("Ask a question: "), model="gpt-4o-mini")
    print("\n")

# embedder.visualize("Mycollection", dimensions=["2d", "3d"])
