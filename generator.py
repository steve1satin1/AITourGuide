from embedder import Embedder
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

class Generator:

    def __init__(self, embedder: Embedder, collection_name, n_results=3, model="gpt-4o-mini"):
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
                               "Υπάρχει περίπτωση ο χρήστης να σου γράψει κάτι το οποίο δεν χρειάζεται να συμβουλευτείς "
                               "τα σχετικά κομμάτια κειμένου για να απαντήσεις, όπως για παράδειγμα 'Ευχαριστώ πολύ' ή "
                               "'γειά σου' σε αυτές τις περιπτώσεις μην λάβεις υπόψην σου τα σχετικά κομμάτια κειμένου που θα σου δοθούν."
                               "Στην περίπτωση που δεν γνωρίζεις την απάντηση στην ερώτηση που έθεσε ο χρήστης πες με ευγενικό τρόπο πως δεν γνωρίζεις την απάντηση και μήπως θέλει να ρωτήσει κάτι άλλο."
                               "Σε κάθε κομμάτι κειμένου που σου παρέχεται θα υπάρχει και η πηγή απο την οποία προήλθε και θα αναγράφεται στο τέλος του μετά την λέξη κλειδί «Πηγή:», "
                               "αν χρησιμοποιήσεις κάποια απο τα κομμάτια αυτά στο τέλος της απάντησης σου παρέθεσε της πηγές απο τα κομμάτια κειμένου που χρησιμοποίησες γράφοντας «Πηγές: (αναφορά των πηγών σε bullets)»"
                               "Μην βάζεις δικές σου πηγές αλλά μόνο αυτές που αναφέρονται σε κάθε κομμάτι κειμένου μετά την λέξη κλειδή «Πηγή:»")

        self._model = model

        self._conversation = [
            {"role": "system", "content": self._system_prompt},
            {"role": "assistant", "content": "Γειά σας είμαι ψηφιακός βοηθός του μουσείου Αιανής πως μπορώ να σας βοηθήσω;"}
        ]

        self._embedder = embedder
        self._collection_name = collection_name
        self._n_results = n_results

    ## ====== PRIVATE METHODS ====== ##
    def _prepare_prompt(self, question: str) -> tuple[str, list[str]]:
        """
        Creates a prompt for the GPT model.
        The prompt will be added to the self._conversation.
        :param question: The user's question.
        :return: The prompt for the GPT model and the contexts if needed.
        """

        prompt = ""
        prompt += f"{question}\n\n"
        prompt += "Παρακαλώ συμβουλεύσου τα παρακάτω σχετικά με την ερώτηση κείμενα πριν απαντήσεις εάν η ερώτηση του χρήστη το απαιτεί: \n\n"

        similars = self._embedder.search_similar(self._collection_name, question, n_results=self._n_results)
        texts = similars[0]
        sources = similars[1]
        for chunk, source in tuple(zip(texts, sources)):
            prompt += chunk + " Πηγή: "+ source + "\n\n"

        print(prompt) # TODO Delete this line
        return prompt, texts

    def _fetch_conversation(self) -> list[dict[str, str]]:
        """
        Fetches the conversation from the self._conversation. Raise exception if empty.
        :return: List of conversation messages
        """

        if self._is_conversation_empty():
            raise Exception("Conversation is empty. Use _update_conversation() to add messages to the conversation.")
        return self._conversation


    def _get_answering_fn(self, streaming=True) -> callable:
        """
        Gets the answering function based on the model.
        :param streaming: weather or not to stream the answer, default to True.
        :return: The function to be used for answering.
        """
        gpt_models = ["chatgpt-4o-latest", "gpt-4o-mini", "o1-preview", "gpt-4o"]
        if self._model in gpt_models:
            if streaming:
                return self._gpt_answering_fn
            else:
                return self._gpt_answering_fn_non_stream
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

    def _gpt_answering_fn_non_stream(self):
        """
        Generates an answer using the GPT model.
        The method _update_conversation() should be called before calling this method.
        :return: The generated answer as a generator of strings. Each string represents a chunk of the answer.
        """

        # Change conversation to meet openai standards
        conversation = self._fetch_conversation()
        conversation[0]["role"] = "developer"

        completion = self._gpt_client.chat.completions.create(
            model=self._model,
            messages=conversation,
        )

        return completion.choices[0].message.content

    def _is_conversation_empty(self) -> bool:
        """
        Checks if the conversation is empty.
        :return: True if empty else False
        """
        return True if not self._conversation else False

    def _make_sources_links(self, text: str) -> str | None:
        """
        Makes the sources links that open in a new tab
        :param text: The text to make the sources links for
        """
        if "Πηγές" in text:
            start_index = text.index("Πηγές:")
            sources_txt = text[start_index:]
            sources_list = sources_txt.split("\n")[1:]
            for source in sources_list:
                n_source = source.strip('- ').replace("\\", "/")
                # text = text.replace(source, f"<a href='file///{n_source}'>{n_source}</a>")
                text = text.replace(source, "<a href='file:///C:/Users/στεργιος/PycharmProjects/Diplomatiki/aiani dedomena/megaloi_domoi.pdf'>link</a>")
            print("sources: ", text)
            return text

    def _prework_for_answ_gen(self, question, model, streaming=True):
        """
        Preparation for answer generation. Sets the model to the specified one,
        Prepares the prompt, updates the conversation list and gets the answering fn based on the given model.
        :param question: The question to answer.
        :param model: The model to use for generating the answer.
        :param streaming: weather or not to stream the answer.
        :return: The answering function
        """
        # Set the model to the specified one
        self._model = model

        # Prepare the prompt
        prompt, _ = self._prepare_prompt(question)

        # Update the conversation list
        self._update_conversation("user", prompt)

        # Get the answering fn based on the given model
        answering_fn = self._get_answering_fn(streaming=streaming)

        return answering_fn



    ## ====== CALLABLE METHODS ====== ##
    def generate_answer(self, question, model):
        """
        Generates an answer to the given question using the given model.
        :param model: The model to use for generating the answer.
        :param question: The question to answer.
        :return: generator of strings. Each string represents a chunk of the answer.
        """

        # Get the answering fn based on the given model
        answering_fn = self._prework_for_answ_gen(question, model, streaming=True)

        answer = ""
        for chunk in answering_fn():
            yield chunk
            answer += chunk

        # Save only the user's question
        self._conversation[-1]["content"] = question

        # Save answer to the conversation
        self._update_conversation("assistant", answer)

    def generate_answer_non_steam(self, question, model) -> str:
        """
        Generates answer without streaming it.
        :param question: The question to answer.
        :param model: The model to use for generating the answer.
        :return: Answer as a string
        """

        # Get the answering fn based on the given model
        answering_fn = self._prework_for_answ_gen(question, model, streaming=False)

        # Generate answer
        answer = answering_fn()

        # Save only the user's question
        self._conversation[-1]["content"] = question

        # Save answer to the conversation
        self._update_conversation("assistant", answer)

        return answer

    def generate_answer_structured(self, question) -> tuple[str, list[str]]:
        """
        Generates answer and provided contexts as structured output.
        :param question: The question for the model to answer.
        :param model: The model to use for answering the question.
        :return: Tuple(answer, contexts)
        """
        # Set a specific system prompt for evaluation
        self._conversation[0]['content'] = (
            "Είσαι ένας ξεναγός του αρχαιολογικού μουσείου Αιανής που βρισκεται στην Κοζάνη (μια μικρή πόλη στην Ελλάδα). "
            "Στόχος σου είναι να απαντάς στις ερωτήσεις που κάνουν οι επισκέπτες. "
            "Για κάθε ερώτηση θα σου παρέχεται σχετικά κομμάτια κειμένου τα οποία μπορείς να συμβουλευτείς για να απαντήσεις στην ερώτηση του χρήστη."
            "Υπάρχει περίπτωση ο χρήστης να σου γράψει κάτι το οποίο δεν χρειάζεται να συμβουλευτείς "
            "τα σχετικά κομμάτια κειμένου για να απαντήσεις, όπως για παράδειγμα 'Ευχαριστώ πολύ' ή "
            "'γειά σου' σε αυτές τις περιπτώσεις μην λάβεις υπόψην σου τα σχετικά κομμάτια κειμένου που θα σου δοθούν."
            "Στην περίπτωση που δεν γνωρίζεις την απάντηση στην ερώτηση που έθεσε ο χρήστης πες με ευγενικό τρόπο πως δεν γνωρίζεις την απάντηση και μήπως θέλει να ρωτήσει κάτι άλλο."
            "Οι απαντήσεις θα πρέπει να είναι λιτές και να περιέχουν μόνο την απάντηση στην ερώτηση όχι περιττές πληροφορίες.")

        # Prepare the prompt
        prompt, contexts = self._prepare_prompt(question)

        # Update the conversation list
        self._update_conversation("user", prompt)

        # Get the answering fn based on the given model
        answering_fn = self._get_answering_fn()

        answer = ""
        for chunk in answering_fn():
            answer += chunk

        # Save only the user's question
        self._conversation[-1]["content"] = question

        # Save answer to the conversation
        self._update_conversation("assistant", answer)

        return answer, contexts

    def get_text_from_audio(self, path) -> str:
        """
        Generates text representation from given audio file path.
        :param path: The path of the audio file
        :return: str
        """

        with open(path, "rb") as audio_file:
            transcript = self._gpt_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="el",
                response_format="text"
            )
        print(f"What came from audio api: {transcript}")
        return transcript

    def text_to_speech(self, text) -> None:
        """
        Streams the generated audio file from the given text.
        :param text: The text to generate to audio.
        :return: None
        """

        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = self._gpt_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        response.stream_to_file(speech_file_path)
        print("finished audio gen...")

    def get_conversation(self):
        pass


# embedder = Embedder()
# embedder.load_docs(directory="aiani dedomena/*", chunking_type=Embedder.ByChar)
#
# if not embedder.collection_exists("Mycollection"):
#     embedder.add_data("Mycollection")
#
# gen = Generator(embedder=embedder, collection_name="Mycollection", n_results=5)
#
# while True:
#     print("==================================================")
#     gen.generate_answer(input("Ask a question: "), model="gpt-4o-mini")
#     print("\n")
#     print("==================================================")
#     print(f"conversation:\n{gen._fetch_conversation()}\n\n")

# embedder.visualize("Mycollection", dimensions=["2d", "3d"])
