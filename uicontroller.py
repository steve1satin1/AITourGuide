import os
import threading
import gradio as gr
from generator import Generator
from embedder import Embedder

class UIController:

    def __init__(self, n_results=5):
        self.gen = Generator(Embedder(), "Mycollection", n_results=n_results)
        self.embedder = Embedder()
        self._prepare_embedder()

        self._used_audio = False # Weather or not user used audio api

    # ===   PRIVATE METHODS   === #
    def _prepare_embedder(self):
        """
        Loads the documents and creates the collection. Don't use it if a collection already exists!!!
        :return: None
        """

        if not self.embedder.collection_exists("Mycollection"):
            print("Loading documents embeddings...")
            self.embedder.load_docs(directory="aiani dedomena/*", chunking_type=Embedder.ByChar)
            self.embedder.add_data("Mycollection")

    def _user(self, user_message, history: list):
        """
        Creates and updates the history of the conversation.
        :param user_message: The user's message to add to the history.
        :param history: The history of the conversation.
        :return: A tuple that consists of a blank string and the updated history.
        The blank string will be sent at the textbox and the history at the chatbot.
        """

        return "", history + [{"role": "user", "content": user_message}]

    def _bot(self, history: list):
        """
        Generates an answer to the user's question and updates the history.
        :param history: The previous history of the conversation.
        :return: A generator of strings. Each string represents a chunk of the answer.
        The answer will be sent at the chatbot.
        """

        question = history[-1]["content"]
        bot_message = self.gen.generate_answer_non_steam(question, model="gpt-4.1")
        print(type(bot_message))

        # TODO ADD CODE FOR PARALLEL SPEECH CREATION
        # If audio api was used, respond with speech
        if self._used_audio:
            t1 = threading.Thread(target=self.gen.text_to_speech, args=(bot_message,))
            t1.start()

        history.append({"role": "assistant", "content": ""})
        for chunk in bot_message:
            history[-1]['content'] += chunk
            yield history

        # # If audio api was used, respond with speech
        # if self._used_audio:
        #     self.gen.text_to_speech(history[-1]['content'])
        #
        # self._used_audio = False


    def _add_message(self, history, message):
        question = ""

        # Delete previous audio file
        if os.path.exists('speech.mp3'):
            os.remove('speech.mp3')

        if message["files"]:
            question = self.gen.get_text_from_audio(message["files"][0])
            self._used_audio = True
        elif message["text"] != "":
            question = message["text"]

        history.append({"role": "user", "content": question})
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def _play_audio(self):
        if self._used_audio:
            while True:
                if os.path.exists("speech.mp3"):
                    self._used_audio = False
                    return "speech.mp3"
        return None


    # ====  CALLABLE METHODS   === #
    def create_ui(self, share=False):
        """
        Creates the UI.
        :param share: If True, a public link will be generated for the UI, else the host will be localhost.
        :return: None.
        """
        with gr.Blocks() as demo:
            with gr.Row():
                gr.Markdown("AI Ξεναγός!")

            bot = gr.Chatbot(
                type="messages",
                value=[{"role": "assistant", "content": "Γειά σας πως μπορώ να σας βοηθήσω;"}]
            )
            # msg = gr.Textbox()

            # msg.submit(self._user, [msg, bot], [msg, bot], queue=False).then(
            #     self._bot, bot, bot
            # )

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                placeholder="Γράψτε την ερώτηση σας...",
                show_label=False,
                sources=["microphone"],
                stop_btn=False,
                autoscroll=True,
                autofocus=True,
            )

            audio = gr.Audio(
                visible=True,
                autoplay=True,
                interactive=False,
                streaming=True,
            )

            chat_msg = chat_input.submit(
                self._add_message, [bot, chat_input], [bot, chat_input]
            )


            bot_msg = chat_msg.then(self._bot, bot, bot)
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            bot_msg.then(self._play_audio, None, audio)



        demo.launch(share=share)


ui = UIController(n_results=20)
# ui.embedder.delete_collections("all")
# ui.embedder.visualize(collection_name="Mycollection", dimensions=["2d", "3d"])
ui.create_ui(share=True)
