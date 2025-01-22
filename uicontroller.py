import gradio as gr  # Ensure the Gradio library is installed by running: pip install gradio
from generator import Generator
from embedder import Embedder

class UIController:

    def __init__(self):
        self.gen = Generator(Embedder(), "Mycollection", n_results=5)
        self.embedder = Embedder()

        self._prepare_embedder()

    # ===   PRIVATE METHODS   === #
    def _prepare_embedder(self):
        """
        Loads the documents and creates the collection.
        :return: None
        """

        if not self.embedder.collection_exists("Mycollection"):
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
        print(history)
        question = history[-1]["content"]
        bot_message = self.gen.generate_answer(question, model="gpt-4o-mini")
        history.append({"role": "assistant", "content": ""})
        for chunk in bot_message:
            history[-1]['content'] += chunk
            yield history

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

            bot = gr.Chatbot(type="messages")
            msg = gr.Textbox()
            # submit = gr.Button("Submit")

            # submit.click(fn=self.gen.generate_answer, inputs=[input], outputs=[output]).then

            msg.submit(self._user, [msg, bot], [msg, bot], queue=False).then(
                self._bot, bot, bot
            )

        demo.launch(share=share)


ui = UIController()
ui.create_ui(share=True)