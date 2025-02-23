import json
import pickle
from xmlrpc.client import Boolean

from dotenv import load_dotenv
import os
import glob
from embedder import Embedder
from mydoc import MyDoc
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.testset import Testset, TestsetSample
from generator import Generator
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import(FactualCorrectness, Faithfulness, LLMContextRecall,
                          NonLLMStringSimilarity, BleuScore, AnswerCorrectness, ResponseRelevancy, SemanticSimilarity,
                          ContextEntityRecall, LLMContextPrecisionWithReference, RougeScore)
from ragas.run_config import RunConfig


class DataStructure(BaseModel):
    user_input: str
    reference_contexts: list[str]
    reference: str

    def to_ragas_sample(self):
        return TestsetSample(
            eval_sample=
                SingleTurnSample(
                    user_input=self.user_input,
                    reference_contexts=self.reference_contexts,
                    reference=self.reference,
            ),
            synthesizer_name="single_hop_specifc_query_synthesizer"
        )

class DataList(BaseModel):
    data: list[DataStructure]

class Tester:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")
        self._gpt_client = OpenAI()

        self.loader = None
        self.docs = None

    # ===  PRIVATE METHODS === #
    def _load_docs(self, directory):
        """
        Loads the pdfs in document objects in docs attribute
        :param directory: The path to the documents
        :return: None
        """
        print("Loading docs...")

        # Load docs from a pickle file if possible
        self.docs = self._load_pickle("docs.pkl")

        if not self.docs:
            doc_paths = glob.glob(directory)
            docs = []
            for path in tqdm(doc_paths, total=len(doc_paths)):
                doc = MyDoc(path)
                docs.append(doc.get_pages())

            if not docs:
                raise ValueError("No documents were loaded. Please check your directory path and document content.")

            self.docs = docs
            self._save_pickle("docs.pkl", self.docs)
        print(f"Loaded {len(self.docs)} document.")

    def _gpt_api_call(self, text: str, model="gpt-4o-mini") -> DataList:

        completion = self._gpt_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Είσαι ένας ξεναγός στο αρχαιολογικό μουσείο αιανής που βρίσκεται στην Κοζάνη "
                            "(μικρή πόλη της Ελλάδας). Σκοπός σου είναι να δημιουργήσεις ένα σέτ δεδομένων που "
                            "θα αξιολογήσουν την ικανότητα ενός άλλου ξεναγού να απαντήσει στις ερωτήσεις επισκεπτών. "
                            "Τα δεδομένα που θα δημιουργήσεις πρέπει να βασίζονται σε δοθέν κείμενο."
                 },
                {"role": "user",
                 "content": "επέστρεψε ακριβώς 3 δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input, reference_contexts "
                            "και reference) όπου κάποια θα μπορούν να απαντηθούν με απλή αναφορά στο κείμενο"
                            "ενώ άλλα θα χρειάζονται συνδιασμό διαφόρων σημείων του κειμένου για να απαντηθούν."
                            "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                            "user_input: ερώτηση πάνω στο κείμενο"
                            "reference_contexts: σημεία του κειμένου πάνω στα οποία βασίζεται η σωστή απάντηση"
                            "reference: Η σωστή απάντηση στην ερώτηση"
                            "Παραδείγμα:"
                            "user_input: Πώς είναι διαμορφομένο το «σπίτι με σκάλες»?"
                            "reference_contexts: ['Πρόκειται για μια οικία ορθογώνιας κάτοψης με μικρά και μεγάλα δωμάτια, "
                            "ενώ στο μέσο της υπάρχει διάδρομος με σκαλοπάτια.', 'Και το σπίτι αυτό λόγω της κλίσης "
                            "του εδάφους είναι χτισμένο σε διαφορετικά επίπεδα τα οποία επικοινωνούν μεταξύ τους με λίθινες σκάλες']"
                            "reference: Το «σπίτι με σκάλες» είναι μια οικεία με ορθογώνια κάτοψη, έχει μικρά και μεγάλα δωμάτια"
                            "ενώ στο μέσο του υπάρχει ένας διάδρομος με σκαλοπάτια. Επιπλέον, λόγω της κλίσης του εδάφους"
                            "είναι χτισμένο σε διαφορετικά επίπεδα τα οποία επικοινωνούν μεταξύ τους με λίθινες σκάλες."
                            "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: " + text
                 }
            ],
            response_format=DataList,
        )

        return completion.choices[0].message.parsed

    @staticmethod
    def _save_pickle(filename, data):
        """
        Saves the given data in a pickle file.
        :param filename: The name of the pickle file.
        :param data: The data to save.
        :return: None
        """
        if ".pkl" not in filename:
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def _load_pickle(filename, raise_exception=False):
        """
        Loads the pickle file with the given name.
        :param filename: The name of the pickle file.
        :param raise_exception: Weather or not to raise exception if filename not found default to False.
        :return: The data stored in the pickle file or None if the file does not exist or is not a pickle file.
        """
        if ".pkl" not in filename:
            filename += ".pkl"

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                return data
        except FileNotFoundError:
            if raise_exception:
                raise FileNotFoundError("You need to accept the data set first!!")
            return None

    def _query_generator(self, generator: Generator) -> tuple[list[str], list[list[str]]]:
        """
        prompts the given generator function with the test set questions and collects the answers and retrieved texts
        :param generator: The generator object that will answer the test questions.
        :return: a tuple of answers and contexts in lists
        """

        # Load test_set_accepted.pkl
        dataset = self._load_pickle("test_set_accepted.pkl")

        dataset = dataset.to_list()
        print(dataset)

        # Extract questions
        questions = []
        for data in dataset:
            questions.append(data["user_input"])

        # Query generator
        print("Answering the test questions")
        ans = []
        cont = []
        for question in tqdm(questions, total=len(questions)):
            answer, context = generator.generate_answer_structured(question, model="gpt-4o-mini")
            ans.append(answer)
            cont.append(context)

        return ans, cont


    def _create_eval_set(self, answers, contexts) -> EvaluationDataset:
        """
        Creates the evaluation set based on the 'test_set_accepted.pkl' answers and contexts.
        :param answers: The answers to the test set question.
        :param contexts: The contexts provided to the model to answer.
        :return: The evaluation set.
        """

        # Load test_set_accepted.pkl
        dataset = self._load_pickle("test_set_accepted.pkl")
        dataset = dataset.to_list()

        # Create the eval set
        for i in range(len(dataset)):
            dataset[i]["retrieved_contexts"] = contexts[i]
            dataset[i]["response"] = answers[i]

        # Convert back to Testset and EvaluationSet
        dataset = Testset.from_list(data=dataset).to_evaluation_dataset()

        print(f"Eval set: {dataset}")
        return dataset

    @staticmethod
    def _get_metrics(metrics: list[str]) -> list:
        """
        Parse through the metrics and gets the metrics objects
        :param metrics: List of the metrics
        :return: List of metrics as ragas objects
        """

        available_metrics = {
            "answer_correctness": AnswerCorrectness(),
            "response_relevancy": ResponseRelevancy(),
            "semantic_similarity": SemanticSimilarity(),
            "context_entity_recall": ContextEntityRecall(),
            "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
            "context_recall": LLMContextRecall(),
            "factual_correctness": FactualCorrectness(),
            "faithfulness": Faithfulness(),
            "non_llm_string_similarity": NonLLMStringSimilarity(),
            "blue_score": BleuScore(),
            "rouge_score": RougeScore(),

        }

        metrics_to_return = [available_metrics.get(metric, "nan") for metric in metrics]

        if not metrics or "nan" in metrics_to_return:
            raise Exception("Please provide metrics for the evaluation. Available metrics are: answer_correctness, "
                            "response_relevancy, semantic_similarity, context_entity_recall, "
                            "llm_context_precision_with_reference, context_recall, factual_correctness, faithfulness, "
                            "non_llm_string_similarity, blue_score, rouge_score")

        return metrics_to_return


    # === CALLABLE METHODS === #

    def generate_test_data(self, directory="aiani dedomena/*", model="gpt-4o-mini") -> Testset:
        """
        Generates Test data set in ragas ready format.
        If the dataset already exists, it will be loaded from a pickle file.
        :param directory: The path to the documents.
        :param model: The model to use for generating the test data. Default is "gpt-4o-mini".
        :return: Evaluation dataset of format Testset[TestsetSample[SingleTurnSample]
        """

        # Check if eval dataset already exists
        generated_data = self._load_pickle("eval_dataset.pkl")
        if generated_data:
            print("test dataset already exists. Loading...")
            return generated_data
        else:
            # If data set does not already exist generate them
            self._load_docs(directory)

            # Generate data for each document
            generated_data = []
            print("Generating data...")
            for doc in tqdm(self.docs, total=len(self.docs)):
                # extract text
                text = ""
                for page in doc:
                    text += page.page_content
                generated_data.extend(self._gpt_api_call(text, model=model).data)
            print(f"Generated data: {generated_data[:]}")
            print("Data generated.")

            # Load them in the appropriate format
            generated_data = Testset(samples=[ragas_sample.to_ragas_sample() for ragas_sample in generated_data])
            print("Dataset: ", generated_data)

            # Save data
            self._save_pickle("test_set_not_accepted.pkl", generated_data)
            print("Test dataset saved.")

        return generated_data

    def upload_dataset(self, dataset_name) -> None:
        """
        Loads 'dataset_name' dataset and uploads it to ragas.
        :param dataset_name: The name of the dataset to upload
        :return: None
        :raises FileNotFoundError: If the dataset does not exist.
        """

        dataset = self._load_pickle(dataset_name)
        if not dataset:
            raise FileNotFoundError("Eval dataset does not exist. Please generate it first.")

        dataset.upload()

    def accept_test_set(self, filename):
        """
        Loads the 'testset.json', the approved test set, into Testset format
        and creates the 'test_set_accepted.pkl' file.
        :param filename: The filename of the test set to load.
        :return: None
        """

        with open(filename, "r", encoding="utf-8") as f:
            annotated_testset = json.load(f)

        samples = []
        for sample in annotated_testset:
            if sample["approval_status"] == "approved":
                samples.append(TestsetSample(**sample))

        testset = Testset(samples=samples)

        self._save_pickle("test_set_accepted.pkl", testset)


    def to_csv(self, dataset: str, filename: str):
        """
        Saves in csv form the give dataset.
        It should be in pkl format and the supporting types are EvaluationResult and EvaluationDataset.
        :param dataset: The data set to transform in csv. It should be a .pkl
        :param filename: The name of the cdv file.
        :return: None
        """

        # Load pkl file
        ds = self._load_pickle(dataset)

        df = ds.to_pandas().to_csv(path_or_buf=filename, encoding="utf-16")
        print(df)




    @classmethod
    def test(cls, generator: Generator, metrics: list[str], save_as="evaluated_dataset.pkl", load_evaluation_set="", upload: Boolean=True):
        """
        Runs a test set on a generator and evaluates the results.
        The test set should be first accepted using the 'accept_test_set()' method.
        The results are saved on file named as 'filename'
        :param generator: A generator to answer the questions.
        :param metrics: List of metrics to evaluate the evaluation set. Available metrics are: answer_correctness,
                        response_relevancy, semantic_similarity, context_entity_recall,
                        llm_context_precision_with_reference, context_recall, factual_correctness, faithfulness,
                        non_llm_string_similarity, blue_score, rouge_score.
        :param save_as: The name of the file that results of evaluation will be saved,
        defaults to 'evaluated_dataset.pkl'.
        :param load_evaluation_set: The name of the evaluation set to load, if none is given, the evaluation set will be generated.
        :param upload: Weather or not to upload the evaluated data set to ragas.
        Default is True
        :return: None
        """

        # Initiate class
        t = cls()

        # Initiate evaluator llm.
        llm = ChatOpenAI(model="gpt-4o-mini")
        evaluator_llm = LangchainLLMWrapper(llm)

        # If specified load evaluation data set
        if load_evaluation_set:
            print("Loading evaluation data set that already exists...")
            eval_set = t._load_pickle(load_evaluation_set)
        else:
            # Answer test question from generator and create evaluation dataset.
            print("creating evaluation data set...")
            answers, contexts = t._query_generator(generator)
            eval_set = t._create_eval_set(answers, contexts)

            # Save eval set
            print("Saving evaluation data set that created...")
            t._save_pickle("evaluation_set.pkl", eval_set)

        # Get evaluation dataset results.
        result = evaluate(
            dataset=eval_set,
            metrics=t._get_metrics(metrics),
            llm=evaluator_llm,
            run_config=RunConfig(
                timeout=360,
                max_retries=20,
                max_wait=120,
                max_workers=4,
            )
        )

        # Save evaluation dataset
        t._save_pickle(save_as, result)

        print(result)

        # Upload evaluation results on ragas.
        if upload:
            result.upload()








# Tester.test(Generator(Embedder(), "Mycollection", n_results=20), load_evaluation_set="evaluation_set.pkl")
#
# t = Tester()
# t.to_csv("evaluated_dataset.pkl", "evaluated_dataset_csv.csv")
# t.accept_test_set("testset.json")

# t.generate_test_data(model="gpt-4o-mini")
# t.upload_dataset("evaluated_dataset.pkl")


# gen = Generator(Embedder(), "Mycollection", n_results=10)
# answers, contexts = t._query_generator(gen)
# t._create_eval_set(answers, contexts, 10)
# print(f"answers: {answers}\n\ncontexts: {contexts}")