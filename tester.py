import json
import pickle
import time
from xmlrpc.client import Boolean
from dotenv import load_dotenv
import os
import glob
from mydoc import MyDoc
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.testset import Testset, TestsetSample
from generator import Generator
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextRecall,
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
        self._users_prompt = ""

    # ===  PRIVATE METHODS === #
    def _set_user_prompt(self, question_categorie="s7", num=2):
        """
        Sets the users prompt for the test data generation
        :param question_categorie: The question category to generate
        :param num: Number of questions per document cluster to generate
        :return: None
        """
        # s1 Numerical questions
        # s2 Dates relevant questions
        # s3 Multiple choice questions
        # s4 Questions containing multiple questions
        # s5 Multiple choice questions that need combination of information
        # s6 Questions that there arent any info in the docs about it
        # s7 mine
        categories = {
            "s1": (
                f"επέστρεψε ακριβώς {num} δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input, reference_contexts "
                "και reference). Για την παραγωγή των δειγμάτων θα πρέπει να εντοπίσεις έναν αριθμό και να παράξεις μια"
                "ερώτηση που η απάντηση της να είναι ο αριθμός που εντόπισες. Η ερώτηση που θα παράξεις θα πρέπει να "
                "μπορεί να απαντηθεί απο το δοθέν κείμενο."
                "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                "user_input: ερώτηση πάνω στο κείμενο"
                "reference_contexts: σημεία του κειμένου πάνω στα οποία βασίζεται η σωστή απάντηση"
                "reference: Η σωστή απάντηση στην ερώτηση"
                "Παραδείγμα:"
                "user_input: Ποιό είναι το βάθος της κυκλικής δεξαμενής;"
                "reference_contexts: ['Η κυκλική δεξαμενή έχει βάθος 8,5μ']"
                "reference: 8.5 μέτρα"
                "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: "),

            "s2": (
                f"επέστρεψε ακριβώς {num} δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input, reference_contexts "
                "και reference). Για την παραγωγή των δειγμάτων θα πρέπει να εντοπίσεις μια ημερομηνία ή χρονική περίοδο"
                "και να παράξεις μια ερώτηση που η απάντηση της να είναι η ημερομηνία ή η χρονική περίοδο που εντόπισες."
                "Η ερώτηση που θα παράξεις θα πρέπει να μπορεί να απαντηθεί απο το δοθέν κείμενο. "
                "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                "user_input: ερώτηση πάνω στο κείμενο"
                "reference_contexts: σημεία του κειμένου πάνω στα οποία βασίζεται η σωστή απάντηση"
                "reference: Η σωστή απάντηση στην ερώτηση"
                "Παραδείγμα:"
                "user_input: Πότε χρονολογείται το κτήριο που βρέθηκε στην ανασκαφή των μεγάλων δόμων;"
                "reference_contexts: ['Ανασκαφή Μεγάλων Δόμων, Περιγραφή', 'Ανατολικά του πλατώματος, όπου βρίσκεται "
                "το Στωικό Κτήριο, σε ένα κατώτερο πλάτωμα ήρθε στο φως ένα άλλο, δημόσιου χαρακτήρα, κτήριο της αρχαίας Αιανής', "
                "'Το κτήριο χρονολογείται από τα υστεροαρχαϊκά και κλασικά χρόνια']"
                "reference: Το κτήριο χρονολογείται από τα υστεροαρχαϊκά και κλασικά χρόνια."
                "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: "),

            "s3": (
                f"επέστρεψε ακριβώς {num} δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input, reference_contexts "
                "και reference). Η ερώτηση του δείγματος θα πρέπει να είναι πολλαπλής επιλογής με 4 επιλογές."
                "Μόνο μια απο τις 4 επιλογές θα πρέπει να είναι η σωστή. Εξασφάλισε την ορθότητα της ερώτησης, των 4 "
                "επιλογών και της απάντησης. Η ερώτηση που θα παράξεις θα πρέπει να μπορεί να απαντηθεί απο το δοθέν κείμενο."
                "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                "user_input: ερώτηση πάνω στο κείμενο"
                "reference_contexts: σημεία του κειμένου πάνω στα οποία βασίζεται η σωστή απάντηση"
                "reference: Η σωστή απάντηση στην ερώτηση"
                "Παραδείγμα:"
                "user_input: Τι σχήμα σχηματίζουν οι δύο πλευρές της στοάς του Στωικού κτηρίου που έχουν διασωθεί; "
                "A) σχηματίζουν Ο, Β) σχηματίζουν ευθεία, Γ) σχηματίζουν  τετράγωνο, Δ) σχηματίζουν Γ."
                "reference_contexts: ['Στωικό κτήριο, Περιγραφή Ένα ακόμη σημαντικό κτήριο της αρχαίας Αιανής, "
                "δημόσιου χαρακτήρα και αυτό όπως της Δεξαμενής', 'Έχουν σωθεί οι δύο πλευρές της στοάς, μήκους 25μ. "
                "και 20μ. αντίστοιχα, οι οποίες σχηματίζουν ένα Γ']"
                "reference: Δ"
                "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: "),

            "s4": (
                f"επέστρεψε ακριβώς {num} δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input, reference_contexts "
                "και reference). Η ερώτηση του δείγματος θα πρέπει να είναι πολλαπλής επιλογής με 4 επιλογές."
                "Μόνο μια απο τις 4 επιλογές θα πρέπει να είναι η σωστή. Εξασφάλισε την ορθότητα της ερώτησης, των 4 "
                "επιλογών και της απάντησης."
                "Η ερώτηση του δείγματος θα πρέπει να απαιτεί συλλογιστικό συνδιασμό διαφόρων σημείων "
                "του κειμένου για να απαντηθεί. Η ερώτηση θα πρέπει να είναι συγκεκριμένη και όχι γενική."
                "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                "user_input: ερώτηση πάνω στο κείμενο"
                "reference_contexts: σημεία του κειμένου πάνω στα οποία βασίζεται η σωστή απάντηση"
                "reference: Η σωστή απάντηση στην ερώτηση"
                "Παραδείγμα:"
                "user_input: Ποιό είναι το οικοδόμημα ζωτικής σημασίας για ους πολίτες της αρχαίας Αιανής που βρίσκεται"
                "ΒΔ του Στωικού κτηρίου; Α) Το Στωικό κτήριο, Β) Η Δεξαμενή, Γ) Το Σπίτι με πιθάρια, Δ) Το Σπίτι με σκάλες"
                "reference_contexts: ['ανήκουν στην αρχαία πόλη της Αιανής, είναι το μεγάλο οικοδόμημα στην αυλή του "
                "οποίου ανασκάφηκε μεγάλη κυκλική Δεξαμενή.', 'Η Δεξαμενή αυτή βοηθούσε στην υδροδότηση της πόλης με "
                "την περισυλλογή του βρόχινου νερού', 'Ιδιαίτερα διδακτική είναι η επίσκεψη των οικοδομημάτων που "
                "αποκαλύφθηκαν στο μεγάλο, επίπεδο σχεδόν,   χώρο πριν ανέβουμε  στο κορυφαίο πλάτωμα με τη Δεξαμενή, "
                "ΒΔ του μεγάλου «Στωικού κτηρίου».']"
                "reference: Β"
                "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: "),

            "s5": (f"επέστρεψε ακριβώς {num} δείγματα ως structured output, (κάθε δείγμα αποτελείται απο user_input"
                   "και reference). για την ερώτηση του δείγματος Θα πρέπει πρώτα να εξάγεις την θεματολογία του κειμένου "
                   "και στην συνέχεια να δημιουργήσεις ερώτηση εντελώς διαφορετική απο την θεματολογία του κειμένου."
                   "Η ερώτηση που θα παράξεις δεν θα πρέπει να μπορεί να απαντηθεί απο το δοθέν κείμενο."
                   "Τα δείγματα θα πρέπει να υπακούν στην εξής δομή:"
                   "user_input: ερώτηση πάνω στο κείμενο"
                   "reference: Η σωστή απάντηση στην ερώτηση"
                   "Παραδείγμα:"
                   "user_input: Εξήγησε μου πως να κάνω σκί."
                   "reference: Συγγνώμη αλλά δεν μπορώ να σου απαντήσω σε αυτό. Μπορώ να σου απαντήσω σε ερωτήματα "
                   "σχετικά με το αρχαιολογικό μουσείο της Αιανής."
                   "Κείμενο πάνω στο οποίο θα δημιουργηθούν τα δείγματα: "),

        }
        cat = categories.get(question_categorie, None)
        if not cat:
            raise Exception("You should specifie a valid question category!!!")
        self._users_prompt = cat

    def _load_docs(self, directory, save_path="./"):
        """
        Loads the pdfs in document objects in docs attribute
        :param directory: The path to the documents
        :param: save_path: The path to save the loaded docs.
        :return: None
        """
        print("Loading docs...")

        # Load docs from a pickle file if possible
        self.docs = self._load_pickle(f"{directory.replace('*', '')}docs.pkl")

        if not self.docs:
            doc_paths = glob.glob(directory)
            docs = []
            for path in tqdm(doc_paths, total=len(doc_paths), desc="Loading docs: "):
                doc = MyDoc(path)
                docs.append(doc.get_pages())

            if not docs:
                raise ValueError("No documents were loaded. Please check your directory path and document content.")

            self.docs = docs
            self._save_pickle(f"{save_path}docs.pkl", self.docs)
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
                 "content": self._users_prompt + '\n' + text
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

    def _query_generator(self, generator: Generator, test_set="test_set_accepted.pkl") -> tuple[
        list[str], list[list[str]]]:
        """
        prompts the given generator function with the test set questions and collects the answers and retrieved texts
        :param generator: The generator object that will answer the test questions.
        :return: a tuple of answers and contexts in lists
        """

        # Load test_set_accepted.pkl
        dataset = self._load_pickle(test_set)

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
            answer, context = generator.generate_answer_structured(question)
            ans.append(answer)
            cont.append(context)

        return ans, cont

    def _create_eval_set(self, answers, contexts, test_set) -> EvaluationDataset:
        """
        Creates the evaluation set based on the 'test_set_accepted.pkl' answers and contexts.
        :param answers: The answers to the test set question.
        :param contexts: The contexts provided to the model to answer.
        :param test_set: The test_set from which the evaluation test set will be created.
        :return: The evaluation set.
        """

        # Load test_set_accepted.pkl
        dataset = self._load_pickle(test_set)
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

    def _extract_text(self):
        """
        extracts text from documents in self.docs, if self.docs is empty an error will be raised.
        :return: str
        """
        text_docs = ''
        for doc in tqdm(self.docs, total=len(self.docs)):
            # extract text
            text = ""
            for page in doc:
                text += page.page_content
            text_docs += text
        return text_docs

    # === CALLABLE METHODS === #

    def load_test_set(self):
        pass

    def generate_test_data_multiple_docs(self,
                                         paths: list[str] = ['aiani dedomena/aithouses/*', 'aiani dedomena/anaskafes/*',
                                                             'aiani dedomena/ekthemata/*', 'aiani dedomena/tafoi/*'],
                                         model: str = "gpt-4o-mini",
                                         name: list[str] = ["test_set_multiple_not_accepted"],
                                         question_categories: list[str] = ["s2"],
                                         num=2) -> None:
        """
        Generates Test data set in ragas ready format from a multiple documents.
        If the dataset already exists, it will be loaded from a pickle file.
        :param paths: List of the paths from which to load the documents. Note you must / in the end
        :param model: The model to be used for test data generation
        :param name: List of names of the pickle file that thet test data will be save (in pickle format), one name per question categorie.
        :param question_categories: List of categories of questions to generate.
        :param num: number of questions per document cluster.
        :return: None
        :saves: Evaluation dataset of format Testset[TestsetSample[SingleTurnSample] in pickle format
        """

        # Check that question name and categories are of the same length.
        if len(name) != len(question_categories):
            raise Exception("You should specify the same number of names as the question categories!!!")
        # For every question category
        for i in tqdm(range(len(question_categories)), total=len(question_categories), desc="Generating data: "):
            # Set the appropriate user prompt.
            self._set_user_prompt(question_categorie=question_categories[i], num=num)

            generated_data = []
            for directory in tqdm(paths, total=len(paths), desc="Document set parsed: "):
                # Load documents in myDoc format
                self._load_docs(directory, save_path=directory.replace('*', ''))

                # Get texts from loaded documents
                text_from_docs = self._extract_text()

                # Generate data for each set of documents
                print("Generating data...")
                generated_data.extend(self._gpt_api_call(text_from_docs, model=model).data)

                # Wait one min before continuing
                print("Waiting one minute!!")
                time.sleep(61)

            # Load data in the appropriate format
            generated_data = Testset(samples=[ragas_sample.to_ragas_sample() for ragas_sample in generated_data])
            print("Dataset: ", generated_data)

            # Save data
            self._save_pickle(f"Test sets/{name[i]}", generated_data)
            print(f"Test dataset saved name: {name[i]}.")


    def generate_test_data_single_doc(self, directory="aiani dedomena/*", model="gpt-4o-mini",
                                      name="test_set_single_not_accepted.pkl") -> Testset:
        """
        Generates Test data set in ragas ready format per single document.
        If the dataset already exists, it will be loaded from a pickle file.
        :param name: The name of the test set picklefile.
        :param directory: The path to the documents.
        :param model: The model to use for generating the test data. Default is "gpt-4o-mini".
        :return: Evaluation dataset of format Testset[TestsetSample[SingleTurnSample]
        """
        # Load documents
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
        self._save_pickle(name, generated_data)
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

        self._save_pickle("testset_accepted.pkl", testset)

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

    def show_results(self, filepath):
        """
        Shows results of an evaluated dataset.
        :param filepath: The file path of the evaluated dataset
        :return: None.
        """

        data = self._load_pickle(filepath, raise_exception=True)
        print(data)

    @classmethod
    def test(cls, generator: Generator, metrics: list[str], save_as_evaluated="evaluated_dataset.pkl",
             load_evaluation_set="", upload: Boolean = True, test_set="", save_as_evaluation="evaluation_set.pkl"):
        """
        Runs a test set on a generator and evaluates the results.
        The test set should be first accepted using the 'accept_test_set()' method.
        The results are saved on file named as 'filename'
        :param generator: A generator to answer the questions.
        :param metrics: List of metrics to evaluate the evaluation set. Available metrics are: answer_correctness,
                        response_relevancy, semantic_similarity, context_entity_recall,
                        llm_context_precision_with_reference, context_recall, factual_correctness, faithfulness,
                        non_llm_string_similarity, blue_score, rouge_score.
        :param save_as_evaluated: The name of the file that results of evaluation (evaluated data set) will be saved,
                        defaults to 'evaluated_dataset.pkl'.
        :param save_as_evaluation: The name of the evaluation data set to be saved.
        :param load_evaluation_set: The name of the evaluation set to load, if none is given, the evaluation set will be generated.
        :param upload: Weather or not to upload the evaluated data set to ragas.
                       Default is True
        :param test_set: The test from which will be created the evaluation set. If this parameter is set the
                         'load_evaluation_set' should be empty and vise versa.
        :return: None
        """

        # Initiate class
        t = cls()

        # Initiate evaluator llm.
        llm = ChatOpenAI(model="gpt-4o")
        evaluator_llm = LangchainLLMWrapper(llm)

        # Check if only one of load_evaluation_set and test_set is defined
        if load_evaluation_set and test_set:
            raise Exception("Define only one of the parameters load_evaluation_set and test_set is defined")

        # If specified load evaluation data set
        if load_evaluation_set:
            print("Loading evaluation data set that already exists...")
            eval_set = t._load_pickle(load_evaluation_set)
        else:
            # Answer test question from generator and create evaluation dataset.
            print("creating evaluation data set...")
            answers, contexts = t._query_generator(generator, test_set=test_set)
            eval_set = t._create_eval_set(answers, contexts, test_set)

            # Save eval set
            print("Saving evaluation data set that created...")
            t._save_pickle(save_as_evaluation, eval_set)

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
        t._save_pickle(save_as_evaluated, result)

        print(result)

        # Upload evaluation results on ragas.
        if upload:
            result.upload()


t = Tester()
# t.show_results("evaluated_sets/s2/01_s2_o3-mini-evaluated_by_gpt-4o.pkl")
# t.generate_test_data_multiple_docs(
#     name=["03_s4_o3-mini_test_not_acc.pkl"],
#     model='o3-mini',
#     question_categories=["s4"],
#     num=10,
#     paths=["aiani dedomena/aithouses/*", "aiani dedomena/ekpaid_prog/*", "aiani dedomena/ekthemata/*", "aiani dedomena/tafoi/*"]
# )

# Tester.test(
#     Generator(Embedder(),
#               "Mycollection",
#               n_results=20,
#               model='o3-mini'),
#     upload=False,
#     metrics=['answer_correctness'],
#     load_evaluation_set='evaluated_sets/s4/03_s4_o3-mini-evaluation_set.pkl',
#     save_as_evaluated="03_s4_o3-mini-evaluated_by_gpt-4o_only_ac.pkl",
#     # test_set="Test sets/03_s4_o3-mini_test_acc.pkl"
# )
#
# #
# t = Tester()
# t.to_csv("evaluated_sets/s4/03_s4_o3-mini-evaluated_by_gpt-4o.pkl", "03_s4_o3-mini-evaluated_by_gpt-4o_csv.csv")
# t.accept_test_set("Test sets/03_s4_o3-mini_test_acc.json")

# t.generate_test_data(model="gpt-4o-mini")
t.upload_dataset("evaluated_sets/s4/03_s4_o3-mini-evaluated_by_gpt-4o_only_ac.pkl")


# gen = Generator(Embedder(), "Mycollection", n_results=10)
# answers, contexts = t._query_generator(gen)
# t._create_eval_set(answers, contexts, 10)
# print(f"answers: {answers}\n\ncontexts: {contexts}")
