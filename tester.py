import pickle
from dotenv import load_dotenv
import os
import glob
from mydoc import MyDoc
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.testset import Testset, TestsetSample

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
    def _load_pickle(filename):
        """
        Loads the pickle file with the given name.
        :param filename: The name of the pickle file.
        :return: The data stored in the pickle file or None if the file does not exist or is not a pickle file.
        """
        if ".pkl" not in filename:
            filename += ".pkl"

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                return data
        except FileNotFoundError:
            return None

    # === CALLABLE METHODS === #

    def generate_test_data(self, directory="aiani dedomena/*", model="gpt-4o-mini") -> Testset:
        """
        Generates evaluation data set in ragas ready format.
        If the dataset already exists, it will be loaded from a pickle file.
        :param directory: The path to the documents.
        :param model: The model to use for generating the test data. Default is "gpt-4o-mini".
        :return: Evaluation dataset of format Testset[TestsetSample[SingleTurnSample]
        """

        # Check if eval dataset already exists
        generated_data = self._load_pickle("eval_dataset.pkl")
        if generated_data:
            print("Eval dataset already exists. Loading...")
            return generated_data
        else:
            # If data set does not already exist generate them
            self._load_docs(directory)

            # Generate data for each document
            generated_data = []
            print("Generating data...")
            for doc in tqdm(self.docs[10:20], total=len(self.docs[10:13])):
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
            self._save_pickle("eval_dataset.pkl", generated_data)
            print("Eval dataset saved.")

        return generated_data

    def visualize_eval_dataset(self) -> None:
        """
        Loads evaluation dataset from 'eval_dataset.pkl' and
        demonstrates them in the ragas dashboard.
        :return: None
        :raises FileNotFoundError: If the dataset does not exist.
        """

        dataset = self._load_pickle("eval_dataset.pkl")
        if not dataset:
            raise FileNotFoundError("Eval dataset does not exist. Please generate it first.")

        # dataset = dataset.to_pandas()
        dataset.upload()



    @classmethod
    def test(cls, generator_function):
        pass


t = Tester()
t.generate_test_data(model="gpt-4o-mini")
t.visualize_eval_dataset()