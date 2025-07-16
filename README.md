# AI Tour Guide

### Introduction
This Master Thesis project is about creating an AI tour Guide for the archaeological museum of Aiani.
It is a chatbot that can answer question about the museum exhibits. The user can interact with the bot via text and via 
voice. Depending on the way the user submit his question the answer will come in text form and voice (if the question
was acoustic).

### Project architecture

The code of the project consists of 6 modules. The first 5 actually make the app:
- constantscls.py (Holds currently one constant)
- mydoc.py (The parser responsible for loading pdfs)
- embedder.py (responsible for creating the embeddings)
- generator.py (responsible for creating the answer to the user's question)
- uicontroller.py (responsible for the UI/UX)
- tester.py (responsible for the testing of the app)

### Quickstart

> **First step: Download the project and install dependencies**
```commandline
git clone https://netsim.cs.uowm.gr/gitlab/miranet/ai-tour-guide.git
pip install -r requirements.txt
```
>**Second step: Create a .env file.**  
> *The file should contain at least an OpenAi API key.
> If you want to use RAGAS GUI for viewing the test question - answers write the RAGAS API key also.*
```
OPENAI_API_KEY=sk-proj-HvwZhlyqL_MVE09LHgb6...
RAGAS_APP_TOKEN=apt.4275-96a6d42...
```
> **Third step: prepare the Vector database.**  
> *In the repository there is an existing vectore database containing the Aiani's pdfs. The database
> created with OpenAI's ``text-embedding-3-large`` and consists of 1500 dimensions.*
```python
from embedder import Embedder

embedder = Embedder()
# Delete previously  created collections
embedder.delete_collections("all")
# Load the documents first
embedder.load_docs(directory="aiani dedomena/to_embed/*", chunking_type=Embedder.ByChar)
# Add the documents in a new collection
embedder.add_data("Mycollection", embedding_model="text-embedding-3-large")
# Check out the new collection
embedder.visualize("Mycollection", dimensions=["2d", "3d"])
```
>**Fourth step: Start the application**  
> *Now that you have the database ready it's time to start the application.*
```python
from uicontroller import UIController

# Initialize UIController with 20 most relevant to the user's question text chunks. You can change that number  
ui = UIController(n_results=20)
# Create the UI. If you make share=True there will be a public link to share for someone to check out the app.
ui.create_ui(share=False)
```
### RAG pipeline evaluation.  
This project provides an evaluation method for the RAG pipeline.
It uses the automate metrics from [RAGAS](https://docs.ragas.io/en/stable/).
The code for the evaluator exists in tester.py module. 
The actual philosophy behind the tester is that it uses a LLM in the role of judge and evaluates the RAG pipeline in
a number of questions. 

### Evaluation Quickstart
>  **1. Create the RAGAS API key**  
> *Go to [Ragas Dashboard](https://app.ragas.io/) and create an account and an API key.
>Add the key in the .env file*    

>**2. Divide the pdfs into separate directories based on their topic**  
> *Each pdf should be in a folder based on its topic. For example all pdfs about a specific exhibition should be in the same directorie.
The reason for this is that the LLM that will produce the test data will first read all the pdfs on a folder.*

>**3. Generate the test data**  
> *There are 5 possible question categories that can be generated*
> - S1: Numerical questions - Their answer is a number.
> - S2: Dates questions - Their answer is a date.
> - S3: Multiple choice questions.
> - S4: Multiple choice questions that need some logic for the answer.
> - S5: Questions that shouldn't be answered.

```python
from tester import Tester

t = Tester()
# Generate the test questions for category s4.
# Save the question - answer pairs in a .pkl file.
t.generate_test_data_multiple_docs(
    name=["s4_o3-mini_test_not_acc.pkl"],
    model='o3-mini',
    question_categories=["s4"],
    num=10,
    paths=["aiani dedomena/aithouses/*", "aiani dedomena/ekpaid_prog/*", "aiani dedomena/ekthemata/*", "aiani dedomena/tafoi/*"]
)
```
> **4. Upload the test data in order to accept them**  
> *After generating the data you should upload them in the ragas platform in order to take a look
at them and accept them (press the tick button in every question) and after that download the JSON file.*

```python
# Upload the test data
t.upload_dataset("s4_o3-mini_test_not_acc.pkl")
# Accept the downloaded json file.
# It will automatically create the 'testset_accepted.pkl' file.
t.accept_test_set("s4_o3-mini_test_acc.json")
```
> **5. Finally run the test to the RAG pipeline**  
>*The final step is to run the test and check the results. In this step the class method
test() is called that does two things. First, it asks the RAG pipeline to answer the test question and
second it evaluates the generated answer based on the correct answers.
If you have previously run the test and the RAG pipeline has already answered the test questions
you can just run the evaluation process for your chosen metric.*

```python
from embedder import Embedder
from generator import Generator

# The LLM evaluator is by default the GPT-4o
Tester.test(
    Generator(Embedder(),
              "Mycollection",
              n_results=20,
              model='o3-mini'), # Create you custom RAG pipeline
    upload=False, # You can upload the results  
    metrics=['answer_correctness'], # The metric for the evaluation (there are plenty of them to choose)
    save_as_evaluated="s4_o3-mini-evaluated_by_gpt-4o.pkl",
    test_set="s4_o3-mini_test_acc.pkl" 
    # If you already run the test and you just want to revaluate the results just use the 'load_evaluation_set' parameter
)
# You can upload the result immediately after the test or after.
t.upload_dataset("s4_o3-mini-evaluated_by_gpt-4o.pkl")
# You can also view it in excel
t.to_csv("s4_o3-mini-evaluated_by_gpt-4o.pkl", "s4_o3-mini-evaluated_by_gpt-4o.csv")
```
