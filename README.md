
# DocuMind

DocuMind is a chatbot that can answer questions about a PDF file. It can do this by using a large language model (LLM) to understand the user's query and then searching the PDF file for the relevant information. The application uses the concept of Retrieval-Augmented Generation (RAG) to generate responses in the context of a particular document. RAG applications augment their generation capabilities by retrieving relevant information from an external knowledge base. This allows RAG applications to produce more informative and comprehensive responses to a wider range of prompts and questions.

## Setup and usage
1. install Ollama and run mistral
https://ollama.com/

2. Clone this repository:
   
 ```
 git clone https://github.com/SonicWarrior1/pdfchat.git
 ```
3. Install all the depenedencies :
   
```
pip install -r requirements.txt
```
4. Open terminal and run the following command:
```
streamlit run app.py
```
