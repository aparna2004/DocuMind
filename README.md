
# DocuMind

DocuMind is a chatbot that can answer questions about a PDF file. It can do this by using a large language model (LLM) to understand the user's query and then searching the PDF file for the relevant information. The application uses the concept of Retrieval-Augmented Generation (RAG) to generate responses in the context of a particular document. RAG applications augment their generation capabilities by retrieving relevant information from an external knowledge base. This allows RAG applications to produce more informative and comprehensive responses to a wider range of prompts and questions.

## Setup and usage
1. install Ollama and run mistral
https://ollama.com/

2. Clone this repository:
   
 ```
 git clone https://github.com/aparna2004/DocuMind
 ```
3. Install all the depenedencies :
   
```
pip install -r requirements.txt
```
4. Open terminal and run the following command:
```
streamlit run app.py
```

## Sample Run
![image](https://github.com/user-attachments/assets/f84928ea-ac30-4f28-9d0f-0699420806ca)
![image](https://github.com/user-attachments/assets/326d031f-3d84-4893-9175-9b159774395f)


