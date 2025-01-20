# Langchain RAG Pipeline

![RAG Pipeline](<RAG Pipeline.jpg>)
## Configure environment

Update variables in .env file
```python
OPENAI_API_KEY = "<your openai api key>
```

## Install dependencies

Run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

Install markdown depenendies with:

```python
pip install "unstructured[md]"
```

## Create database

Load, split, embed, and save data to ChromaDB

```python
python rag-data.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "What was my mood last Monday?"
```

> This is an iteration of the pixegami [Langchain RAG Tutorial](https://github.com/pixegami/langchain-rag-tutorial/tree/main) 

