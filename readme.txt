HOW TO RUN THE PROJECT
REQUIREMENTS

Python 3.11 or higher

PostgreSQL 16

pgvector extension enabled

Groq API key

SETUP

Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

Install Python dependencies
pip install -r requirements.txt

DATABASE CONFIGURATION

Create the database
CREATE DATABASE rag_chatbot;

Enable pgvector extension
CREATE EXTENSION vector;

ENVIRONMENT VARIABLES

Create a file src/.env (based on src/.env.example) and set:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_chatbot
DB_USER=postgres
DB_PASSWORD=YOUR_PASSWORD

GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=llama-3.3-70b-versatile

DATA INGESTION

Run the ingestion script to load dialogues and generate embeddings:

python src/ingest.py

RUN THE APPLICATION

Launch the Streamlit application:

streamlit run src/app.py

Open your browser at:

http://localhost:8501

NOTES

The ingestion step must be executed before running the application

Do not commit the .env file to GitHub

For small datasets, no vector index is required