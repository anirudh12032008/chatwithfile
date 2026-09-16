# Chat with your notes
upload a pdf, txt or md file and ask questions about it, the app will find the part of file related to your question and has an LLM answer from them also shows which part it used via RAG

it runs on render free plan so downtime or delay is unavoidable

## screenshots

<img width="1704" height="958" alt="Screenshot 2026-09-16 at 8 54 31 AM" src="https://github.com/user-attachments/assets/6d7237e7-0205-45e4-a97f-c4de98ebb0e9" />

## what it does
- upload a txt, pdf or md from the side bar
- ask a question
- it would stream the responses and each ans shows the file and a preview of the text it contained
- you can also index in a folder via CLI mode

## how it works ( RAG )
LOAD -> CHUNK (unc ref lolll) -> STORE -> RETRIEVE -> GENERATE

- load - read file
- chunk - RecursiveCharacterTextSplitter
- embed - using fastembed model running locally
- store - chroma vector db
- retrieve - finds the chunks most similar to your questions
- generate - groq llama-3.1-8b-instant will answer using those chunks

## tech stack
backend - python, fastapi, uvicorn
RAG - langchain, chrome, fastembed, pypdf
LLM - groq
frontend - html css js
deployment - render 

## project structure
backend.py -> helps connect frontend and rag
rag.py -> the main heart
main.py -> cli
static/ -> frontend


## running locally 
```bash
git clone <!-- repo url -->
cd chatwithfile
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn backend:app --reload
```
Open http://localhost:8000

**With Docker:**
```bash
docker build -t chatwithfile .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here chatwithfile
```

**CLI:**
```bash
python main.py index --docs docs   # put files in ./docs first
python main.py chat
```


## how it was made with problems
 i started it off as a CLI so i got index and answers working in the terminal first then added fast api and a web ui on top
i added server sent events on the backend and read the system on the frontend so answers appear word by word instead of all at once
render kept failing embedding a whole document at once used too much memory on the free plan so i switched to batches


## limitations
- only one file at a time
- index lives in memory and resets with server
- no user account so everyone sees same file

## ai disclosure
code is written by me but claude was used for debugging the SSE streaming and the render deploy, a little help with ui was taken from ai but not slop and the app itself uses ai models from groq
