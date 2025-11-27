# Clap Tests Web App

Clap Tests is a browser-based acoustic analysis tool for everyday rooms.

## Run locally (Windows)

```bash
git clone https://github.com/RowanSakoulas/clap-tests-webapp.git
cd clap-tests-webapp

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

uvicorn webapi.serve:app --reload

Then open this in your browser:
http://127.0.0.1:8000/
```

## Optional: enable AI summaries

By default the app runs without the AI room summary. To enable AI feedback:

1. Get an OpenAI API key.
2. In the repo root, create a file named openai_key.txt.
3. Put your API key in that file as a single line.
4. Restart uvicorn and reload the page. The AI summary panel will be enabled.