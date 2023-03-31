# InfiniteArt
## This bot creates art by automatically searching the web for prompt inspiration!
---
### *It is a work in progress, and is not yet complete.*
---
## How to use:
1. Navigate to the project folder and install the requirements with `pip install -r requirements.txt`
    * requirements:
        * requests
        * beautifulsoup4
        * pillow
        * openai
        * spacy
        * werkzeug
2. Download the spaCy NLP model with `python -m spacy download en_core_web_md`
3. Add your API keys to the `api_keys.py` file. The OpenAI key is mandatory, the google keys are optional.
4. Run `main.py` to start the bot. (you can add a number as an argument to specify how many images to generate, like so: `python main.py 10`)
---
### Future plans:
* Add editing capabilities (using dalle, pillow and/or something else)
* Make the agent more "agent-like" (currently it runs kind of like a script, but I want it to be able to make meaningful decisions on its own)
* More dynamic mood system