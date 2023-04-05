# Import libraries and modules
import requests
from bs4 import BeautifulSoup
import random
from PIL import Image, ImageDraw, ImageFilter
import openai
import spacy
from PIL import Image
from io import BytesIO
from datetime import datetime
import math
from werkzeug.utils import secure_filename
from api_keys import OPENAI_KEY, GOOGLE_KEY, GOOGLE_CX
import sys


# Load the English language model (for comparing sentences by "artisticness")
print("-"*50)
print("loading my english brain...")
nlp = spacy.load("en_core_web_md")
print("done loading my english brain!")


class ArtCreatorAgent:
    def __init__(self):
        self.mood = None
        self.base_image = None
        self.edited_image = None
        self.art_topic = None
        self.artistic_opinions = None
        self.source = None
        self.use_google_api = False
        # This will be used as a reference sentence to compare sentences by "artisticness"
        self.artistic_ideal = None
        # Check and set API keys
        self.check_api_keys()

    def check_api_keys(self):
        if GOOGLE_KEY != "YOUR_GOOGLE_KEY" and GOOGLE_CX != "YOUR_GOOGLE_CX":
            print("yayy i have google api keys :D")
            self.use_google_api = True
            self.google_key = GOOGLE_KEY
            self.google_cx = GOOGLE_CX
        else:
            print("oh shoot, i dont have google api keys :(")
            print("i guess i'll just have to scrape the web instead :(")
            self.use_google_api = False

        if OPENAI_KEY == "YOUR_OPENAI_KEY":
            print("nooo!! i dont have an openai key :( i cant generate text... i am useless :(")
            raise Exception("No OpenAI key provided")

        else:
            openai.api_key = OPENAI_KEY

    def choose_mood(self):
        """
        Sets agent's mood randomly
        """
        # Create a list of possible emotions or moods
        emotions = [
            "happy",
            "sad",
            "angry",
            "excited",
            "calm",
            "anxious",
            "confused",
            "surprised",
        ]

        # Choose a random emotion from the list
        random_emotion = random.choice(emotions)

        # Create a prompt to ask the model to generate a novel mood
        prompt = f"Generate a novel aesthetic related to the emotion '{random_emotion}'."
        self.artistic_ideal = prompt

        # Make the API call
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.9,
        )

        # Extract the generated mood from the response
        self.mood = response.choices[0].text.strip()

        print("im rlly feeling the aesthetic of", self.mood, "today!!!!!")
        print("-"*50)

    def scrape_web(self):
        """Scrape the web for an artistic topic

        If you have a Google Custom Search API key, the bot will use that to get search results.
        If you don't have a Google Custom Search API key, the bot will scrape the web normally.
        """
        print("googling", self.mood, "for epic inspiration...\n")

        if self.use_google_api:
            search_results = self.scrape_web_with_api()
        else:
            search_results = self.scrape_web_without_api()

        threshhold = 0.8
        for result in search_results:
            url = result.get("link")
            print("ok imma clickin on", url, "\n")
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            art = self.most_artistic_sentence(soup)
            if art["score"] > threshhold:
                print("\n\n Okey so hear me out, i read this on the web and thought it was sooo cooool:\n\n", art["text"], "\n")
                self.art_topic = art["text"]
                self.source = url
                break
            else:
                threshhold -= 0.1

    def scrape_web_without_api(self):
        url = f"https://www.google.com/search?q={self.mood}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

        # Make the web call
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("yayy google let me in!")
        else:
            print("oh no google blocked me :(")
        soup = BeautifulSoup(response.text, "html.parser")
        search_results = soup.select(".g")

        return [{"link": result.select_one("a")["href"]} for result in search_results]

    def scrape_web_with_api(self):
        """Scrape the web using Google's Custom Search API to find a topic

        This is an alternative to the scrape_web() method above.

        """
        # Make the API call
        search_url = "https://www.googleapis.com/customsearch/v1"
        search_params = {
            "key": self.google_key,
            "cx": self.google_cx,
            "q": self.mood
        }
        search_response = requests.get(search_url, params=search_params).json()
        search_results = search_response.get("items", [])

        return search_results

    def generate_base_image(self):
            print("-"*50)
            print("Okay imma make the prompt better for DALLE!!!")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": "You are a generative image AI prompt curation bot. When a user enters a prompt, re-write the prompt so that it will have a striking image as a result. Use all the tricks in the book. Use your creativity! The end result should be a coherent prompt that will generate a striking image of something interesting."},
                        {"role": "user", "content": f"{self.art_topic}"},

                    ]
                )
            prompt = response["choices"][0]["message"]["content"]

            self.base_image = self.generate_image(prompt)

    def get_artistic_opinions(self):
        # Use GPT-4 to generate artistic opinions about the current image
        opinions = generate_artistic_opinions(self.base_image)
        self.artistic_opinions = opinions

    def edit_image(self):
        # Edit the image using Pillow and DALL-E based on the artistic opinions
        self.edited_image = apply_image_edits(self.base_image, self.artistic_opinions)

    def is_happy_with_result(self):
        # Determine if the agent is happy with the result
        return True  # or use some evaluation method

    def create_art(self):
        self.choose_mood()
        self.scrape_web()
        self.generate_base_image()
        print("-"*50)
        print("source:", self.source)
        print()
        print("URL:", self.base_image)
        print("-"*50)
        print("downloading image...")
        response = requests.get(self.base_image)
        img = Image.open(BytesIO(response.content))
        img.save("generated/"+secure_filename(f"{self.mood.replace('.', '')}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png"))
        print("image downloaded! showing...")
        img.show()

        # # Main function to create art
        # self.choose_mood()
        # self.scrape_web()
        # self.generate_base_image()

        # while not self.is_happy_with_result():
        #     self.get_artistic_opinions()
        #     self.edit_image()

        # return self.edited_image

    def most_artistic_sentence(self, soup):
        # Extract all text from the soup object
        text = soup.get_text()

        # Split the text into sentences
        sentences = list(nlp(text).sents)

        # This sentence will be compared to all other sentences
        reference_tokens = nlp(self.artistic_ideal)

        # Find the most artistic sentence by comparing their artistic scores
        mx = 0 # max similarity
        most_artistic = None
        for sentence in sentences:
            if type(sentence.text) == None:
                break

            # Calculate the similarity between the sentences using word embeddings
            similarity = nlp(sentence.text).similarity(reference_tokens)

            # Catch any None values (happens if no similarity is found..?)
            if most_artistic == None:
                most_artistic = sentence

            if similarity > mx:
                # print(f"Ooo this sentence is more artsy...: {sentence.text}\n")
                mx = similarity
                most_artistic = sentence
                star = " ⭐" + str(similarity)
            else:
                star = ""
            preview = sentence.text[:15]
            print(f"getting inspiredd: {preview}"+ ("."*math.floor(similarity*10))+star)

        return {"text":most_artistic.text.replace('\n', ''), "score":mx}

    def generate_image(self, prompt):
        # Use DALL-E API to generate an image given a prompt
        tags = f", {self.mood}"
        prompt=prompt+tags
        prompt = prompt.replace("\n", "").replace("\r", "").replace("\t", "")
        print(f"\nOkye dokye im gonna make this: \n\n{prompt}\n")

        response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
        )
        return response['data'][0]['url']

# Helper functions

def search_image_online(query):
    # Search the web for an image using the query
    pass

def generate_artistic_opinions(image):
    # Use GPT-4 API to generate artistic opinions about the image
    pass

def apply_image_edits(image, opinions):
    # Edit the image using Pillow based on artistic opinions
    pass


# Main function
def main(artworks=1):
    # Get the number of artworks to create from the command line
    if len(sys.argv) > 1:
        artworks = int(sys.argv[1])

    # Create and use the ArtCreatorAgent
    agent = ArtCreatorAgent()
    for i in range(artworks):
        print("-"*50)
        print(f"creating art {i+1}/{artworks}")
        try:
            agent.create_art()
        except Exception as e:
            print("error:", e)
            print("skipping this one...")
            continue

if __name__ == "__main__":
    main(1)
