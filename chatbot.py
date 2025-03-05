

import numpy as np
import random
import string
import warnings
from tkinter import *
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

warnings.filterwarnings('ignore')


FAQ = {
    "greetings": [
        "hello",
        "hi there",
        "how are you",
        "what's up"
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you later"
    ],
    "product_info": [
        ("What is your return policy?", 
         "We offer a 30-day return policy for unused products."),
        ("How long does shipping take?", 
         "Standard shipping takes 3-5 business days."),
        ("Do you offer international shipping?", 
         "Yes, we ship to over 50 countries worldwide."),
        ("What payment methods do you accept?", 
         "We accept all major credit cards and PayPal."),
        ("How can I track my order?", 
         "You can track your order using the tracking number sent to your email.")
    ]
}

ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("FAQ Chatbot")
        master.geometry("600x500")
        master.configure(bg="#f0f0f0")

        self.vectorizer = None
        self.tfidf_matrix = None
        self.response_list = []
        
        self.setup_components()
        self.initialize_bot()
        self.bind_events()

    def setup_components(self):
        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(
            self.master, 
            wrap=WORD, 
            width=60, 
            height=20,
            font=("Arial", 10)
        )
        self.chat_history.tag_config("user", foreground="blue")
        self.chat_history.tag_config("bot", foreground="green")
        self.chat_history.pack(pady=10, padx=10, fill=BOTH, expand=True)
        self.chat_history.configure(state='disabled')

        # Input frame
        input_frame = Frame(self.master, bg="#f0f0f0")
        input_frame.pack(pady=10, fill=X)

        # User input field
        self.user_input = Entry(
            input_frame, 
            width=50, 
            font=("Arial", 12)
        )
        self.user_input.pack(side=LEFT, padx=5, ipady=3)

        # Send button
        self.send_btn = Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12)
        )
        self.send_btn.pack(side=LEFT, padx=5)

    def bind_events(self):
        self.master.bind('<Return>', lambda event: self.send_message())

    def initialize_bot(self):
        questions, responses = self.build_qa_pairs()
        processed_questions = [self.preprocess_text(q) for q in questions]
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
        self.response_list = responses
        
        self.display_bot_message("Hello! I'm here to help with your questions. Type 'exit' to end the chat.")

    def preprocess_text(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [ps.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    def build_qa_pairs(self):
        sentences = []
        responses = []
        
        sentences.extend(FAQ["greetings"])
        responses.extend(["Hello! How can I help you today?"] * len(FAQ["greetings"]))
        
        sentences.extend(FAQ["goodbye"])
        responses.extend(["Goodbye! Have a great day!"] * len(FAQ["goodbye"]))
        
        for question, answer in FAQ["product_info"]:
            sentences.append(question)
            responses.append(answer)
        
        return sentences, responses

    def get_bot_response(self, user_input):
        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        
        similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        best_match_idx = similarity_scores.argmax()
        
        if similarity_scores[best_match_idx] > 0.5:
            return self.response_list[best_match_idx]
        return "I'm sorry, I don't have information about that. Can you please rephrase your question?"

    def display_user_message(self, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(END, f"You: {message}\n", "user")
        self.chat_history.configure(state='disabled')
        self.chat_history.see(END)

    def display_bot_message(self, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(END, f"Bot: {message}\n", "bot")
        self.chat_history.configure(state='disabled')
        self.chat_history.see(END)

    def send_message(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
            
        if user_text.lower() == 'exit':
            self.master.destroy()
            return
            
        self.display_user_message(user_text)
        self.user_input.delete(0, END)
        
        try:
            response = self.get_bot_response(user_text)
            self.display_bot_message(response)
        except Exception as e:
            self.display_bot_message(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    root = Tk()
    chatbot = ChatbotGUI(root)
    root.mainloop()