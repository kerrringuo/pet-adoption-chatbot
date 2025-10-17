# Pet Adoption Chatbot

A simple NLP-powered chatbot that helps users find adoptable cats or dogs, and answer basic pet care questions.

---

## Overview

This chatbot uses **intent classification** and **entity extraction** to understand user requests such as:  
> "find a brown dog in KL"  
> "how to care for a kitten"

It supports conversational refinement — users can specify breed, color, and location step by step.

---

## Project Structure
src/
├── chatbot_pipeline.py # Main chatbot logic and conversation flow
├── intent_classifier.py # Intent classification (MiniLM + Logistic Regression)
├── entity_extractor.py # NER model for breed, color, state, age, etc.
├── synonyms.py # Synonym mapping and canonicalization
├── models/
│ ├── ner/ # Fine-tuned DistilBERT NER model
│ └── intent/ # Trained intent classifier + vectorizer

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/pet-adoption-chatbot.git
   cd pet-adoption-chatbot/src

   pip install -r requirements.txt

   python chatbot_pipeline.py

---

## Developer Notes

chatbot_pipeline.py
Main entry point. Handles conversation flow, connects the intent classifier and entity extractor, and manages session state.

intent_classifier.py
Predicts user intent (e.g., find_pet, pet_care, greeting, thank_you, goodbye).
Uses MiniLM embeddings + Logistic Regression.

entity_extractor.py
Performs Named Entity Recognition (NER) using a fine-tuned DistilBERT model.
Extracts entities such as breed, color, age, and state.

synonyms.py
Contains synonym mappings and normalization logic for consistent entity values.

models/
Stores trained model files. These load automatically when the chatbot starts.

---

## Example Conversation

Bot: Hello! 👋 I can help you find cats 🐱 or dogs 🐶 for adoption, or answer pet care questions.
You: hi i want a dog
Bot: Which state or area are you in? 🏙️
You: kl
Bot: Added state: Kuala Lumpur. Got it! Searching for dogs in Kuala Lumpur...
You: golden one
Bot: Added color: golden. Got it! Searching for golden dogs in Kuala Lumpur...
You: actually change to cat
Bot: Okay, updated pet type to cat. Got it! Searching for golden cats in Kuala Lumpur...
You: thanks
Bot: You're most welcome! 😊 Anything else you'd like to ask?
You: bye
Bot: Goodbye! 👋
