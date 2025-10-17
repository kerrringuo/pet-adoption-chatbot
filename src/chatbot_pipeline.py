from intent_classifier import IntentClassifier
from entity_extractor import EntityExtractor
from rapidfuzz import process
from synonyms import SYNONYMS, canonicalize
from transformers.utils import logging as hf_logging
import warnings, re

# Silence noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# Required and optional entities for pet search
REQUIRED_ENTITIES = ["PET_TYPE", "STATE"]
OPTIONAL_ENTITIES = ["BREED", "COLOR", "SIZE", "GENDER", "AGE", "FURLENGTH"]


# ---------------------------------------------------------------------------
# AUTOCORRECT
# ---------------------------------------------------------------------------
def autocorrect_text(text, known_words=None, threshold=85):
    """Light typo correction for meaningful words only."""
    if known_words is None:
        known_words = [
            "dog", "cat", "adopt", "adoption", "puppy", "kitten",
            "male", "female", "small", "large", "brown", "white", "black",
            "golden", "cream", "short", "long", "fur",
            "Johor", "Penang", "Melaka", "Selangor", "Kuala", "Lumpur",
            "Perak", "Sabah", "Sarawak"
        ]
    words, corrected = text.split(), []
    for w in words:
        if len(w) <= 3:  # skip very short tokens
            corrected.append(w)
            continue
        match, score, _ = process.extractOne(w, known_words)
        corrected.append(match if score >= threshold and match.lower() != w.lower() else w)
    return " ".join(corrected)


# ---------------------------------------------------------------------------
# CHATBOT PIPELINE
# ---------------------------------------------------------------------------
class ChatbotPipeline:
    """End-to-end pet adoption chatbot with clean fallbacks and context."""

    def __init__(self):
        self.intent_clf = IntentClassifier()
        self.ner_extractor = EntityExtractor()
        self.session = {"intent": None, "entities": {}, "greeted": False}

    # -----------------------------------------------------------------------
    # MAIN MESSAGE HANDLER
    # -----------------------------------------------------------------------
    def handle_message(self, user_input: str) -> str:
        user_input = user_input.strip()

        # --- Show greeting if user presses Enter at start ---
        if not user_input:
            return self._get_greeting()

        # --- Typo correction ---
        user_input = autocorrect_text(user_input)

        # --- Small talk shortcuts ---
        lower = user_input.lower()
        if lower in ["no", "nope", "nah"]:
            return "Alright 😊 Let me know anytime if you change your mind."
        if lower in ["hi", "hey", "hello"]:
            self.session["greeted"] = True
            return "Hello again! 👋 How can I help you today?"

        # # --- First-time greeting ---
        # if not self.session["greeted"]:
        #     self.session["greeted"] = True
        #     return self._get_greeting()

        # --- Intent classification ---
        intent, conf = self.intent_clf.predict(user_input)

        if not self.session["greeted"]:
            self.session["greeted"] = True
            if intent == "greeting":
                return self._get_greeting()

        prev_intent = self.session.get("intent")

        # --- Handle unknown / low-confidence intents gracefully ---
        if intent == "unknown" or conf < 0.55:
            return self._handle_unknown(user_input)

        # --- Intent switching ---
        if self._is_new_intent(intent, prev_intent):
            self.session = {"intent": intent, "entities": {}, "greeted": True}

        self.session["intent"] = intent

        # --- Route by intent ---
        if intent == "find_pet":
            return self._handle_find_pet(user_input)
        if intent == "pet_care":
            # RAG INTEGRATION PLACEHOLDER
            return "(RAG) 🧠 Fetching pet care advice... (placeholder)"
        if intent == "thank_you":
            return "You're most welcome! 😊 Anything else you'd like to ask?"
        if intent == "greeting":
            return "Hello again! 👋 How can I help you today?"
        if intent == "goodbye":
            return "Goodbye! 👋 Hope you find your perfect furry friend 🐶🐱"

        # --- Default fallback ---
        return (
            "I'm not sure I understood. 🤔 "
            "You can say 'I want to adopt a cat in Penang' or 'How to care for a puppy?'."
        )

    # -----------------------------------------------------------------------
    # FIND-PET HANDLER
    # -----------------------------------------------------------------------
    def _handle_find_pet(self, user_input: str) -> str:
        ents = self._extract_entities(user_input)
        if "NOTICE" in ents:
            return ents["NOTICE"]
        if ents:
            return self._update_entities_and_respond(ents)

        missing = [e for e in REQUIRED_ENTITIES if e not in self.session["entities"]]
        return self.ask_for(missing[0]) if missing else self._confirm_and_search()

    # -----------------------------------------------------------------------
    # UNKNOWN / LOW-CONFIDENCE HANDLER
    # -----------------------------------------------------------------------
    def _handle_unknown(self, user_input: str) -> str:
        """Handles unclear or nonsense inputs gracefully."""
        if self.session.get("intent") == "find_pet":
            if len(user_input.split()) <= 2:
                token = user_input.strip()
                # Only wrap descriptive words, not short codes
                if len(token.split()) == 1 and len(token) > 2:
                    pseudo = f"I want a {token} {self.session['entities'].get('PET_TYPE', 'pet')}"
                    ents = self._extract_entities(pseudo)
                else:
                    ents = self._extract_entities(user_input)
                if ents:
                    return self._update_entities_and_respond(ents)

            # Second try, direct extraction
            ents = self._extract_entities(user_input)
            if "NOTICE" in ents:
                return ents["NOTICE"]
            if ents:
                return self._update_entities_and_respond(ents)

            # Graceful confusion fallback
            if any(self.session["entities"].get(e) for e in REQUIRED_ENTITIES):
                return (
                    "Hmm, I didn’t quite catch that. "
                    "Could you tell me a bit more — like 'small cream dog' or 'female cat'?"
                )

            return (
                "I'm not sure I understood that. "
                "Could you tell me what kind of pet and which state you’re in? "
                "For example: 'I'm looking for a cat in Johor'."
            )

        # Generic fallback
        return (
            "I'm not sure I understood. 🤔 "
            "You can say 'I want to adopt a cat in Penang' or 'How to care for a puppy?'."
        )

    # -----------------------------------------------------------------------
    # ENTITY EXTRACTION & VALIDATION
    # -----------------------------------------------------------------------
    def _extract_entities(self, text: str):
        """Runs NER extraction, applies synonym-based fallbacks, and filters invalid or out-of-scope entities."""
        ents = self.ner_extractor.extract(text)

        # --- Quick keyword check for out-of-scope animals ---
        species_out_of_scope = [
            "hamster", "hamsters", "rabbit", "rabbits", "bird", "birds",
            "parrot", "parrots", "fish", "fishes", "snake", "turtle"
        ]
        if any(w in text.lower() for w in species_out_of_scope):
            return {
                "NOTICE": (
                    "Sorry, I currently only help with cats 🐱 and dogs 🐶. "
                    "Would you like to search for one of those instead?"
                )
            }

        # --- Basic sanity checks ---
        if not text.strip() or len(text.strip()) < 2:
            return {"NOTICE": "Hmm, I didn’t quite catch that. Could you try again?"}
        if len(text) > 4 and not any(ch in "aeiou" for ch in text.lower()):
            return {"NOTICE": "Hmm, I didn’t quite catch that. Could you try again?"}

        # --- Remove placeholders / irrelevant tokens ---
        if "PET_TYPE" in ents and ents["PET_TYPE"].lower() in ["one", "it", "animal", "pet"]:
            ents.pop("PET_TYPE")
        if "AGE" in ents and str(ents["AGE"]).lower() in ["one", "1", "single", "johor"]:
            ents.pop("AGE")

        # --- Restrict supported species ---
        if "PET_TYPE" in ents and ents["PET_TYPE"].lower() not in ["dog", "cat"]:
            return {
                "NOTICE": (
                    "Sorry, I currently only help with cats 🐱 and dogs 🐶. "
                    "Would you like to search for one of those instead?"
                )
            }

        # --- Breed validation ---
        valid_breeds = [b.lower() for b in SYNONYMS.keys()]
        if "BREED" in ents:
            breed_val = ents["BREED"].lower()
            species_terms = [
                "hamster", "rabbit", "bird", "parrot", "fish", "turtle", "snake", "guinea",
                "hamsters", "rabbits", "birds", "parrots", "fishes"
            ]
            if (
                breed_val in species_terms
                or (breed_val not in valid_breeds and not re.search(r"[aeiou]", breed_val))
            ):
                ents.pop("BREED")

        # --- Drop short nonsense tokens ---
        for k, v in list(ents.items()):
            if len(v) < 3 or not any(ch in "aeiou" for ch in v.lower()):
                ents.pop(k)

        # --- Color keyword fallback using synonym map ---
        color_variants = []
        for canon, variants in SYNONYMS.items():
            if canon.lower() in [
                "black", "white", "brown", "golden", "cream", "gray",
                "orange", "yellow", "blue", "red", "tabby", "calico", "tortoiseshell"
            ]:
                color_variants.extend([canon.lower()] + [v.lower() for v in variants])

        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        for word in tokens:
            if word in color_variants and "COLOR" not in ents:
                ents["COLOR"] = canonicalize(word)
                break

        # --- No meaningful entities detected ---
        if not ents:
            return {"NOTICE": "Hmm, I didn’t quite catch that. Could you try again?"}

        return ents


    # -----------------------------------------------------------------------
    # UPDATE SESSION + RESPOND
    # -----------------------------------------------------------------------
    def _update_entities_and_respond(self, ents: dict) -> str:
        if "NOTICE" in ents:
            return ents["NOTICE"]

        confirm = []
        old_pet = self.session["entities"].get("PET_TYPE", "").lower()
        new_pet = ents.get("PET_TYPE", "").lower()

        if new_pet:
            if old_pet and new_pet != old_pet:
                for k in ["BREED", "FURLENGTH"]:
                    self.session["entities"].pop(k, None)
                confirm.append(f"Okay, updated pet type to {new_pet}.")
            self.session["entities"]["PET_TYPE"] = new_pet

        for k, v in ents.items():
            if k in ["PET_TYPE", "NOTICE"]:
                continue
            prev = self.session["entities"].get(k)
            readable = k.replace("_", " ").lower()
            if prev and prev != v:
                confirm.append(f"Okay, updated {readable} to {v}.")
            elif not prev:
                confirm.append(f"Added {readable}: {v}.")
            self.session["entities"][k] = v

        missing = [e for e in REQUIRED_ENTITIES if e not in self.session["entities"]]
        msg = " ".join(confirm)
        return f"{msg} {self.ask_for(missing[0])}" if missing else f"{msg} {self._confirm_and_search()}"

    # -----------------------------------------------------------------------
    # FINAL SEARCH MESSAGE
    # -----------------------------------------------------------------------
    def _confirm_and_search(self) -> str:
        ents = self.session["entities"]
        pet = ents.get("PET_TYPE", "pet")
        state = ents.get("STATE", "your area")
        details = [v for v in [
            ents.get("SIZE"), ents.get("COLOR"), ents.get("GENDER"),
            ents.get("AGE"), ents.get("FURLENGTH"), ents.get("BREED")
        ] if v]
        desc = " ".join(details + [pet])
        if not ents.get("BREED"):
            desc += "s"
        return f"Got it! Searching for {desc} in {state}..."

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------
    def _is_new_intent(self, intent, prev_intent):
        return (
            intent not in ["unknown", None]
            and prev_intent not in ["unknown", None]
            and intent != prev_intent
        )

    def ask_for(self, entity: str) -> str:
        prompts = {
            "PET_TYPE": "Are you looking for a dog or a cat?",
            "STATE": "Which state or area are you in?",
            "BREED": "Do you have a preferred breed?",
            "COLOR": "Any color preference?",
            "SIZE": "Do you prefer small or large pets?",
            "GENDER": "Male or female?",
            "AGE": "Puppy/kitten or adult?",
            "FURLENGTH": "Do you prefer short or long fur?",
        }
        return prompts.get(entity, f"Could you tell me the {entity.lower()}?")

    def _get_greeting(self) -> str:
        return (
            "Hello! 👋 I can help you find cats 🐱 or dogs 🐶 for adoption, "
            "or answer pet care questions.\n"
            "You can say things like:\n"
            "• I'm looking for a dog in Johor\n"
            "• How to care for a cat?\n"
            "So, what would you like to do today?"
        )

    def reset(self) -> str:
        self.session = {"intent": None, "entities": {}, "greeted": False}
        return self._get_greeting()


# ---------------------------------------------------------------------------
# LOCAL TESTING
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = ChatbotPipeline()
    print("Bot:", bot.handle_message(""))
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit", "bye", "goodbye", "stop", "end"]:
            print("Bot: Goodbye! 👋")
            break
        if msg.lower() in ["restart", "reset", "new chat"]:
            print("Bot:", bot.reset())
            continue
        print("Bot:", bot.handle_message(msg))