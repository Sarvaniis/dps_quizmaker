import streamlit as st
from PIL import Image
import pytesseract
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer
import language_tool_python
import random
import spacy
# import spacy.cli  # Add this import for the spaCy cli
import os

nlp = spacy.load("en_core_web_sm")

# # Ensure model is installed
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")


# 🔽 Download the spaCy model (only runs if not already downloaded)
# spacy.cli.download("en_core_web_sm")

# 🌐 Page settings
st.set_page_config(page_title="Text Understanding App", layout="wide")
st.title("Welcome to the Text Understanding App")
st.sidebar.header("Options")

# 📦 Load all models
@st.cache_resource
def load_models():
    summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Use remote server to avoid Java dependency
    grammar_tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org/')
    
    spacy_model = spacy.load("en_core_web_sm")
    
    return summarizer_model, grammar_tool, spacy_model


summarizer, tool, nlp = load_models()

# 🔤 Translation helper
def translate_text(text, target_lang='en'):
    model_name = f'Helsinki-NLP/opus-mt-auto-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ✅ Grammar scoring
def evaluate_text(text):
    matches = tool.check(text)
    score = max(0, 100 - len(matches) * 5)
    return score, matches

# 📌 OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# 🧠 Auto Quiz Generator
def generate_quiz(text):
    doc = nlp(text)
    questions = []

    for sent in doc.sents:
        entities = [ent for ent in sent.ents if ent.label_ in ("PERSON", "ORG", "GPE", "DATE", "EVENT", "WORK_OF_ART")]
        if not entities:
            noun_chunks = list(sent.noun_chunks)
            if noun_chunks:
                entity = random.choice(noun_chunks).text
            else:
                continue
        else:
            entity = random.choice(entities).text

        question_text = sent.text.replace(entity, "______")
        correct_answer = entity

        all_entities = [ent.text for ent in doc.ents if ent.text != entity and len(ent.text.split()) < 4]
        distractors = random.sample(all_entities, min(3, len(all_entities))) if all_entities else ["Option A", "Option B", "Option C"]

        options = [correct_answer] + distractors
        random.shuffle(options)

        question = f"Fill in the blank: {question_text}"
        questions.append({"question": question, "options": options, "answer": correct_answer})

    return questions

# 📄 Tabs
tab1, tab2 = st.tabs(["📜 Text Input", "🖼️ Image Upload"])

# ===================== 📜 TEXT INPUT TAB =====================
with tab1:
    st.header("Text Input Analysis")
    user_text = st.text_area("Enter your text:", height=200)

    if user_text.strip():
        st.subheader("🔍 Language Detection")
        if len(user_text.split()) < 2:
            lang = "en"
            st.info("Too short to detect language, defaulting to English.")
        else:
            try:
                lang = detect(user_text)
                st.success(f"Detected Language: {lang}")
            except Exception as e:
                lang = "en"
                st.error(f"Detection failed. Defaulted to English. Error: {e}")

        # 📝 Summary
        st.subheader("📝 Summary")
        if len(user_text.split()) < 30:
            st.warning("Text too short to summarize.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    summary = summarizer(user_text, max_length=60, min_length=25, do_sample=False)[0]['summary_text']
                    st.info(summary)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")

        # 🌐 Translation
        st.subheader("🌐 Translate to English")
        if lang != "en":
            try:
                with st.spinner("Translating..."):
                    translated = translate_text(user_text, target_lang="en")
                    st.code(translated)
            except Exception as e:
                st.warning(f"Translation failed: {e}")
        else:
            st.write("No translation needed. Already in English!")

        # 📊 Grammar Evaluation (✅ ADDED)
        st.subheader("📊 Grammar Evaluation")
        score, issues = evaluate_text(user_text)
        st.metric(label="Grammar Score", value=f"{score}/100")

        if issues:
            st.write("✏️ Common Issues:")
            for i in issues[:5]:
                st.write(f"• {i.message} (Suggestions: {i.replacements})")

        # 🧩 Quiz Generation
        st.subheader("🧩 Auto-Generated Quiz")
        quiz = generate_quiz(user_text)
        if quiz:
            for i, q in enumerate(quiz, 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                for opt in q['options']:
                    st.markdown(f"- {opt}")
                st.markdown(f"<small style='color:gray;'>Answer: {q['answer']}</small>", unsafe_allow_html=True)
        else:
            st.info("Not enough data to generate quiz.")

# ===================== 🖼️ IMAGE TAB =====================
with tab2:
    st.header("Upload Image for OCR & Quiz")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text..."):
            extracted = extract_text_from_image(img)

        st.subheader("📝 Extracted Text")
        st.text_area("OCR Result", extracted, height=200)

        st.subheader("📊 Grammar Evaluation")
        score, issues = evaluate_text(extracted)
        st.metric(label="Grammar Score", value=f"{score}/100")

        if issues:
            st.write("✏️ Common Issues:")
            for i in issues[:5]:
                st.write(f"• {i.message} (Suggestions: {i.replacements})")

        st.subheader("🧩 Quiz from Image Text")
        quiz = generate_quiz(extracted)
        if quiz:
            for i, q in enumerate(quiz, 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                for opt in q['options']:
                    st.markdown(f"- {opt}")
                st.markdown(f"<small style='color:gray;'>Answer: {q['answer']}</small>", unsafe_allow_html=True)
        else:
            st.info("Not enough OCR content to generate quiz.")
