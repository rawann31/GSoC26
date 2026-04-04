import os
import csv
from PIL import Image
import google.generativeai as genai
import speech_recognition as sr

import pandas as pd
import json
import re

def configure_genai():
    genai.configure(api_key=API_KEY)

def parse_label(image_path):
    # Extract parts: healthy/non_healthy, wheat, disease_name
    parts = image_path.split(os.sep)
    if "healthy" in parts:
        return "wheat", "Healthy"
    else:
        # Last part is disease or pest folder
        disease_name = parts[-1]
        return "wheat", disease_name
# ----------------------------------------------------------


def ask_and_print(image_path, question):
    response = ask_gemini_about_image(image_path, question)
    print("\n--- Gemini Response ---")
    print(response)
    print("-----------------------\n")
# ----------------------------------------------------------

def get_voice_input(prompt="What is the disease or pest in the image?"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    print(prompt)
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return audio, text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return "",""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return "",""
# -------------------------------------------------------

def load_fewshot_images(train_dir, max_images_per_class=2):
    """
    Load images from train_dir structured as:
    train/
      class_name/
        image1.jpg
        image2.jpg
    Returns a list of tuples: (class_name, PIL.Image)
    """
    fewshot_examples = []

    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ][:max_images_per_class]

        for img_path in images:
            fewshot_examples.append((
                class_name.upper(),
                Image.open(img_path).convert("RGB")
            ))

    return fewshot_examples
# -------------------------------------------------------

def ask_gemini_about_image(
    image_path,
    prompt,
    fewshot=False,
    train_dir=None,
    max_images_per_class=2,
    model_name="gemini-3-flash-preview"
):
    """
    - image_path: path to the query image
    - provided prompt
    - Zero-shot if fewshot=False
    - Few-shot if train_dir is provided
    """

    def configure_genai():
        genai.configure(api_key=API_KEY)
        
    model = genai.GenerativeModel(model_name)
    query_image = Image.open(image_path).convert("RGB")

    content = [prompt]

    # ---------------- FEW-SHOT MODE ----------------
    if fewshot:
        if train_dir is None:
            raise ValueError("Few-shot requires train_dir to be provided.")

        fewshot_examples = load_fewshot_images(train_dir, max_images_per_class)

        content.append("\n--- LABELED TRAINING EXAMPLES ---\n")
        for class_name, img in fewshot_examples:
            content.append(f"Class: {class_name}")
            content.append(img)
        content.append("\n--- END OF TRAINING EXAMPLES ---\n")
        content.append("Now analyze the following query image.\n")

    # ---------------- QUERY IMAGE ----------------
    content.append(query_image)

    response = model.generate_content(
        content,
        generation_config={"temperature": 0.0}
    )

    return str(response)