from transformers import pipeline

classifier = pipeline("ner", model="Liubavaa/mountain_ner_model")

# example of text
text = "The highest mountain on Earth is Everest in the Himalayas of Asia, whose summit is 8,850 m (29,035 ft) above mean sea level."
classifier(text)
