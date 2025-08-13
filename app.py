from transformers import pipeline

# english to french
translator = pipeline("translation_en_to_fr")
translation = translator("I love you darling")
print(translation)
