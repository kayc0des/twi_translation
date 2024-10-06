from transformers import MarianMTModel, MarianTokenizer

# Load the pretrained model and tokenizer
model_name = "model/"
tokenizer_name = "model/"

model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

def translate(text, model, tokenizer):
    '''
    Function that translates a given text to Twi

    Args:
        text -> the text to be translated
        trainer -> trainer instance that contains the model
        tokenizer -> tokenizer instance to tokenize the text

    Returns:
        Transalted text
    '''
    input_encodings = tokenizer(text, return_tensors='pt', padding=True)

    # Generate translation
    translated_tokens = model.generate(**input_encodings)

    # Decode the output
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    return translated_text

# Example translation
text_to_translate = ["I am quite hungry"]
translated_text = translate(text_to_translate, model, tokenizer)
print(f"Translation: {translated_text}")