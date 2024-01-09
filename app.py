import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

model_name = "allenai/t5-small-next-word-generator-qoogle"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=T5Config.from_pretrained(model_name))

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

def main():
    st.title("Next Word Prediction App")

    user_input = st.text_input("Enter a sentence:", "Which two counties are the biggest economic powers")

    if st.button("Predict"):
        try:
            predicted_words = run_model(user_input)
            st.success(f"Possible next words: {predicted_words}")
        except Exception as e:
            st.error(f"Error occurred while making the prediction: {e}")

if __name__ == "__main__":
    main()
