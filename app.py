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

    st.header("Problem-Solving Agent")
    st.subheader("Agent:")
    st.markdown("The agent is the Next Word Prediction AI itself. It predicts the next word in a given sequence, serving as a tool for applications like predictive text, auto-completion, or virtual keyboards.")

    st.subheader("Environment:")
    st.markdown("The environment is the textual context or sequence of words in which the Next Word Prediction AI operates. It represents the input where the agent needs to predict the next word.")

    st.subheader("Goal:")
    st.markdown("The goal for the agent is to accurately predict the next word based on the context provided by the preceding words. The aim is to enhance user experience by offering coherent and contextually relevant predictions.")

    st.subheader("Actions:")
    st.markdown("The actions of the agent involve selecting the next word from a set of possible words in the vocabulary.")

    st.subheader("Performance Measure:")
    st.markdown("The performance of the agent is evaluated based on the accuracy of its predictions. Metrics such as precision, recall, or other measures related to prediction quality are used to assess its effectiveness.")

    st.subheader("Problem-solving Strategy:")
    st.markdown("The agent employs machine learning models, like recurrent neural networks (RNNs) or transformers, to learn patterns and relationships within the language, enabling it to make accurate predictions.")

    st.subheader("Learning:")
    st.markdown("The agent learns from training data, continuously adapting its predictive model based on observed patterns in the input sequences, improving its ability to make contextually appropriate predictions over time.")

    st.code("function SIMPLE-NEXT-WORD-PREDICTION-AGENT(context) returns a predicted word")
    st.code("inputs: context, a sequence of words")
    st.code("static: model, a language model for prediction, initially empty")
    st.code("prediction, a word predicted by the model, initially null")
    st.code("learning, a process to update the language model based on new data")
    st.code("predict-WORD(model, context) if prediction is null then do learning(observeNewData(context)) model-UPDATE(model, context) prediction-MAKE-PREDICTION(model, context)")
    st.code("return prediction")
    st.markdown("In this example, the ""SIMPLE-NEXT-WORD-PREDICTION-AGENT"" is a basic agent for next-word prediction. It uses a language model to predict the next word in a given context. The agent observes the context, updates its language model, and makes predictions based on learned patterns. The learning process allows the agent to adapt to new data and improve its predictive capabilities over time. The agent's goal is to provide accurate predictions for the next word in a sequence.")

    st.title("Next Word Prediction AI - Model-based Reflex Agent")

    st.header("Reasoning:")
    st.markdown(
        "It uses a model, typically a language model, to understand the relationships and patterns in the given context (sequence of words). "
        "The agent's actions (word predictions) are based on the learned model, incorporating knowledge about the language structure."
    )

    st.header("Reflex Nature:")
    st.markdown(
        "The agent is reactive, responding to the current context without considering long-term consequences. "
        "It doesn't have a comprehensive world model but relies on the learned model for making predictions."
    )

    st.header("Model Update:")
    st.markdown(
        "The agent's model is updated over time, learning from new data to adapt to evolving language patterns."
    )

    st.title("The most important algorithms used in building Next Word Prediction AI is the Recurrent Neural Network (RNN)")

    paragraph = """
    RNN stands for Recurrent Neural Network, and it is a type of artificial neural network designed for processing sequences of data. 
    Unlike traditional neural networks, which process each input independently, RNNs have the ability to capture and remember information 
    from previous inputs in the sequence. This makes them particularly well-suited for tasks involving sequential or time-dependent data, 
    such as natural language processing, speech recognition, and time series prediction.
    """

    st.write(paragraph)

    st.title("Applications of Next Word Prediction AI")

    st.header("1. Mobile Keyboards:")
    st.markdown(
        "**How it works:** When users type on their mobile keyboards, the Next Word Prediction AI suggests the next word in the sentence based on the context of the words typed so far."
    )
    st.markdown(
        "**Explanation:** The AI analyzes the sequence of typed words and predicts the next word by considering language patterns. "
        "This speeds up typing as users can tap on the suggested word instead of typing the entire word."
    )

    st.header("2. Text Editors and Word Processors:")
    st.markdown(
        "**How it works:** In applications like Microsoft Word or Google Docs, Next Word Prediction AI can assist users by suggesting the next word as they type."
    )
    st.markdown(
        "**Explanation:** The AI analyzes the ongoing sentence to predict the most likely next word, providing users with real-time suggestions for completing their sentences more efficiently."
    )

    st.header("3. Search Engines:")
    st.markdown(
        "**How it works:** Search engines often use predictive text to suggest search queries or phrases based on the initial keywords entered by the user."
    )
    st.markdown(
        "**Explanation:** As users type their search queries, the AI predicts the next word or phrase by considering popular search terms, helping users refine their queries quickly."
    )

    st.header("4. Messaging Apps:")
    st.markdown(
        "**How it works:** In messaging applications like WhatsApp or iMessage, Next Word Prediction AI assists users by suggesting the next word or completing sentences."
    )
    st.markdown(
        "**Explanation:** The AI analyzes the ongoing conversation to predict the next word, providing users with suggestions that align with their communication style, making typing more efficient."
    )

    st.header("5. Voice Recognition Systems:")
    st.markdown(
        "**How it works:** Next Word Prediction AI can be integrated into voice recognition systems to predict the next word in a spoken sentence."
    )
    st.markdown(
        "**Explanation:** As users speak, the AI analyzes the context and predicts the next word to improve the accuracy of voice-to-text conversion. "
        "This enhances the overall performance of voice recognition systems."
    )

    st.header("6. Email Clients:")
    st.markdown(
        "**How it works:** In email applications, Next Word Prediction AI can aid users in composing emails by suggesting the next word or completing sentences."
    )
    st.markdown(
        "**Explanation:** The AI considers the context of the email content to predict the next word, facilitating faster and more accurate email composition."
    )

    st.title("Next Word Prediction AI Conclution")

    paragraph2 = """
    Next Word Prediction AI is a powerful application of artificial intelligence that significantly enhances user experience in various contexts. 
    By leveraging strategies such as recurrent neural networks, it excels in predicting the next word in a sequence based on contextual analysis. 
    This technology finds widespread usage in mobile keyboards, text editors, search engines, messaging apps, voice recognition systems, and email clients. 
    The key strengths lie in its ability to adapt to language patterns, provide real-time suggestions, and improve typing efficiency across different platforms. 
    The continuous learning nature of these models, coupled with their application in problem-solving agents, positions Next Word Prediction AI as a valuable tool 
    that not only anticipates user intent but also contributes to the broader landscape of natural language processing.
    """

    st.write(paragraph2)

    st.title("Try Out the model:")

    user_input = st.text_input("Enter a sentence:", "Which two counties are the biggest economic powers")

    if st.button("Predict"):
        try:
            predicted_words = run_model(user_input)
            st.success(f"Possible next words: {predicted_words}")
        except Exception as e:
            st.error(f"Error occurred while making the prediction: {e}")

    st.markdown("---")

    # Adding hyperlinks with JavaScript to redirect on click
    st.markdown(
        """
        <div style="text-align: center;">
        Made by
            <a href="https://in.linkedin.com/in/harshatr" target="_blank" rel="noopener noreferrer" onclick="window.open('https://in.linkedin.com/in/harshatr'); return false;">Harsha TR</a>
            &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/pkb1202" target="_blank" rel="noopener noreferrer" onclick="window.open('https://www.linkedin.com/in/pkb1202'); return false;">Prabhanjana K</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
