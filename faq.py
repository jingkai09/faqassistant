import streamlit as st
import pandas as pd
import numpy as np
import openai 

openai.api_key =  st.secrets["mykey"]

# Load data and embeddings
data = pd.read_csv("qa_dataset_with_embeddings.csv")
question_embeddings = data['Question_Embedding'].values  # Assuming embeddings are stored as a NumPy array

def find_answer(user_question, question_embeddings, data):
    # Load the user's question embedding (assuming you have a function to generate this)
    user_embedding = generate_user_embedding(user_question)  # Replace with your embedding generation function

    # Calculate cosine similarity
    similarities = np.dot(question_embeddings, user_embedding) / (np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(user_embedding))

    # Find the most similar question
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]

    # Set a similarity threshold (you can adjust this)
    threshold = 0.7

    if similarity_score > threshold:
        answer = data['Answer'][most_similar_index]
        return answer, similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None

def main():
    st.title("Health FAQ Assistant")

    user_question = st.text_input("Ask your health question:")
    if st.button("Submit"):
        answer, similarity_score = find_answer(user_question, question_embeddings, data)
        st.text_area("Answer:", answer)
        if similarity_score:
            st.write("Similarity score:", similarity_score)

if __name__ == "__main__":
    main()
