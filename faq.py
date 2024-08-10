import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key =  st.secrets["mykey"]

# Load the dataset
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert the embeddings from strings to arrays
def convert_embedding(x):
    try:
        return np.fromstring(x.strip("[]"), sep=",")
    except:
        return np.array([])

df['Question_Embedding'] = df['Question_Embedding'].apply(convert_embedding)

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the stored questions
def get_embeddings(questions):
    return np.array([model.encode(q) for q in questions])

# Streamlit app title
st.title('Smart FAQ Assistant')

# Input field for user questions
user_question = st.text_input("Enter your question about heart, lung, or blood health:")

# Button to trigger the answer search
search_button = st.button("Find Answer")

# When the button is clicked
if search_button and user_question:
    # Generate embedding for the user's question
    user_embedding = model.encode([user_question])
    
    # Ensure embeddings are in the correct shape for comparison
    if not df['Question_Embedding'].empty:
        embeddings = get_embeddings(df['Question'].tolist())
        
        # Calculate cosine similarity between user question and dataset questions
        similarities = cosine_similarity(user_embedding, embeddings)
        
        # Find the most similar question
        max_similarity = similarities.max()
        if max_similarity > 0.7:  # Adjust this threshold as needed
            most_similar_idx = similarities.argmax()
            answer = df.iloc[most_similar_idx]['Answer']
            st.write(f"**Answer:** {answer}")
            st.write(f"**Similarity Score:** {max_similarity:.2f}")
        else:
            st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
    else:
        st.error("No questions found in the dataset.")

# Clear button to reset the input field
if st.button("Clear"):
    st.experimental_rerun()  # Reset the input field

# Option to show common FAQs
if st.checkbox("Show common FAQs"):
    st.write(df[['Question', 'Answer']].head(20))  # Display the first 20 FAQs
