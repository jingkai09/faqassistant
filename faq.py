import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key =  st.secrets["mykey"]

# Load the CSV file
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert embeddings from strings to numpy arrays
df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(eval(x)))

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit interface
st.title('Health Q&A Assistant')

user_question = st.text_input("Ask a question about heart, lung, or blood health:")

if st.button('Get Answer'):
    if user_question:
        # Generate embedding for the user's question
        user_embedding = model.encode([user_question])[0]
        
        # Compute cosine similarity between user embedding and dataset embeddings
        df['similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity([user_embedding], [x]).flatten()[0])
        
        # Find the highest similarity score
        max_similarity = df['similarity'].max()
        best_match = df.loc[df['similarity'].idxmax()]
        
        # Set a threshold for the similarity score
        threshold = 0.75
        if max_similarity > threshold:
            st.write(f"**Answer:** {best_match['Answer']}")
            st.write(f"**Similarity Score:** {max_similarity:.2f}")
        else:
            st.write("I apologize, but I don't have information on that topic yet. Could you please ask another question?")
    else:
        st.write("Please enter a question.")

# Clear button to reset input field
if st.button('Clear'):
    st.experimental_rerun()
