import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Set up OpenAI API key (make sure to set this in Streamlit secrets)
openai.api_key = st.secrets["mykey"]

# Load the CSV file
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert embeddings from strings to numpy arrays and check the dimensions
def convert_embedding(embedding_str):
    embedding = np.array(eval(embedding_str))
    if len(embedding.shape) != 1:  # Ensure it's a 1D array
        raise ValueError(f"Invalid embedding shape: {embedding.shape}")
    return embedding

df['Question_Embedding'] = df['Question_Embedding'].apply(convert_embedding)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Or use the correct model for your dataset

# Regenerate embeddings for the entire dataset if necessary (if using a new model)
# Uncomment this if you're regenerating the embeddings:
# df['Question_Embedding'] = df['Question'].apply(lambda q: model.encode(q))

# Streamlit interface
st.title('Health Q&A Assistant')

user_question = st.text_input("Ask a question about heart, lung, or blood health:")

if st.button('Get Answer'):
    if user_question:
        # Generate embedding for the user's question
        user_embedding = model.encode([user_question])[0]
        
        # Debugging: Check the shape of the user's embedding
        st.write(f"User question embedding shape: {user_embedding.shape}")

        # Print out first embedding from the dataset for comparison
        st.write(f"First dataset embedding shape: {df['Question_Embedding'].iloc[0].shape}")
        
        # Compute cosine similarity between user embedding and dataset embeddings
        try:
            df['similarity'] = df['Question_Embedding'].apply(
                lambda x: cosine_similarity([user_embedding], [x]).flatten()[0]
            )
        except ValueError as e:
            st.write(f"Error in cosine similarity calculation: {e}")
            st.stop()
        
        # Find the highest similarity score
        max_similarity = df['similarity'].max()
        best_match = df.loc[df['similarity'].idxmax()]
        
        # Debugging: Print out max similarity and corresponding answer
        st.write(f"Max similarity: {max_similarity}")
        st.write(f"Best match answer: {best_match['Answer']}")
        
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
