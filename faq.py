import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert the embeddings from strings to arrays
def convert_embedding(x):
    try:
        return np.fromstring(x.strip("[]"), sep=",")
    except:
        return np.array([])  # Return empty array on conversion failure

df['Question_Embedding'] = df['Question_Embedding'].apply(convert_embedding)

# Check if embeddings are loaded correctly
if len(df['Question_Embedding'].iloc[0]) == 0:
    st.error("Error loading embeddings. Please check the embedding format in the CSV file.")
else:
    # Load the pre-trained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

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
        
        # Ensure embeddings are 2D arrays
        embeddings = np.vstack(df['Question_Embedding'].values)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # Reshape if only one embedding
        
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

    # Clear button to reset the input field
    if st.button("Clear"):
        st.experimental_rerun()  # Reset the input field

    # Option to show common FAQs
    if st.checkbox("Show common FAQs"):
        st.write(df[['Question', 'Answer']].head(10))  # Display the first 10 FAQs
