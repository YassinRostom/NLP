import numpy as np
import pandas as pd

import spacy

from collections import Counter
from itertools import combinations

import networkx as nx
from pyvis.network import Network


# Pipeline Architecture:
# 1. Data Ingestion and NLP Pipeline Setup.
# 2. Full Text Preprocessing
# 3. Determine the frequency of pair-words
    # 3.1. Generate a CSV that have pair words with more than 100 counts
    # 3.2. Generate a CSV that contains Single Word Frequency 
# 4. Visualize the co-occurrence network


# 1. Data Ingestion from a CSV file.
# ===================================
def ingest_data_and_nlp_pipeline_setup(csv_file):
    df = pd.read_csv(csv_file)
    print("Data Ingested Successfully!")
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Apply spaCy to each row in the column and store the Doc object in a new column
    # NB: That makes it easier to manage the doc obj
    df['spacy_doc'] = df['short_description'].apply(lambda x: nlp(str(x)))
    return df

# 2. Full Text Preprocessing
# ===================================

def get_tokens_no_stop(doc):
    '''
        Only keep tokens if
            - Not a stop word
            - Not punctuation
            - Not in the remove_words list
            - A product name
            - POS is NOUN/VERB/PROPN/ADJ (ents_of_interest - contains spans whihc have spaces so that wwon't be useful)
            - NER is ORG/PRODUCT/EVENT    
    '''
    global ents_of_interest # access the global set
    tokens_lst = []
    
    product_names = ["smartsheet", "miro", "buildsmart", "box", "onedrive", "teams", "sharepoint"]
    
    remove_words = ['|','ge','aerospace','gea','login]gea','re','-']
    
    
    for token in doc:
        if token.text != token.is_stop and token.text != token.is_punct:
            lemma = token.lemma_.lower()
            # Check if token is a product name
            if lemma in remove_words:
                pass
            elif lemma in product_names:
                tokens_lst.append(lemma)
            # Check if token is part of an entity of interest
            elif lemma in ents_of_interest:
                tokens_lst.append(lemma) 
            # Check POS tag
            elif token.pos_ in {'NOUN', 'VERB', 'PROPN', 'ADJ'}:
                tokens_lst.append(lemma)       

    return tokens_lst


# 3. Determine the frequency of pair-words
# ========================================
def pair_words_frequency(df):
    '''
        This function calculates the frequency of pairs of tokens across all rows.
        The result is stored in a Counter object.
    '''
    # Create a Counter to store co-occurrence counts
    co_occurrence_counter = Counter()

    # Go through each row of filtered tokens
    for tokens in df['tokens_Filtered']:
        # Get unique tokens for this row
        unique_tokens = set(tokens)

        # Create a list of all the combination of words in a record
        all_comb_of_words = combinations(unique_tokens, 2)

        # For each combination of words = increment by 1
        for pair in all_comb_of_words:
            co_occurrence_counter[pair] += 1

    return co_occurrence_counter

# 3.1 Generate a CSV that have pair words with more than 100 counts
def generate_pair_words_csv(co_occurrence_counter):
    lst_words = []

    # Only add word pairs with a count greater than 100
    for (word1,word2),weight in co_occurrence_counter.items():
        if weight > 100:
            lst_words.append([word1, word2, weight])

    df_summery = pd.DataFrame(data=lst_words, columns=['word1', 'word2', 'Count'])

    df_summery = df_summery.sort_values(by='Count', ascending=False)

    # generate a csv 
    df_summery.to_csv('co_occurrence_summery.csv', index=False)
    print("Co-occurrence summary CSV generated successfully!")

# 3.2. Generate a CSV that contains Single Word Frequency 
def single_word_frequency_csv(df):

    lst_all_tickets = []

    # Flatten the list of lists into a single list
    for lst_words in df['tokens_Filtered']:
        for word in lst_words:
            lst_all_tickets.append(word)

    # Count the frequency of each keyword
    keyword_freq = Counter(lst_all_tickets)

    lst_words = []

    # Only add word pairs with a count greater than 100
    for (word1),weight in keyword_freq.items():
        if weight > 100:
            lst_words.append([word1, weight])

    df_summery = pd.DataFrame(data=lst_words, columns=['word1', 'Count'])

    df_summery = df_summery.sort_values(by='Count', ascending=False)

    # generate a csv 
    df_summery.to_csv('single_word_freq_summery.csv', index=False)
    print("single_word_freq_summery CSV generated successfully!")

# 4. Visualize the co-occurrence network
def visualize_co_occurrence_network(co_occurrence_counter):
    '''
        This function visualizes the co-occurrence network using NetworkX and Pyvis.
        It creates a graph from the co-occurrence counts and displays it HTML interactively.
    '''
    
    # Step 1: Create the NetworkX Graph
    G = nx.Graph()

    # Step 2: Add edges with weights
    for (word1, word2), weight in co_occurrence_counter.most_common(150):
        G.add_edge(word1, word2, weight=1)

    # Step 3: Create a Pyvis network
    net = Network(
        notebook=True,            # Enable notebook mode for Jupyter
        height="1000px",          # Set the height of the network window
        width="2500px",          # Set the width of the network window
        bgcolor="#FFFFFF",     # Set the background color to white
        font_color="black",        # Set the font color for node labels
        select_menu= True,
    )

    # Step 4: Load the NetworkX graph into Pyvis
    net.from_nx(G)             

    # Optional: Set physics for better layout
    net.force_atlas_2based()     # Use the Force Atlas 2 layout algorithm for better node spacing

    # Step 5: Show and save the interactive network as HTML
    net.show("Index.html")  # Save and open the interactive network in your browser
    print("Co-occurrence network generated successfully!")

if __name__ == "__main__":

    # 1. Data Ingestion and NLP Pipeline Setup.
    df = ingest_data_and_nlp_pipeline_setup('crunchbase.csv')

    # Create set of NER (ORG, PRODUCT, EVENT)
    ents_of_interest = set()
    for i in range(0,len(df['spacy_doc'])):
        for ent in df['spacy_doc'][i].ents:
            # Only keep entities with the desired labels
            if ent.label_ in {'ORG', 'PRODUCT', 'EVENT'}:
                # Add the entity text (lowercased) to the set
                ents_of_interest.add(ent.text.lower())

    # 2. Full Text Preprocessing
    df['tokens_Filtered'] = df['spacy_doc'].apply(get_tokens_no_stop)

    # 3. Determine the frequency of pair-words
    co_occurrence_counter = pair_words_frequency(df)

    # 3.1. Generate a CSV that have pair words with more than 100 counts
    generate_pair_words_csv(co_occurrence_counter)

    # 3.2. Generate a CSV that contains Single Word Frequency 
    single_word_frequency_csv(df)

    # 4. Visualize the co-occurrence network
    visualize_co_occurrence_network(co_occurrence_counter)