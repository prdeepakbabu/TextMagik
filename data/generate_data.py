# process_movies.py

import os
from datasets import load_dataset

def process_movies(output_file="movies.txt"):
    """
    Loads the 'MongoDB/embedded_movies' dataset from Hugging Face,
    then writes each movie's genre, title, plot, and fullplot
    to a single text file as 'genre,title,plot,fullplot'.
    """
    # Load the dataset (assuming a single split 'train')
    dataset = load_dataset("MongoDB/embedded_movies", split="train")

    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        for record in dataset:
            # Because some records may have multiple genres, choose the first one or provide fallback
            genres = record.get("genres", [])
            genre = genres[0] if genres else "Unknown"

            title = record.get("title", "No Title")
            plot = record.get("plot", "No Plot")
            fullplot = record.get("fullplot", "No Fullplot")

            # Format each line as CSV-like (with commas)
            line = f"{genre},{title},{plot},{fullplot}\n"
            f.write(line)

    print(f"All movies written to '{output_file}' in 'genre,title,plot,fullplot' format.")

if __name__ == "__main__":
    # Run the processing function
    process_movies(output_file="movies.txt")