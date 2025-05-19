import pyconll
import collections
import sys # Used for exiting if file not found

# Demonstrate how to use the pyconll library to load a CoNLL-U file


def count_upos_tags_from_conllu(filepath):
    """
    Loads a CoNLL-U file and counts the occurrences of each Universal POS tag.

    Args:
        filepath (str): The path to the CoNLL-U file.

    Returns:
        collections.Counter: A Counter object where keys are UPOS tags (str)
                             and values are their counts (int).
                             Returns None if the file cannot be loaded.
    """
    pos_counts = collections.Counter()

    try:
        # Load the CoNLL-U file
        # load_from_file returns an iterable Conll object
        conll_data = pyconll.load_from_file(filepath)

        # Iterate through each sentence in the data
        for sentence in conll_data:
            # print(sentence.text) # Print the sentence form for debugging
            # Iterate through each token in the sentence
            for token in sentence:
                # The UPOS tag is stored in the 'upos' attribute
                # We should check if it's not None or empty,
                # as some lines (like multi-word tokens) might not have it.
                if token.upos:
                    pos_counts[token.upos] += 1

        return pos_counts

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        # Catch other potential errors during parsing
        print(f"An error occurred while processing the file: {e}")
        return None

# --- Main execution part ---
if __name__ == "__main__":
    conllu_file_path = '../UD_English-EWT/en_ewt-ud-dev.conllu'

    print(f"Processing file: {conllu_file_path}")
    tag_counts = count_upos_tags_from_conllu(conllu_file_path)

    if tag_counts is not None:
        print("\nUniversal POS Tag Counts:")
        # Sort tags alphabetically for consistent output
        sorted_tags = sorted(tag_counts.items())
        if sorted_tags:
            for tag, count in sorted_tags:
                print(f"- {tag}: {count}")
        else:
            print("No POS tags found or counted in the file.")
    else:
        # Error message already printed in the function
        sys.exit(1) # Exit with an error code