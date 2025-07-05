# Minimal test code
docs = [
    "The cat sat on the mat.",
    "Dogs are wonderful pets.",
    "Cats and dogs are friendly.",
    "Python is a great programming language.",
    "Artificial intelligence is evolving rapidly."
]

topics, probs = get_topics(docs)
print("Topics:", topics)
print("Probs:", probs)

keywords = extract_keywords()
print("Extracted keywords per topic:", keywords)
