import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Sample feedback from users about a service they received
user_feedback = [
    "The customer service was excellent. The staff was friendly and helpful throughout the process.",
    "I am disappointed with the product quality. It did not meet my expectations.",
    "The delivery was fast, and the package arrived in perfect condition. I am satisfied with the service.",
    "The website interface is user-friendly, making it easy to navigate and find what I needed.",
    "I had a terrible experience with the support team. They were unresponsive and did not resolve my issue.",
    "The pricing is reasonable, and the discounts offered are attractive.",
    "The service exceeded my expectations. I would highly recommend it to others.",
    "The product was delivered late, causing inconvenience. This needs improvement.",
    "I encountered technical difficulties while using the app. It needs bug fixing.",
    "The overall experience was average. There is room for improvement in various aspects."
]

# Text preprocessing
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

processed_user_feedback = [preprocess_text(feedback) for feedback in user_feedback]

# Create a dictionary and document-term matrix
dictionary = corpora.Dictionary(processed_user_feedback)
corpus = [dictionary.doc2bow(text) for text in processed_user_feedback]

# Perform LDA
num_topics = 3  # Number of topics to discover
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Display the topics and associated keywords
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx + 1}: {topic}")

# Assign the feedback to their respective topics
topic_assignments = [max(lda_model[text], key=lambda x: x[1])[0] for text in corpus]

# Organize feedback into topic clusters
clusters = {}
for i, topic_id in enumerate(topic_assignments):
    if topic_id not in clusters:
        clusters[topic_id] = []
    clusters[topic_id].append(user_feedback[i])

# Display the clustered feedback
for cluster_id, feedbacks_in_cluster in clusters.items():
    print(f"\nCluster {cluster_id + 1}:")
    for feedback in feedbacks_in_cluster:
        print(f"{feedback}\n")
