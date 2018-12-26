#!/usr/bin/env python2
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk, re, sys, glob, string, random, io

reload(sys)     
sys.setdefaultencoding('utf8')

lemmatizer = WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')

# method preprocesses sentences, lemmatizing words and removing
def preprocess_sentence(words):
    words = [word.lower() for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in stop_words]
    # remove words that are only punctuation symbols after tokenizing by word
    words = [word for word in words if word not in string.punctuation and word != "''" and word != '``']
    return words

# method loads sentences from cluster files into a list of strings
def load_sentences(cluster_paths):
    cluster_sentences = []
    for doc_path in cluster_paths:
        with io.open(doc_path,encoding='utf-8-sig') as file:
            cluster_sentences.extend(nltk.sent_tokenize(file.read().replace('\xe2\x80\x9c', '\"').replace('\xe2\x80\x9d', '\"').replace('\xe2\x80\x99', '\'')))

    return cluster_sentences

# method calculates probability from each word given the document cluster paths
def calculate_word_probabilities(cluster_paths):
    word_probabilities = {}
    word_counter = 0.0
    for cluster_path in cluster_paths:
        with io.open(cluster_path, encoding='utf-8-sig') as file:
            words = preprocess_sentence(word_tokenize(file.read().replace('\xe2\x80\x9c', '\"').replace('\xe2\x80\x9d', '\"').replace('\xe2\x80\x99', '\'').replace('\xe2\x80\x94', '-')))
            word_counter += len(words)

            for word in words:
                if word not in word_probabilities:
                    word_probabilities[word] = 1.0
                else:
                    word_probabilities[word] += 1.0
    # divide each dictionary count by total frequency of words                
    for word_probability in word_probabilities:
        word_probabilities[word_probability] = word_probabilities[word_probability]/float(word_counter)

    return word_probabilities

# method calculates the weight score of a sentence given the sentence and the word probability dictionary
def calculate_sentence_score(sentence, word_probabilities):
    score = 0.0
    word_counter = 0.0
    words = word_tokenize(sentence)
    words = preprocess_sentence(words)
    for word in words:
        if word in word_probabilities:
            score += word_probabilities[word]
            word_counter += 1
    
    return float(score)/float(word_counter)

# method updates the word probability dictionary given the obtained best sentence and the dictionary itself
def update_word_probabilities(best_sentence, word_probabilities):
    words = word_tokenize(best_sentence)
    words = preprocess_sentence(words)
    for word in words:
        if word in word_probabilities:
            word_probabilities[word] = word_probabilities[word] ** 2

# method finds the word with the highest probability from the word probability dictionary
def find_max_probability_word(word_probabilities):
    max_word_probability = float('-inf')
    max_probability_word = None

    for word in word_probabilities:
        # if word in words_probabilities:
        if word_probabilities[word] > max_word_probability:
            max_probability_word = word
            max_word_probability = word_probabilities[word]

    return max_probability_word

# method finds the best scoring sentence that contains the highest probability word, only used for
# orig and simplified methods
def find_max_sentence(sentences, word_probabilities, is_simplified):
    max_sentences = []
    max_score = float('-inf')
    best_sentence = None
    max_probability_word = find_max_probability_word(word_probabilities)

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = preprocess_sentence(words)

        if max_probability_word in words:
            max_sentences.append(sentence)
        
    for max_sentence in max_sentences:
        current_score = calculate_sentence_score(max_sentence, word_probabilities)

        if current_score > max_score:
            best_sentence = max_sentence
            max_score = current_score

    if not is_simplified:
        update_word_probabilities(best_sentence, word_probabilities)
    
    sentences[:] = [sentence for sentence in sentences if sentence != best_sentence]


    return best_sentence

# method finds the sentence with the highest average word probability
def find_max_sentence_best_avg(sentences, word_probabilities):
    max_score = float('-inf')
    best_sentence = None

    for sentence in sentences:
        current_score = calculate_sentence_score(sentence, word_probabilities)

        if current_score > max_score:
            best_sentence = sentence
            max_score = current_score

    update_word_probabilities(best_sentence, word_probabilities)

    sentences[:] = [sentence for sentence in sentences if sentence != best_sentence]

    return best_sentence

# executes the specified method on the specified cluster given the command line arguments
def main():
    
    method_name = sys.argv[1]
    doc_cluster = sys.argv[2]
    # used to random select article in event of leading method
    articles_per_cluster = 3
    cluster_number = int(doc_cluster[10])
    cluster_paths = glob.glob(doc_cluster)
    word_probabilities = calculate_word_probabilities(cluster_paths)
    sentences = load_sentences(cluster_paths)
    summary = []
    word_counter = 0

    if method_name == 'orig':
        while(word_counter < 100):
            temp_sentence = find_max_sentence(sentences, word_probabilities, False)
            words = temp_sentence.split(' ')
            if word_counter + len(words) < 100:
                summary.append(temp_sentence)
                word_counter += len(words)
            else:
                remaining_words = 100 - word_counter
                summary.append(" ".join(words[:remaining_words]))
                break
    elif method_name == 'best-avg':
        while(word_counter < 100):
            temp_sentence = find_max_sentence_best_avg(sentences, word_probabilities)
            words = temp_sentence.split(' ')
            if word_counter + len(words) < 100:
                summary.append(temp_sentence)
                word_counter += len(words)
            else:
                remaining_words = 100 - word_counter
                summary.append(" ".join(words[:remaining_words]))
                break
    elif method_name == 'simplified':
        while(word_counter < 100):
            temp_sentence = find_max_sentence(sentences, word_probabilities, True)
            words = temp_sentence.split(' ')
            if word_counter + len(words) < 100:
                summary.append(temp_sentence)
                word_counter += len(words)
            else:
                remaining_words = 100 - word_counter
                summary.append(" ".join(words[:remaining_words]))
                break
    elif method_name == 'leading':
        # randomly selected article from cluster
        doc_number = random.randint(1,articles_per_cluster)
        cluster_path = []
        cluster_path.append('./docs\\doc' + str(cluster_number) + '-' + str(doc_number) + '.txt')
        sentences = load_sentences(cluster_path)
        sentence_counter = 0
        while(word_counter < 100):
            temp_sentence = sentences[sentence_counter]
            words = temp_sentence.split(' ')
            if word_counter + len(words) < 100:
                summary.append(temp_sentence)
                word_counter += len(words)
                sentence_counter += 1
            else:
                remaining_words = 100 - word_counter
                summary.append(" ".join(words[:remaining_words]))
                break
    print " ".join(summary)

if __name__ == '__main__':
	main()