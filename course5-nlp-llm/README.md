# Course 5: NLP & Large Language Models

> **From Classical NLP Foundations to Modern LLMs: A Comprehensive Learning Roadmap**

---

## Learning Path Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    NLP FOUNDATIONS → MODERN LLMs ROADMAP                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   🎯 Foundation        📝 Lexical         🌳 Syntactic        💡 Semantic        │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐      │
│   │  Math &  │   ──►  │  Word    │   ──►  │ Grammar  │   ──►  │ Meaning  │      │
│   │  Python  │        │Processing│        │Structure │        │Extraction│      │
│   └──────────┘        └──────────┘        └──────────┘        └──────────┘      │
│                                                                                  │
│        │                                                            │            │
│        ▼                                                            ▼            │
│   🧠 Deep Learning     ⚡ Transformers     🚀 LLMs           🌟 Applications     │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐      │
│   │RNN/LSTM  │   ──►  │Attention │   ──►  │ GPT/BERT │   ──►  │ RAG/Agents│     │
│   │Seq2Seq   │        │ Is All   │        │ Alignment│        │ ChatBots │      │
│   └──────────┘        └──────────┘        └──────────┘        └──────────┘      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Lexical Processing (Word-Level)](#1-lexical-processing-word-level)
2. [Syntactic Processing (Structure/Grammar)](#2-syntactic-processing-structuregrammar)
3. [Semantic Processing (Meaning)](#3-semantic-processing-meaning)
4. [Pragmatic/Discourse Processing](#4-pragmaticdiscourse-processing)
5. [Deep Learning for NLP](#5-deep-learning-for-nlp)
6. [Transformer Architecture](#6-transformer-architecture)
7. [Large Language Models (LLMs)](#7-large-language-models-llms)
8. [Applications & Practice Projects](#8-applications--practice-projects)

---

## 1. Lexical Processing (Word-Level)

> **What it is:** Breaking text into tokens and understanding individual word properties.

### 1.0 Text Analytics Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT ANALYTICS PROCESSING STACK               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Lexical           Syntactic         Semantic        Pragmatic │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐   ┌────────┐│
│   │ Word-    │  ──► │ Grammar  │  ──► │ Meaning  │──►│Context ││
│   │ Level    │      │ Structure│      │ Extract  │   │ Intent ││
│   └──────────┘      └──────────┘      └──────────┘   └────────┘│
│   • Tokens          • PoS Tags        • Entities     • Sentiment│
│   • Stems           • Parse Trees     • Relations    • Discourse│
│   • TF-IDF          • Dependencies    • WSD          • QA       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Text Encoding

Before processing text, understand character encoding:

| Encoding | Description | Character Set |
|----------|-------------|---------------|
| **ASCII** | 7-bit encoding | 128 characters (English) |
| **Unicode (UTF-8)** | Variable-width (1-4 bytes) | All languages, emoji |
| **UTF-16** | 2 or 4 bytes per character | Java, Windows internals |
| **UTF-32** | Fixed 4 bytes | Memory-intensive but simple |

```python
# Text encoding examples
text = "Hello 你好"
print(text.encode('utf-8'))    # b'Hello \xe4\xbd\xa0\xe5\xa5\xbd'
print(text.encode('ascii', errors='ignore'))  # b'Hello '
```

### 1.2 Regular Expressions

Essential tool for pattern matching in text.

**Special Characters:**

| Char | Meaning | Example |
|------|---------|---------|
| `.` | Any character | `a.c` matches "abc", "a1c" |
| `^` | Start of string | `^Hello` |
| `$` | End of string | `world$` |
| `*` | 0 or more | `ab*` matches "a", "ab", "abb" |
| `+` | 1 or more | `ab+` matches "ab", "abb" |
| `?` | 0 or 1 | `colou?r` matches "color", "colour" |
| `[]` | Character class | `[aeiou]` matches any vowel |
| `\d` | Digit [0-9] | `\d{3}` matches "123" |
| `\w` | Word char [a-zA-Z0-9_] | `\w+` matches "hello_123" |
| `\s` | Whitespace | `\s+` matches spaces/tabs |

```python
import re

# Email validation
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
print(re.match(email_pattern, "user@example.com"))  # Match

# Extract phone numbers
text = "Call me at 123-456-7890 or 987-654-3210"
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)
# ['123-456-7890', '987-654-3210']

# Replace patterns
text = "Price: $100.50"
cleaned = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
# 'Price 10050'
```

### 1.3 Word Frequencies and Zipf's Law

**Zipf's Law:** The frequency of any word is inversely proportional to its rank.
$$f(r) \propto \frac{1}{r^\alpha}$$

Where $r$ is rank and $\alpha \approx 1$ for natural language.

```python
from collections import Counter
import matplotlib.pyplot as plt

# Analyze word frequencies
text = "the quick brown fox jumps over the lazy dog the fox is quick"
words = text.split()
freq = Counter(words)

# Most common words
print(freq.most_common(3))  # [('the', 3), ('quick', 2), ('fox', 2)]

# Visualize Zipf distribution
ranks = range(1, len(freq) + 1)
frequencies = sorted(freq.values(), reverse=True)
plt.loglog(ranks, frequencies)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf's Law")
```

### 1.4 Stop Words Removal

Remove high-frequency, low-information words.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is an example showing stop word removal"
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(text.lower())
filtered = [w for w in tokens if w not in stop_words]
print(filtered)  # ['example', 'showing', 'stop', 'word', 'removal']
```

### 1.5 Tokenization

Splitting text into words, subwords, or characters.

**Classical Tokenization:**

| Tokenizer Type | Description | Use Case |
|----------------|-------------|----------|
| **Word Tokenizer** | Split on whitespace/punctuation | Basic NLP |
| **Sentence Tokenizer** | Split into sentences | Document segmentation |
| **Tweet Tokenizer** | Handles @mentions, #hashtags | Social media |
| **Regexp Tokenizer** | Custom patterns | Domain-specific |

```python
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer

# Word tokenization
text = "Hello! This is a test."
tokens = word_tokenize(text)
# ['Hello', '!', 'This', 'is', 'a', 'test', '.']

# Sentence tokenization
doc = "Dr. Smith went to the U.S.A. He visited N.Y.C."
sentences = sent_tokenize(doc)
# ['Dr. Smith went to the U.S.A.', 'He visited N.Y.C.']

# Tweet tokenization
tweet_tokenizer = TweetTokenizer()
tweet = "Amazing product! @company #awesome https://example.com"
tokens = tweet_tokenizer.tokenize(tweet)
# ['Amazing', 'product', '!', '@company', '#awesome', 'https://example.com']
```

**Modern Subword Tokenization:**

| Tokenizer Type | Description | Example |
|----------------|-------------|---------|
| **BPE** (Byte Pair Encoding) | Merge frequent character pairs | Used by GPT, LLaMA |
| **WordPiece** | Similar to BPE, likelihood-based | Used by BERT |
| **SentencePiece** | Language-agnostic, treats text as raw stream | Used by T5, mT5 |
| **Unigram** | Probabilistic subword model | Used by XLNet |

**Implementation:**
```python
# Hugging Face Tokenizers
from transformers import AutoTokenizer

# BERT tokenizer (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = bert_tokenizer.tokenize("unhappiness")
# Output: ['un', '##happiness']

# GPT-2 tokenizer (BPE)
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokens = gpt_tokenizer.tokenize("unhappiness")
# Output: ['unh', 'app', 'iness']

# Custom BPE training
from tokenizers import Tokenizer, models, trainers
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

### 1.6 Text Normalization: Stemming vs Lemmatization

**Stemming:** Crude chopping of word endings using rules.

| Technique | Description | Example |
|-----------|-------------|---------|
| **Porter Stemmer** | Most common, aggressive | "running" → "run", "ponies" → "poni" |
| **Snowball Stemmer** | Multilingual, improved Porter | "running" → "run", "generously" → "generous" |
| **Lancaster Stemmer** | Very aggressive | Can over-stem words |

**Lemmatization:** Dictionary-based reduction to root form.

| Technique | Description | Example |
|-----------|-------------|---------|
| **WordNet Lemmatizer** | Uses WordNet dictionary | "better" → "good", "running" → "run" |
| **spaCy Lemmatizer** | Rule-based + lookup tables | Context-aware lemmatization |

```python
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Stemming
porter = PorterStemmer()
snowball = SnowballStemmer('english')

words = ["running", "runs", "ran", "runner", "easily", "fairly"]
print([porter.stem(w) for w in words])
# ['run', 'run', 'ran', 'runner', 'easili', 'fairli']

print([snowball.stem(w) for w in words])
# ['run', 'run', 'ran', 'runner', 'easili', 'fair']

# Lemmatization (POS-aware)
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos='a'))  # good (adjective)
print(lemmatizer.lemmatize("running", pos='v'))  # run (verb)
print(lemmatizer.lemmatize("geese"))  # goose
```

**When to use what:**
- **Stemming:** Fast, good for search/IR, doesn't need POS tags
- **Lemmatization:** Accurate, preserves meaning, needs POS tags, slower

### 1.7 Bag-of-Words (BoW) Model

Represent documents as vectors of word counts (ignoring order).

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love deep learning"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
# ['amazing' 'deep' 'is' 'learning' 'love' 'machine']

print(bow_matrix.toarray())
# [[0 0 0 1 1 1]   # Doc 1
#  [1 0 1 1 0 1]   # Doc 2
#  [0 1 0 1 1 0]]  # Doc 3
```

**Limitations:**
- Loses word order ("dog bites man" = "man bites dog")
- No semantic understanding
- High dimensionality with large vocabulary

### 1.8 TF-IDF (Term Frequency-Inverse Document Frequency)

Weigh words by importance: frequent in document, rare across corpus.

**Formula:**
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Where:
- $\text{TF}(t, d) = \frac{\text{count of term } t \text{ in doc } d}{\text{total terms in } d}$
- $\text{IDF}(t) = \log\frac{\text{total documents}}{\text{documents containing } t}$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are enemies"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())

# Words appearing in all documents (like "the") get lower scores
# Unique words (like "enemies") get higher scores
```

**Application: Spam Detection with Naive Bayes**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data
emails = [
    "Win free money now click here",
    "Meeting at 3pm tomorrow",
    "Claim your prize immediately",
    "Project deadline next week"
]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Predict
test_email = ["Free prize win now"]
X_test = vectorizer.transform(test_email)
print(clf.predict(X_test))  # [1] (spam)
```

### 1.9 Canonicalization Techniques

Transform words to standard forms for better matching.

#### 1.9.1 Phonetic Hashing: Soundex Algorithm

Map similar-sounding words to same code.

**Soundex Steps:**
1. Keep first letter
2. Replace consonants: B,F,P,V→1, C,G,J,K,Q,S,X,Z→2, D,T→3, L→4, M,N→5, R→6
3. Remove vowels (A,E,I,O,U,H,W,Y)
4. Remove duplicates
5. Pad/truncate to 4 characters

```python
import phonetics

# Soundex encoding
print(phonetics.soundex("Smith"))    # S530
print(phonetics.soundex("Smythe"))   # S530 (same!)
print(phonetics.soundex("Johnson"))  # J525
print(phonetics.soundex("Jonson"))   # J525 (same!)

# Useful for matching names despite spelling variations
```

#### 1.9.2 Edit Distance (Levenshtein Distance)

Minimum edits (insert/delete/substitute) to transform one string to another.

**Dynamic Programming Algorithm:**

```python
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # delete
                                   dp[i][j-1],    # insert
                                   dp[i-1][j-1])  # substitute

    return dp[m][n]

print(levenshtein_distance("kitten", "sitting"))  # 3
# k→s, e→i, +g

# Using library
from Levenshtein import distance
print(distance("kitten", "sitting"))  # 3
```

**Damerau-Levenshtein Distance:** Also allows transposition (swap adjacent chars).

#### 1.9.3 Norvig's Spell Corrector

Statistical approach using edit distance + word probabilities.

```python
import re
from collections import Counter

def words(text):
    return re.findall(r'\w+', text.lower())

# Build word frequency dictionary from corpus
WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of word"
    return WORDS[word] / N

def candidates(word):
    "Generate possible spelling corrections"
    return (known([word]) or
            known(edits1(word)) or
            known(edits2(word)) or
            [word])

def known(words):
    "Filter to words that appear in dictionary"
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits 1 edit away"
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits 2 edits away"
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correction(word):
    "Most probable spelling correction"
    return max(candidates(word), key=P)

print(correction("speling"))  # spelling
print(correction("korrecter"))  # corrector
```

### 1.10 Pointwise Mutual Information (PMI)

Measure association between words (collocation).

**Formula:**
$$\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) \cdot P(y)}$$

- PMI > 0: Words co-occur more than expected
- PMI = 0: Independent
- PMI < 0: Words avoid each other

```python
import math
from collections import Counter

corpus = "new york times new york city los angeles times".split()
bigrams = list(zip(corpus[:-1], corpus[1:]))

# Count frequencies
word_freq = Counter(corpus)
bigram_freq = Counter(bigrams)
total_words = len(corpus)
total_bigrams = len(bigrams)

def pmi(word1, word2):
    p_xy = bigram_freq[(word1, word2)] / total_bigrams
    p_x = word_freq[word1] / total_words
    p_y = word_freq[word2] / total_words

    if p_xy == 0:
        return 0
    return math.log2(p_xy / (p_x * p_y))

print(pmi("new", "york"))  # High positive PMI (strong association)
print(pmi("new", "angeles"))  # Lower or negative PMI
```

### 1.11 N-gram Models

Model probability of word sequences.

**Bigram Approximation:**
$$P(w_1, w_2, ..., w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-1})$$

```python
from nltk import bigrams, trigrams
from nltk.probability import ConditionalFreqDist

text = "the cat sat on the mat the cat was fat".split()
bg = list(bigrams(text))

# Conditional frequency distribution
cfd = ConditionalFreqDist(bg)

# Probability of word given previous word
print(list(cfd["the"].items()))  # [('cat', 2), ('mat', 1)]
print(cfd["the"].freq("cat"))  # P(cat | the) = 2/3
```

### 1.3 Word Representations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WORD REPRESENTATION EVOLUTION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   One-Hot          Sparse Embeddings       Dense Embeddings      Contextual │
│   ┌─────┐          ┌─────────────┐         ┌─────────────┐      ┌─────────┐ │
│   │1 0 0│   ──►    │  TF-IDF     │   ──►   │  Word2Vec   │ ──►  │  BERT   │ │
│   │0 1 0│          │  (sparse)   │         │  GloVe      │      │  GPT    │ │
│   │0 0 1│          │             │         │  FastText   │      │         │ │
│   └─────┘          └─────────────┘         └─────────────┘      └─────────┘ │
│   |V| dims         |V| dims sparse         50-300 dims          768+ dims   │
│   No semantics     Frequency-based         Static similarity    Dynamic!    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Word2Vec

**Skip-gram:** Predict context words from center word
$$P(w_{context}|w_{center}) = \frac{e^{v_{context} \cdot v_{center}}}{\sum_{w \in V} e^{v_w \cdot v_{center}}}$$

**CBOW (Continuous Bag of Words):** Predict center word from context
$$P(w_{center}|w_{context}) = \frac{e^{v_{center} \cdot \bar{v}_{context}}}{\sum_{w \in V} e^{v_w \cdot \bar{v}_{context}}}$$

```python
from gensim.models import Word2Vec

# Train Word2Vec
sentences = [["machine", "learning", "is", "fun"],
             ["deep", "learning", "requires", "data"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1 for skip-gram

# Find similar words
model.wv.most_similar("learning")

# Word arithmetic
# king - man + woman ≈ queen
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

#### GloVe (Global Vectors)

Combines global matrix factorization with local context window:
$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $X_{ij}$ is co-occurrence count of words $i$ and $j$.

```python
# Load pre-trained GloVe
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")  # 100-dim vectors
glove.most_similar("computer")
```

#### FastText

Extends Word2Vec with subword information (handles OOV words):
$$v_{word} = \sum_{g \in G_{word}} z_g$$

Where $G_{word}$ is the set of n-grams in the word.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1)
# Can generate embeddings for OOV words!
model.wv["unseen_word_xyz"]  # Works due to subword info
```

### 1.4 Classical → Modern Mapping

| Classical | Modern Equivalent |
|-----------|-------------------|
| Word2Vec/GloVe | Learned Token Embeddings in Transformer |
| BPE Tokenization | GPT/LLaMA tokenizers |
| WordPiece | BERT tokenizer |
| Static embeddings | **Contextual embeddings** (dynamic per context) |

---

## 2. Syntactic Processing (Structure/Grammar)

> **What it is:** Understanding grammatical structure and relationships between words.

### 2.0 What is Syntax?

**Syntax:** Set of rules governing sentence structure in a language.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTACTIC PROCESSING LAYERS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PoS Tagging       Parsing           Relations                 │
│   ┌──────────┐      ┌──────────┐      ┌──────────────────┐      │
│   │ Label    │  ──► │ Tree     │  ──► │ Named Entity     │      │
│   │ Word     │      │ Structure│      │ Recognition      │      │
│   │ Classes  │      │          │      │                  │      │
│   └──────────┘      └──────────┘      └──────────────────┘      │
│   NN, VB, JJ        Parse Trees        PER, ORG, LOC            │
│                     Dependencies                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Part-of-Speech (POS) Tagging

Assigning grammatical tags (noun, verb, adjective, etc.) to each word.

#### 2.1.1 POS Tag Categories

**Open Class (Content Words):** New words can be added

| Class | Description | Examples |
|-------|-------------|----------|
| **Noun** | Person, place, thing | dog, Paris, happiness |
| **Verb** | Action, state | run, think, is |
| **Adjective** | Describes noun | quick, beautiful, tall |
| **Adverb** | Describes verb/adj | quickly, very, always |
| **Interjection** | Exclamation | wow, ouch, hello |

**Closed Class (Function Words):** Fixed set

| Class | Description | Examples |
|-------|-------------|----------|
| **Preposition** | Relation to nouns | in, on, at, by |
| **Pronoun** | Replaces noun | he, she, it, they |
| **Conjunction** | Connects words/clauses | and, but, or, because |
| **Article** | Defines noun | a, an, the |
| **Determiner** | Specifies noun | this, that, some, many |
| **Numeral** | Numbers | one, two, first, second |

#### 2.1.2 Penn Treebank POS Tags

| Tag | Description | Example |
|-----|-------------|---------|
| **NN** | Noun, singular | "dog" |
| **NNS** | Noun, plural | "dogs" |
| **NNP** | Proper noun, singular | "London" |
| **VB** | Verb, base form | "run" |
| **VBD** | Verb, past tense | "ran" |
| **VBG** | Verb, gerund | "running" |
| **VBZ** | Verb, 3rd person sing | "runs" |
| **JJ** | Adjective | "quick" |
| **JJR** | Adjective, comparative | "quicker" |
| **RB** | Adverb | "quickly" |
| **PRP** | Personal pronoun | "I", "you", "he" |
| **DT** | Determiner | "the", "a" |

#### 2.1.3 PoS Tagging with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Sumit is an adjunct faculty at UpGrad.")
for token in doc:
    print(f"{token.text:10} {token.pos_:6} {token.tag_:4} {spacy.explain(token.tag_)}")

# Output:
# Sumit      PROPN  NNP  noun, proper singular
# is         AUX    VBZ  verb, 3rd person singular present
# an         DET    DT   determiner
# adjunct    ADJ    JJ   adjective
# faculty    NOUN   NN   noun, singular or mass
# at         ADP    IN   preposition or subordinating conjunction
# UpGrad     PROPN  NNP  noun, proper singular
# .          PUNCT  .    punctuation mark, sentence closer
```

### 2.2 PoS Tagging Techniques

#### 2.2.1 Rule-Based Tagging

Use hand-crafted rules based on word endings, context, etc.

```python
def rule_based_tagger(word):
    if word.endswith('ing'):
        return 'VBG'  # Gerund
    elif word.endswith('ed'):
        return 'VBD'  # Past tense
    elif word.endswith('ly'):
        return 'RB'   # Adverb
    elif word[0].isupper():
        return 'NNP'  # Proper noun
    else:
        return 'NN'   # Default: noun

print(rule_based_tagger("running"))  # VBG
print(rule_based_tagger("quickly"))  # RB
```

**Limitations:** Can't handle ambiguity, requires extensive manual rules

#### 2.2.2 Hidden Markov Model (HMM) for PoS Tagging

Model tag sequences as a Markov chain with hidden states (tags) and observations (words).

```
┌─────────────────────────────────────────────────────────────────┐
│                    HMM FOR POS TAGGING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Hidden States (Tags):   DT  ──► JJ  ──► NN  ──► VB           │
│                           │      │      │      │                │
│                           │      │      │      │                │
│   Observations (Words):   the    quick  fox   jumps             │
│                                                                  │
│   • Transition Prob: P(tag_i | tag_{i-1})                       │
│   • Emission Prob:   P(word_i | tag_i)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Transition Matrix:** $P(\text{tag}_i | \text{tag}_{i-1})$
   - Probability of tag given previous tag
   - Example: P(NN | DT) = high (determiner often followed by noun)

2. **Emission Matrix:** $P(\text{word}_i | \text{tag}_i)$
   - Probability of word given tag
   - Example: P("dog" | NN) = high

**Viterbi Algorithm:** Find most likely tag sequence

$$\text{Best tag sequence} = \arg\max_{\text{tags}} P(\text{tags} | \text{words})$$

```python
import nltk
from nltk.tag import hmm

# Train HMM tagger
train_data = nltk.corpus.treebank.tagged_sents()[:3000]
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

# Tag new sentence
sentence = ["The", "quick", "brown", "fox"]
print(tagger.tag(sentence))
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN')]
```

### 2.3 Parsing

#### 2.3.1 Constituency Parsing

Hierarchical tree structure (phrase structure grammar) - breaks sentences into nested constituents.

```
                    S
           ┌───────┴───────┐
           NP              VP
    ┌──────┼─────┐    ┌────┴────┐
   DET    ADJ   NOUN  V        PP
    │      │     │    │    ┌───┴───┐
   The   quick  fox jumps PREP    NP
                           │   ┌──┴──┐
                         over DET  NOUN
                               │    │
                              the  dog
```

**Phrase Types:**
- **NP** (Noun Phrase): "the quick brown fox"
- **VP** (Verb Phrase): "jumps over the lazy dog"
- **PP** (Prepositional Phrase): "over the lazy dog"
- **S** (Sentence): Complete sentence

#### 2.3.2 Dependency Parsing

Directed graph showing head-modifier relationships - each word depends on another word (its "head").

```
        ┌─────────nsubj─────────┐
        │                       │
        │    ┌──amod──┐  ┌─amod─┤
        ▼    ▼        │  ▼      │
       The quick brown fox jumps over the lazy dog
                           │          │         │
                           └───prep───┘         │
                                    └───pobj────┘
```

**Key Dependency Relations:**

| Relation | Description | Example |
|----------|-------------|---------|
| **nsubj** | Nominal subject | "Fox" is subject of "jumps" |
| **nsubjpass** | Passive nominal subject | "Ball was kicked" |
| **dobj** | Direct object | "I eat *apples*" |
| **amod** | Adjectival modifier | "*quick* fox" |
| **prep** | Prepositional modifier | "jumps *over*" |
| **pobj** | Object of preposition | "over the *dog*" |
| **aux** | Auxiliary verb | "is *running*" |
| **det** | Determiner | "*the* dog" |

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The quick brown fox jumps over the lazy dog")

# Print dependencies
for token in doc:
    print(f"{token.text:10} --{token.dep_:10}--> {token.head.text:10}")

# Output:
# The        --det--------> fox
# quick      --amod-------> fox
# brown      --amod-------> fox
# fox        --nsubj------> jumps
# jumps      --ROOT-------> jumps
# over       --prep-------> jumps
# the        --det--------> dog
# lazy       --amod-------> dog
# dog        --pobj-------> over

# Visualize dependency tree
from spacy import displacy
displacy.render(doc, style="dep", jupyter=False)
# In Jupyter: displacy.render(doc, style="dep")
```

**Practical Applications:**

1. **Passive Voice Detection:**

```python
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern: passive nominal subject
passive_rule = [{'DEP': 'nsubjpass'}]
matcher.add('PassiveVoice', [passive_rule])

doc = nlp("The ball was kicked by John.")
matches = matcher(doc)

if matches:
    print("Passive voice detected!")
# Output: Passive voice detected!
```

2. **Grammar Checking (Grammarly-style):**

```python
# Detect subject-verb agreement errors
doc = nlp("The dogs runs fast")
for token in doc:
    if token.dep_ == 'nsubj':
        subject = token
        verb = token.head
        # Check plurality mismatch
        if subject.tag_ == 'NNS' and verb.tag_ == 'VBZ':
            print(f"Error: Plural subject '{subject}' with singular verb '{verb}'")
```

3. **Heteronyms Identification:**

Words spelled same but different pronunciation/meaning based on PoS.

```python
# "lead" (noun) vs "lead" (verb)
# "read" (present) vs "read" (past)

doc1 = nlp("I lead the team")       # lead (verb) /liːd/
doc2 = nlp("This is a lead pipe")   # lead (noun) /lɛd/

for doc in [doc1, doc2]:
    for token in doc:
        if token.text == "lead":
            print(f"{token.text}: {token.pos_}")
# lead: VERB
# lead: NOUN
```

### 2.4 Named Entity Recognition (NER)

Identify and classify named entities in text.

#### 2.4.1 Entity Types

**Common Entity Types (OntoNotes 5.0):**

| Tag | Entity Type | Examples |
|-----|-------------|----------|
| **PERSON** / **PER** | People | "Elon Musk", "Marie Curie" |
| **ORG** | Organizations | "Google", "United Nations" |
| **GPE** | Geo-Political Entity | "Paris", "California" |
| **LOC** | Location | "Mount Everest", "Amazon River" |
| **DATE** | Date | "January 15", "tomorrow", "2024" |
| **TIME** | Time | "3:00 PM", "morning" |
| **MONEY** | Monetary values | "$100", "€50" |
| **PERCENT** | Percentages | "20%", "one-third" |
| **PRODUCT** | Products | "iPhone", "Windows 11" |
| **EVENT** | Events | "World War II", "Olympics" |
| **WORK_OF_ART** | Artworks | "Mona Lisa", "The Matrix" |
| **LANGUAGE** | Languages | "English", "Python" |

#### 2.4.2 IOB Labeling Scheme

**Inside-Outside-Beginning (IOB) Format:**

| Tag | Meaning | Example |
|-----|---------|---------|
| **B-TAG** | Beginning of entity | B-PER for "Elon" in "Elon Musk" |
| **I-TAG** | Inside entity | I-PER for "Musk" in "Elon Musk" |
| **O** | Outside (not an entity) | "is" in "Elon is CEO" |

```
Sentence: "Elon Musk founded SpaceX in California"

Elon      B-PER
Musk      I-PER
founded   O
SpaceX    B-ORG
in        O
California B-GPE
```

#### 2.4.3 NER with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino in 1976 and is worth $2.5 trillion")

for ent in doc.ents:
    print(f"{ent.text:20} {ent.start_char:3} {ent.end_char:3} {ent.label_:10}")

# Output:
# Apple Inc.           0   10  ORG
# Steve Jobs           27  37  PERSON
# Cupertino            41  50  GPE
# 1976                 54  58  DATE
# $2.5 trillion        72  85  MONEY

# Visualize entities
from spacy import displacy
displacy.render(doc, style="ent")
```

#### 2.4.4 Custom NER Training

**Why Custom NER?**
- Domain-specific entities (medical, legal, financial)
- Entities not in pre-trained models
- Better accuracy for specific use cases

**Example: Data Anonymization**

```python
# Replace sensitive entities with placeholders
import spacy
nlp = spacy.load("en_core_web_sm")

text = "John Smith called from 555-1234 to discuss account #12345"
doc = nlp(text)

anonymized = text
for ent in doc.ents:
    if ent.label_ == "PERSON":
        anonymized = anonymized.replace(ent.text, "[NAME]")
    elif ent.label_ == "CARDINAL":
        anonymized = anonymized.replace(ent.text, "[NUMBER]")

print(anonymized)
# "[NAME] called from [NUMBER]-[NUMBER] to discuss account #[NUMBER]"
```

### 2.5 Conditional Random Fields (CRF) for Custom NER

**Why CRF for NER?**
- Models dependencies between labels (IOB constraints: I-PER can't follow B-ORG)
- Uses features from context window
- Fast training and inference
- No need for large training data

#### 2.5.1 CRF Model Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRF FOR SEQUENCE LABELING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input:     Elon    Musk   founded  SpaceX                     │
│              │       │       │        │                          │
│   Features: ┌─────┐┌─────┐┌──────┐┌──────┐                      │
│             │ cap ││ cap ││lower ││ cap  │                      │
│             │-2:on││-2:sk││-4:ded││-1:eX │                      │
│             └─────┘└─────┘└──────┘└──────┘                      │
│                                                                  │
│   Labels:   B-PER  I-PER    O     B-ORG                         │
│              │      │       │      │                             │
│              └──────┴───────┴──────┘                             │
│                   CRF considers label transitions                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**CRF Formula:**
$$P(y|x) = \frac{1}{Z(x)} \exp\left(\sum_{i=1}^{n} \sum_k w_k f_k(y_{i-1}, y_i, x, i)\right)$$

Where:
- $y$ = label sequence
- $x$ = input sequence
- $f_k$ = feature functions
- $w_k$ = weights
- $Z(x)$ = normalization constant

#### 2.5.2 Feature Functions

Define features that help identify entities:

```python
def word_features(sentence, i):
    """Extract features for word at position i"""
    word = sentence[i]

    features = {
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],  # suffix
        'word[-2:]': word[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.startswith_capital': word[0].isupper() if word else False,
    }

    # Previous word features
    if i > 0:
        prev_word = sentence[i-1]
        features.update({
            '-1:word.lower': prev_word.lower(),
            '-1:word.istitle': prev_word.istitle(),
            '-1:word.isupper': prev_word.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    # Next word features
    if i < len(sentence) - 1:
        next_word = sentence[i+1]
        features.update({
            '+1:word.lower': next_word.lower(),
            '+1:word.istitle': next_word.istitle(),
            '+1:word.isupper': next_word.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

def sentence_features(sentence):
    """Extract features for all words in sentence"""
    return [word_features(sentence, i) for i in range(len(sentence))]
```

#### 2.5.3 Training CRF Model

```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# Sample training data (list of sentences, each with words and labels)
train_sents = [
    ([('Elon', 'B-PER'), ('Musk', 'I-PER'), ('founded', 'O'), ('SpaceX', 'B-ORG')]),
    ([('Apple', 'B-ORG'), ('is', 'O'), ('in', 'O'), ('Cupertino', 'B-LOC')]),
    # ... more training examples
]

# Extract features and labels
X_train = [sentence_features([w for w, _ in sent]) for sent in train_sents]
y_train = [[label for _, label in sent] for sent in train_sents]

# Train CRF
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,              # L1 regularization
    c2=0.1,              # L2 regularization
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

# Predict on test data
test_sent = [['Steve', 'Jobs', 'created', 'Apple']]
X_test = [sentence_features(test_sent[0])]
y_pred = crf.predict(X_test)

print(y_pred[0])
# ['B-PER', 'I-PER', 'O', 'B-ORG']
```

#### 2.5.4 Model Evaluation

```python
from sklearn_crfsuite import metrics

# Evaluate on test set
y_pred = crf.predict(X_test)
y_true = [[label for _, label in sent] for sent in test_sents]

# Overall F1 score
f1 = metrics.flat_f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.3f}")

# Per-entity type performance
labels = list(crf.classes_)
labels.remove('O')  # Remove 'O' tag

print(metrics.flat_classification_report(
    y_true, y_pred, labels=labels, digits=3
))

# Output:
#              precision    recall  f1-score   support
#
#       B-PER      0.850     0.900     0.874       100
#       I-PER      0.800     0.850     0.824        50
#       B-ORG      0.900     0.880     0.890       120
#       B-LOC      0.870     0.910     0.889        80
```

#### 2.5.5 Feature Importance

```python
# Examine which features are most important
from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print(f"{label_from:6} -> {label_to:6}   weight={weight:6.3f}")

# Transition features (label dependencies)
print("Top transition features:")
trans_features = Counter(crf.transition_features_).most_common(10)
print_transitions(trans_features)

# Output:
# B-PER  -> I-PER    weight= 3.450
# B-ORG  -> I-ORG    weight= 3.200
# O      -> B-PER    weight= 1.800
# I-PER  -> O        weight= 1.500
# B-ORG  -> O        weight= 1.400

# State features (word features)
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print(f"{label:6}   {attr:30}   weight={weight:6.3f}")

print("\nTop state features:")
state_features = Counter(crf.state_features_).most_common(20)
print_state_features(state_features)

# Output:
# B-PER    word.istitle                 weight= 2.800
# B-ORG    word.istitle                 weight= 2.500
# B-PER    -1:word.lower:mr             weight= 2.200
# B-LOC    word.istitle                 weight= 2.100
```

#### 2.5.6 Complete Example: Custom Medical NER

```python
# Train CRF for medical entity recognition
# Entities: DISEASE, DRUG, SYMPTOM

medical_train = [
    ([
        ('Patient', 'O'),
        ('diagnosed', 'O'),
        ('with', 'O'),
        ('diabetes', 'B-DISEASE'),
        ('takes', 'O'),
        ('Metformin', 'B-DRUG'),
        ('for', 'O'),
        ('high', 'B-SYMPTOM'),
        ('blood', 'I-SYMPTOM'),
        ('sugar', 'I-SYMPTOM')
    ]),
    # ... more examples
]

# Extract features
X_train = [sentence_features([w for w, _ in sent]) for sent in medical_train]
y_train = [[label for _, label in sent] for sent in medical_train]

# Train
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predict
test_sentence = ['Administered', 'aspirin', 'for', 'headache']
X_test = [sentence_features(test_sentence)]
prediction = crf.predict(X_test)[0]

print(list(zip(test_sentence, prediction)))
# [('Administered', 'O'),
#  ('aspirin', 'B-DRUG'),
#  ('for', 'O'),
#  ('headache', 'B-SYMPTOM')]
```

### 2.6 Sequence Labeling Models Comparison

| Model | Description | Strengths | Use When |
|-------|-------------|-----------|----------|
| **HMM** | Hidden Markov Model | Fast, simple, generative | Small data, baseline |
| **CRF** | Conditional Random Fields | Discriminative, considers global sequence, interpretable | Medium data, need interpretability |
| **BiLSTM-CRF** | Neural + CRF | Best of both: representation learning + sequence constraints | Large data, high accuracy needed |
| **Transformer** | BERT, RoBERTa | Contextual embeddings, state-of-the-art | Very large data, best accuracy |

**CRF Formula:**
$$P(y|x) = \frac{1}{Z(x)} \exp\left(\sum_{i=1}^{n} \sum_k \lambda_k f_k(y_{i-1}, y_i, x, i)\right)$$

**BiLSTM-CRF Architecture:**

```python
# BiLSTM-CRF with PyTorch
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, x, tags=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            return -self.crf(emissions, tags)  # Negative log-likelihood
        return self.crf.decode(emissions)  # Best tag sequence
```

### 2.5 Classical → Modern Mapping

| Classical | Modern Equivalent |
|-----------|-------------------|
| Explicit parse trees | Implicit syntactic knowledge in attention patterns |
| Dependency parsing | Attention heads learn syntactic dependencies |
| POS tagging (separate model) | Emergent in pre-trained LLMs (probe tasks show this) |
| Sequential modeling (RNN/LSTM) | Self-attention (parallel, long-range) |

---

## 3. Semantic Processing (Meaning)

> **What it is:** Understanding what text actually means beyond structure.

### 3.1 Named Entity Recognition (NER)

Identifying entities: persons, organizations, locations, dates, etc.

| Entity Type | Tag | Example |
|-------------|-----|---------|
| Person | PER | "**Elon Musk** founded SpaceX" |
| Organization | ORG | "Elon Musk founded **SpaceX**" |
| Location | LOC | "Apple is headquartered in **Cupertino**" |
| Date | DATE | "Meeting on **January 15, 2024**" |
| Money | MONEY | "Revenue of **$1.5 billion**" |

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino in 1976")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Apple Inc.: ORG
# Steve Jobs: PERSON
# Cupertino: GPE
# 1976: DATE
```

### 3.2 Word Sense Disambiguation (WSD)

Determining which meaning of a word is used in context.

```
"I deposited money at the bank"     → bank (financial institution)
"I sat by the river bank"           → bank (river edge)
"I need to bank on this investment" → bank (rely upon)
```

**Modern Approach:** Contextual embeddings naturally disambiguate!
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Same word, different contexts
sent1 = "I deposited money at the bank"
sent2 = "I sat by the river bank"

# Get contextual embeddings
inputs1 = tokenizer(sent1, return_tensors='pt')
inputs2 = tokenizer(sent2, return_tensors='pt')

with torch.no_grad():
    emb1 = model(**inputs1).last_hidden_state  # "bank" embedding differs!
    emb2 = model(**inputs2).last_hidden_state
```

### 3.3 Semantic Role Labeling (SRL)

Who did what to whom, when, where, how?

```
"John gave Mary a book yesterday at the library"

ARG0 (Agent):     John
VERB:             gave
ARG1 (Theme):     a book
ARG2 (Recipient): Mary
ARGM-TMP:         yesterday
ARGM-LOC:         at the library
```

### 3.4 Coreference Resolution

Linking mentions that refer to the same entity.

```
"John went to the store. He bought milk. The man was tired."
 ^^^^                     ^^              ^^^^^^^
   └──────────────────────┴───────────────┘ (same entity)
```

```python
# Using spaCy with neuralcoref (or modern alternatives)
import spacy

# For modern spaCy, use transformer-based models
nlp = spacy.load("en_core_web_trf")
doc = nlp("John went to the store. He bought milk.")
# Access coreference clusters
```

### 3.5 Sentence Embeddings

| Method | Description | Quality |
|--------|-------------|---------|
| **Bag-of-Words** | Word count vectors | Low |
| **TF-IDF** | Term frequency × inverse document frequency | Medium |
| **Doc2Vec** | Paragraph vectors | Medium |
| **Sentence-BERT** | Fine-tuned BERT for similarity | High |
| **Universal Sentence Encoder** | Multi-task trained encoder | High |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is fascinating",
    "I love artificial intelligence",
    "The weather is nice today"
]

embeddings = model.encode(sentences)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
# sentences[0] and sentences[1] will have high similarity
```

### 3.6 Classical → Modern Mapping

| Classical | Modern Equivalent |
|-----------|-------------------|
| WordNet for WSD | Contextual embeddings resolve ambiguity naturally |
| Knowledge graphs | Parametric knowledge stored in model weights |
| Separate NER/SRL models | Unified via prompting or fine-tuning |
| Sentence embeddings | [CLS] token or mean pooling of hidden states |

---

## 4. Pragmatic/Discourse Processing

> **What it is:** Understanding meaning in context, speaker intent, and coherence.

### 4.1 Sentiment Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS LEVELS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Document Level     Sentence Level     Aspect-Based              │
│  ┌───────────┐      ┌───────────┐      ┌───────────────────┐    │
│  │ Overall   │      │ Per-sent  │      │ "Food was great   │    │
│  │ sentiment │      │ sentiment │      │  but service slow"│    │
│  │ of review │      │           │      │  Food: +, Service:-│   │
│  └───────────┘      └───────────┘      └───────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from transformers import pipeline

# Zero-shot sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product! It's amazing!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Aspect-based sentiment (custom approach)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Fine-tuned ABSA model
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### 4.2 Intent Detection

What does the user want?

| Intent | Example Utterance |
|--------|-------------------|
| **book_flight** | "I want to fly to Paris next week" |
| **check_weather** | "What's the weather like today?" |
| **play_music** | "Play some jazz music" |
| **set_alarm** | "Wake me up at 7 AM" |

```python
from transformers import pipeline

# Zero-shot classification for intent
classifier = pipeline("zero-shot-classification")

text = "I want to book a flight to Tokyo"
candidate_labels = ["book_flight", "book_hotel", "check_weather", "play_music"]

result = classifier(text, candidate_labels)
# {'labels': ['book_flight', 'book_hotel', ...], 'scores': [0.95, 0.02, ...]}
```

### 4.3 Question Answering

| QA Type | Description | Example |
|---------|-------------|---------|
| **Extractive** | Extract span from context | "The capital of France is **Paris**" |
| **Abstractive** | Generate new answer | Summarize or paraphrase |
| **Open-domain** | Search + Read | No given context |

```python
from transformers import pipeline

# Extractive QA
qa_pipeline = pipeline("question-answering")

context = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf
tropical rainforest in the Amazon biome that covers most of the Amazon basin
of South America. This basin encompasses 7,000,000 km² of which 5,500,000 km²
are covered by the rainforest.
"""

question = "How large is the Amazon rainforest?"
result = qa_pipeline(question=question, context=context)
# {'answer': '5,500,000 km²', 'score': 0.89, 'start': 234, 'end': 247}
```

### 4.4 Text Summarization

| Type | Description | Models |
|------|-------------|--------|
| **Extractive** | Select important sentences | TextRank, LexRank |
| **Abstractive** | Generate new summary | BART, T5, Pegasus |

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Amazon rainforest is the world's largest tropical rainforest, covering
over 5.5 million square kilometers. It is home to approximately 10% of all
species on Earth. The forest produces about 20% of the world's oxygen and
plays a crucial role in regulating the global climate. However, deforestation
threatens this vital ecosystem, with thousands of square kilometers lost each year.
"""

summary = summarizer(article, max_length=50, min_length=25)
print(summary[0]['summary_text'])
```

### 4.5 Classical → Modern Mapping

| Classical | Modern Equivalent |
|-----------|-------------------|
| Dialogue state tracking | Conversation handled via context window |
| Intent classifiers | Zero-shot/few-shot prompting |
| Extractive summarization | Abstractive generation by LLMs |
| Pipeline QA systems | End-to-end generation (RAG for retrieval) |

---

## 5. Deep Learning for NLP

> **Transition:** From classical methods to neural approaches.

### 5.1 Recurrent Neural Networks (RNN)

Process sequences one element at a time, maintaining hidden state.

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

**Problems:**
- Vanishing/exploding gradients
- Difficulty with long-range dependencies

### 5.2 LSTM (Long Short-Term Memory)

Solves vanishing gradient with gating mechanisms.

```
┌─────────────────────────────────────────────────────────────────┐
│                        LSTM CELL                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────┐                                       │
│              ┌────►│  ×  │◄────┐                                │
│              │     └──┬──┘     │                                │
│              │        │        │                                │
│     c_{t-1} ─┤     ┌──▼──┐     │                                │
│              │     │  +  │─────┼────► c_t                       │
│              │     └──┬──┘     │                                │
│              │   ┌────┴────┐   │                                │
│              │   │    ×    │   │                                │
│              │   └────┬────┘   │                                │
│              │        │        │                                │
│     ┌────────┴────────┼────────┴────────┐                       │
│     │     Forget      │      Input      │                       │
│     │      Gate       │       Gate      │                       │
│     └────────┬────────┴────────┬────────┘                       │
│              │                 │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│     h_{t-1} ──────────┼──────────────────────────► h_t          │
│                       │                                         │
│              x_t ─────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**LSTM Equations:**

| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | What to forget |
| **Input** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | What to remember |
| **Output** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | What to output |
| **Cell** | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | Memory update |
| **Hidden** | $h_t = o_t \odot \tanh(c_t)$ | Output state |

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embeds)
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)
```

### 5.3 Seq2Seq with Attention

The precursor to Transformers!

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEQ2SEQ WITH ATTENTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Encoder:    [h1]  [h2]  [h3]  [h4]                            │
│                 │     │     │     │                              │
│                 └─────┴─────┴─────┘                              │
│                         │                                        │
│                    Attention Weights                             │
│                    [0.1, 0.7, 0.1, 0.1]                         │
│                         │                                        │
│                    Context Vector                                │
│                         │                                        │
│   Decoder:         [s1] ──► [s2] ──► [s3]                       │
│                     │        │        │                          │
│                    "I"     "love"   "you"                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Attention Mechanism:**
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$
$$c_i = \sum_j \alpha_{ij} h_j$$

Where $e_{ij} = a(s_{i-1}, h_j)$ is the alignment score.

---

## 6. Transformer Architecture

> **"Attention Is All You Need"** — The architecture powering everything today.

### 6.0 Complete Transformer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORMER ARCHITECTURE                                       │
│                        ("Attention Is All You Need", 2017)                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│         ENCODER (N× layers)                      DECODER (N× layers)                    │
│        ┌─────────────────────┐                  ┌─────────────────────┐                 │
│        │                     │                  │                     │                 │
│        │  ┌───────────────┐  │                  │  ┌───────────────┐  │   Output        │
│        │  │  Add & Norm   │  │                  │  │  Add & Norm   │  │  Probabilities  │
│        │  └───────┬───────┘  │                  │  └───────┬───────┘  │      ↑          │
│        │          │          │                  │          │          │      │          │
│   ·····│··········│··········│··················│··········│··········│·· ┌──────┐ ····│
│   :    │  ┌───────┴───────┐  │                  │  ┌───────┴───────┐  │   │Softmax│    :
│   :    │  │  Feed Forward │  │                  │  │  Feed Forward │  │   └──┬───┘    :
│   :    │  └───────┬───────┘  │                  │  └───────┬───────┘  │      │        :
│   :    │          │          │                  │          │          │   ┌──┴───┐    :
│   N    │  ┌───────┴───────┐  │                  │  ┌───────┴───────┐  │   │Linear│    :
│   ×    │  │  Add & Norm   │  │                  │  │  Add & Norm   │  │   └──────┘    :
│   :    │  └───────┬───────┘  │                  │  └───────┬───────┘  │              :
│   :    │          │          │                  │          │          │              :
│   :    │  ┌───────┴───────┐  │    ┌─────────┐   │  ┌───────┴───────┐  │              :
│   :    │  │  Multi-Head   │  │    │         │   │  │  Multi-Head   │  │              :
│   :    │  │  Attention    │  │────┤ Cross   ├───│  │  Attention    │  │              :
│   ·····│··└───────┬───────┘··│····│Attention│···│··└───────┬───────┘··│··············│
│        │          │          │    │         │   │          │          │              │
│        │          │          │    └─────────┘   │  ┌───────┴───────┐  │              │
│        │          │          │                  │  │  Add & Norm   │  │              │
│        │          │          │                  │  └───────┬───────┘  │              │
│        │          │          │                  │          │          │              │
│        │          │          │                  │  ┌───────┴───────┐  │              │
│        │          │          │                  │  │ Masked Multi- │  │              │
│        │          │          │                  │  │ Head Attention│  │              │
│        │          │          │                  │  └───────┬───────┘  │              │
│        └──────────┼──────────┘                  └──────────┼──────────┘              │
│                   │                                        │                         │
│           ┌───────┴───────┐                        ┌───────┴───────┐                 │
│           │      ⊕        │                        │      ⊕        │                 │
│           │  (Positional  │                        │  (Positional  │                 │
│           │   Encoding)   │                        │   Encoding)   │                 │
│           └───────┬───────┘                        └───────┬───────┘                 │
│                   │                                        │                         │
│           ┌───────┴───────┐                        ┌───────┴───────┐                 │
│           │    Input      │                        │    Output     │                 │
│           │   Embedding   │                        │   Embedding   │                 │
│           └───────┬───────┘                        └───────┬───────┘                 │
│                   │                                        │                         │
│           ┌───────┴───────┐                        ┌───────┴───────┐                 │
│           │    Inputs     │                        │    Outputs    │                 │
│           │               │                        │ (shifted right)│                │
│           └───────────────┘                        └───────────────┘                 │
│                                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Description | Purpose |
|-----------|-------------|---------|
| **Input Embedding** | Convert tokens to dense vectors | Learnable representation |
| **Positional Encoding** | Add position information | Sequence order awareness |
| **Multi-Head Attention** | Parallel attention heads | Multiple relationship types |
| **Masked Multi-Head Attention** | Causal masking (decoder) | Prevent looking ahead |
| **Cross-Attention** | Encoder-Decoder connection | Align source and target |
| **Feed Forward** | Position-wise FFN | Non-linear transformation |
| **Add & Norm** | Residual + LayerNorm | Training stability |

### 6.1 Self-Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION MECHANISM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: "The cat sat"                                          │
│                                                                  │
│           ┌───────────────────────────────────────┐             │
│           │         Query (Q), Key (K), Value (V) │             │
│           └───────────────────────────────────────┘             │
│                            │                                     │
│                    ┌───────┼───────┐                            │
│                    ▼       ▼       ▼                            │
│                   [Q]     [K]     [V]                           │
│                    │       │       │                            │
│                    └───┬───┘       │                            │
│                        │           │                            │
│                   Q × K^T          │                            │
│                        │           │                            │
│                   ÷ √d_k           │                            │
│                        │           │                            │
│                    Softmax         │                            │
│                        │           │                            │
│                        └─────×─────┘                            │
│                              │                                   │
│                         Attention                                │
│                          Output                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)
```

### 6.2 Positional Encoding

Inject sequence order information (Transformers have no inherent notion of position).

**Sinusoidal Encoding:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 6.3 Architecture Variants

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER ARCHITECTURE VARIANTS               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENCODER-ONLY          DECODER-ONLY         ENCODER-DECODER     │
│  (Bidirectional)       (Autoregressive)     (Seq2Seq)           │
│                                                                  │
│  ┌─────────────┐       ┌─────────────┐      ┌─────────────┐     │
│  │   [CLS]     │       │ Token 1 ──► │      │   Encoder   │     │
│  │   Token 1   │       │ Token 2 ──► │      │      │      │     │
│  │   Token 2   │       │ Token 3 ──► │      │      ▼      │     │
│  │   Token 3   │       │     ...     │      │   Decoder   │     │
│  │   [SEP]     │       │ Token N ──► │      │      │      │     │
│  └─────────────┘       └─────────────┘      └─────────────┘     │
│                                                                  │
│  Examples:             Examples:             Examples:           │
│  • BERT                • GPT-2/3/4           • T5                │
│  • RoBERTa             • LLaMA               • BART              │
│  • ALBERT              • Claude              • mT5               │
│  • DeBERTa             • Mistral             • NLLB              │
│                                                                  │
│  Best for:             Best for:             Best for:           │
│  • Classification      • Text generation     • Translation       │
│  • NER                 • Chatbots            • Summarization     │
│  • Embeddings          • Code completion     • QA                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Architecture | Type | Examples | Use Case |
|--------------|------|----------|----------|
| **Encoder-only** | Bidirectional | BERT, RoBERTa, ALBERT | Classification, NER, embeddings |
| **Decoder-only** | Autoregressive | GPT, LLaMA, Claude | Text generation, chat |
| **Encoder-Decoder** | Seq2Seq | T5, BART, mT5 | Translation, summarization |

---

## 7. Large Language Models (LLMs)

> **Putting It All Together:** Scale + Data + Compute → Emergent Capabilities

### 7.1 Pre-training Objectives

| Objective | Description | Models |
|-----------|-------------|--------|
| **MLM** (Masked Language Modeling) | Predict masked tokens | BERT, RoBERTa |
| **CLM** (Causal Language Modeling) | Predict next token | GPT, LLaMA |
| **Span Corruption** | Predict corrupted spans | T5 |
| **Prefix LM** | Bidirectional prefix + autoregressive | PaLM |

**Masked Language Modeling:**
```
Input:  "The [MASK] sat on the [MASK]"
Output: "The cat sat on the mat"
```

**Causal Language Modeling:**
```
Input:  "The cat sat on the"
Output: "mat"
```

### 7.2 Scaling Laws

$$L(N, D, C) \propto N^{-0.076} + D^{-0.095} + C^{-0.050}$$

Where:
- $N$ = Number of parameters
- $D$ = Dataset size
- $C$ = Compute budget

**Key Insight:** Performance improves predictably with scale across all three axes.

| Model | Parameters | Training Tokens |
|-------|------------|-----------------|
| BERT-base | 110M | 3.3B |
| GPT-2 | 1.5B | 40B |
| GPT-3 | 175B | 300B |
| LLaMA-2 | 7B-70B | 2T |
| GPT-4 | ~1.7T (estimated) | ~13T |

### 7.3 Fine-tuning Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING STRATEGIES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Full Fine-tuning    LoRA               Prefix-Tuning           │
│  ┌─────────────┐     ┌─────────────┐    ┌─────────────┐         │
│  │ Update ALL  │     │ Low-Rank    │    │ Learnable   │         │
│  │ parameters  │     │ Adapters    │    │ Prefix      │         │
│  │             │     │ (0.1% params)│   │ Tokens      │         │
│  └─────────────┘     └─────────────┘    └─────────────┘         │
│                                                                  │
│  100% params         0.1-1% params      0.1% params             │
│  Expensive           Efficient          Very efficient          │
│  Risk forgetting     Preserves base     Task-specific           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**LoRA (Low-Rank Adaptation):**

$$W' = W + BA$$

Where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

### 7.4 Alignment Techniques

Making models helpful, harmless, and honest.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ALIGNMENT PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Pre-training        SFT              RLHF/DPO                 │
│   ┌──────────┐       ┌──────────┐     ┌──────────┐             │
│   │ Internet │  ──►  │ Curated  │ ──► │ Human    │             │
│   │ Text     │       │ Examples │     │ Feedback │             │
│   │ (raw)    │       │          │     │          │             │
│   └──────────┘       └──────────┘     └──────────┘             │
│                                                                  │
│   Learns language    Learns format    Learns values             │
│   & knowledge        & following      & preferences             │
│                      instructions                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Technique | Description | Pros | Cons |
|-----------|-------------|------|------|
| **SFT** (Supervised Fine-Tuning) | Train on (prompt, response) pairs | Simple, effective | Needs quality data |
| **RLHF** | Reinforcement Learning from Human Feedback | Aligns with preferences | Complex, unstable |
| **DPO** (Direct Preference Optimization) | Direct preference learning without RL | Simpler than RLHF | May be less flexible |
| **Constitutional AI** | Self-improvement with principles | Scalable | Still needs initial guidance |

**DPO Loss:**
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 7.5 Inference Techniques

#### Prompting Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **Zero-shot** | No examples | "Translate to French: Hello" |
| **Few-shot** | Provide examples | "EN: Hello → FR: Bonjour\nEN: Goodbye → FR:" |
| **Chain-of-Thought** | Step-by-step reasoning | "Let's think step by step..." |
| **Self-Consistency** | Multiple reasoning paths, vote | Sample multiple CoT, majority vote |

```python
# Chain-of-Thought Prompting
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
Roger starts with 5 balls.
He buys 2 cans × 3 balls = 6 balls.
Total = 5 + 6 = 11 balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more,
how many apples do they have?

A: Let's think step by step.
"""
```

#### Retrieval-Augmented Generation (RAG)

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query                                                     │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────┐     ┌───────────────────┐                       │
│   │  Embed    │────►│  Vector Database  │                       │
│   │  Query    │     │  (documents)      │                       │
│   └───────────┘     └───────────────────┘                       │
│                              │                                   │
│                              ▼                                   │
│                     Top-K Similar Docs                           │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────┐               │
│   │  Prompt = Query + Retrieved Context         │               │
│   └─────────────────────────────────────────────┘               │
│                              │                                   │
│                              ▼                                   │
│                     ┌───────────────┐                           │
│                     │     LLM       │                           │
│                     └───────────────┘                           │
│                              │                                   │
│                              ▼                                   │
│                     Grounded Answer                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create RAG chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
response = qa_chain.run("What are the main findings in the research paper?")
```

### 7.6 Emergent Capabilities

Abilities that appear only at scale:

| Capability | Description | Example Models |
|------------|-------------|----------------|
| **In-context learning** | Learn from examples in prompt | GPT-3+ |
| **Chain-of-thought reasoning** | Step-by-step problem solving | PaLM, GPT-4 |
| **Code generation** | Write functional code | Codex, GPT-4, Claude |
| **Tool use** | Use external APIs/tools | GPT-4, Claude |
| **Multilingual** | Transfer across languages | mT5, BLOOM |

---

## 8. Applications & Practice Projects

### 8.1 Industry Applications

| Application | Techniques | Industry |
|-------------|------------|----------|
| **Chatbots/Assistants** | LLMs, RAG, Intent Detection | Customer Service |
| **Document Processing** | NER, Summarization, QA | Legal, Finance |
| **Sentiment Analysis** | Classification, Aspect-based | Marketing, Social Media |
| **Machine Translation** | Seq2Seq, Transformer | Localization |
| **Code Generation** | LLMs, Fine-tuning | Software Development |
| **Content Generation** | LLMs, Prompting | Media, Marketing |
| **Search/Recommendations** | Embeddings, Semantic Search | E-commerce |

### 8.2 Practice Projects

| Level | Project | Skills |
|-------|---------|--------|
| **Beginner** | Sentiment Analysis on Movie Reviews | Tokenization, Classification, Fine-tuning BERT |
| **Beginner** | Named Entity Recognition | Sequence Labeling, spaCy, Transformers |
| **Intermediate** | Question Answering System | Extractive QA, BERT/RoBERTa |
| **Intermediate** | Text Summarization | T5, BART, Abstractive Generation |
| **Advanced** | RAG Chatbot | Embeddings, Vector DB, LangChain, LLM |
| **Advanced** | Fine-tune LLM with LoRA | PEFT, LoRA, QLoRA, Training |
| **Advanced** | Build Custom NER System | CRF, BiLSTM-CRF, Domain Adaptation |

### 8.3 Recommended Tools & Libraries

| Category | Tools |
|----------|-------|
| **NLP Libraries** | spaCy, NLTK, Stanza |
| **Transformers** | Hugging Face Transformers, Tokenizers |
| **LLM Frameworks** | LangChain, LlamaIndex, Haystack |
| **Vector Databases** | Pinecone, Weaviate, Chroma, FAISS |
| **Fine-tuning** | PEFT, Axolotl, LLaMA-Factory |
| **Experiment Tracking** | Weights & Biases, MLflow |
| **Deployment** | vLLM, TGI, Ollama |

---

## Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│                      KEY TAKEAWAYS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Classical NLP isn't obsolete                                │
│     → Helps you understand what Transformers learn implicitly   │
│                                                                  │
│  2. Transformers unify the pipeline                             │
│     → One architecture: tokenization → generation end-to-end    │
│                                                                  │
│  3. Attention replaces explicit structures                      │
│     → Syntactic/semantic relations emerge in attention patterns │
│                                                                  │
│  4. Scale + Data + Compute                                      │
│     → The recipe for modern LLM capabilities                    │
│                                                                  │
│  5. Alignment is critical                                       │
│     → Raw pre-training isn't enough; RLHF/DPO makes models      │
│       useful and safe                                           │
│                                                                  │
│  6. RAG extends LLM capabilities                                │
│     → Ground responses in real data, reduce hallucinations      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Learning Path

```
[Foundation]
    │
    ├─► Linear Algebra, Probability, Python/PyTorch
    │
[Classical NLP] (2-3 weeks)
    │
    ├─► Tokenization → Word Embeddings (Word2Vec, GloVe)
    ├─► POS Tagging → Dependency Parsing
    ├─► NER, Sentiment Analysis
    │
[Deep Learning for NLP] (2 weeks)
    │
    ├─► RNNs → LSTMs → Seq2Seq + Attention
    │
[Transformers] (2-3 weeks)
    │
    ├─► "Attention Is All You Need" paper
    ├─► BERT (encoder) → GPT (decoder) → T5 (enc-dec)
    │
[LLMs & Alignment] (3-4 weeks)
    │
    ├─► Scaling laws, pre-training, fine-tuning
    ├─► RLHF, prompting, RAG
    └─► Hands-on: Hugging Face Transformers, LangChain
```

---

## Essential Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Word2Vec | 2013 | Dense word embeddings |
| GloVe | 2014 | Global co-occurrence embeddings |
| Seq2Seq | 2014 | Encoder-decoder for sequences |
| Attention Mechanism | 2015 | Bahdanau attention |
| **Attention Is All You Need** | 2017 | Transformer architecture |
| BERT | 2018 | Bidirectional pre-training |
| GPT-2 | 2019 | Large-scale language models |
| T5 | 2020 | Text-to-text framework |
| GPT-3 | 2020 | In-context learning |
| InstructGPT/RLHF | 2022 | Alignment with human feedback |
| LLaMA | 2023 | Open-source LLMs |
| DPO | 2023 | Direct preference optimization |

---

[← Back to Course 4](../course4-deep-learning/README.md) | [Main Roadmap](../README.md)
