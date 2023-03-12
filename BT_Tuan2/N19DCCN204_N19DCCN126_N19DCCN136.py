# %%
"""
WEEK 2
- N19DCCN204: Phạm Văn Thuận
- N19DCNC126: Lê Hoài Nhân
- N19DCCN136: Nguyễn Trọng Tín
Stages:
1. Processing data (docs, vocabulary, queries) -> OK
2. Calculating tf (docs, queries) depend on vocabulary (fixed length)
3. Calculating idf (docs, queries) depend on vocabulary (fixed length)
4. Calculating cosine similarity between each query and docs. 
  (for one query corresponding docs) -> weight
5. Sorting decrease with weight
"""
# %%
# Import library and download stop words
import numpy as np
from functools import reduce
import string
import nltk
from nltk.corpus import stopwords
import math

nltk.download("stopwords")

# %%
"""re_processing_words
@Param {string} vocab (according to ./term-vocab)
@Return {list<string>} a list of lowercase-words sorted by alphabet
"""


def re_processing_words(vocab):
    process_words = []
    for line in vocab:
        splitted_line = line.split()[:-1]

        for word in splitted_line:
            if word.strip() and not word.isdigit():
                process_words.append(word.lower())
    process_words.sort()
    return process_words


# %%
"""remove_stop_words
@Param {list<string>} docs: list of document (according to doc-text)
@Return {list<string>} list of document was removed stop-word
"""


def remove_stop_words(docs):
    english_stop_words = stopwords.words("english")
    new_docs = []
    for doc in docs:
        new_docs.append([word for word in doc.split() if word not in english_stop_words])
    return new_docs


# %%
"""re_processing_docs
@Param {list<string>} docs:  docs: list of document (according to doc-text)
@Return {list<string>}
"""


def re_processing_docs(docs):
    processed_docs = []
    # removing digit and break line ('\n)
    st1_pr_docs = list(filter(lambda y: not y.isdigit(), map(lambda x: x[:-1], docs)))
    # splitting allow forward flash
    table_remove_punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    docs_tmp = []
    for doc in st1_pr_docs:
        if doc.endswith("/"):
            processed_docs.append(" ".join(docs_tmp))
            docs_tmp.clear()
        else:
            docs_tmp.append(" ".join([w.lower().translate(table_remove_punctuation) for w in doc.split()]))
    return processed_docs


# %%
# pre-processing docs
docs = None
with open("./dataset/doc-text", "r") as f:
    docs = f.readlines()

    f.close()
processed_docs = remove_stop_words(re_processing_docs(docs))

# %%
# pre-processing vocabulary
vocab = None
with open("./dataset/term-vocab", "r") as f:
    vocab = f.readlines()

    f.close()
processed_vocab = re_processing_words(vocab)

# %%
# re-processing query docs
query_docs = None
with open("./dataset/query-text", "r") as f:
    query_docs = f.readlines()
    f.close()

processed_query = remove_stop_words(re_processing_docs(query_docs))

# %%
# Calc tf

tf_query_docs = [[query_doc.count(word) for word in processed_vocab] for query_doc in processed_query]
tf_docs = [[doc.count(word) for word in processed_vocab] for doc in processed_docs]

# %%
# Testing -> Depend on vocabulary, so these values will equal!
print(len(processed_vocab))
print(len(tf_query_docs[0]), len(tf_query_docs[1]))
print(len(tf_docs[0]), len(tf_docs[1]))


# %%
# Calc weight_tf of queries and docs
def calc_weight_tf(tf_matrix):
    weight_tf_matrix = []
    for tf_vec in tf_matrix:
        weight_tf_vec = []
        for tf in tf_vec:
            if tf <= 0:
                weight_tf_vec.append(0)
            else:
                weight_tf_vec.append(1 + math.log10(tf))
        weight_tf_matrix.append(weight_tf_vec)

    return weight_tf_matrix


weight_tf_query_docs = calc_weight_tf(tf_query_docs)
weight_tf_docs = calc_weight_tf(tf_docs)

# %%
# Statistic words in vocabulary corresponding the docs
num_of_docs = len(processed_docs)
df_docs = []

for word in processed_vocab:
    df_tmp = 0
    for doc in processed_docs:
        if word in doc:
            df_tmp += 1

    # Handling divide by 0
    if df_tmp == 0:
        df_docs.append(1)
    else:
        df_docs.append(df_tmp)

print(len(df_docs))

# %%
# Calc idf
idf_docs = [math.log10(num_of_docs / df) for df in df_docs]

print(len(idf_docs))
print(idf_docs)


# %%
# Calc weight (w_tf * idf) and normalize
def calc_weight_tf_idf_vec(weight_tf_vec, idf_vec):
    return [np.multiply(np.array(weight_tf), np.array(idf_vec)) for weight_tf in weight_tf_vec]


def normalize_cosine(weight_vec):
    result = []
    for weight in weight_vec:
        len_doc = np.linalg.norm(weight)
        if len_doc == 0:
            result.append(weight)
        else:
            result.append(weight / len_doc)

    return result


weight_docs_normalize = normalize_cosine(calc_weight_tf_idf_vec(weight_tf_docs, idf_docs))
weight_queries_normalize = normalize_cosine(calc_weight_tf_idf_vec(weight_tf_query_docs, idf_docs))

print(weight_docs_normalize)
print(weight_queries_normalize)

# %%
# Calc cosine similarity
cos_sim_vec = []
for weight_query in weight_queries_normalize:
    cos_sim_dict = {}
    len_weight_query = np.linalg.norm(weight_query)
    for doc_id, weight_doc in enumerate(weight_docs_normalize):
        product_len = np.linalg.norm(weight_doc) * len_weight_query
        if product_len == 0:
            cos_sim_dict[doc_id + 1] = np.dot(weight_doc, weight_query)
        else:
            cos_sim_dict[doc_id + 1] = np.dot(weight_doc, weight_query) / product_len

    list_tmp = list(cos_sim_dict.items())
    list_tmp.sort(key=lambda x: x[1], reverse=True)
    cos_sim_vec.append(dict(list_tmp))

print(cos_sim_vec)
# %%
for i, cos_sim_val in enumerate(cos_sim_vec):
    print("The 3th's docs similarity with the query (desc) ", i + 1, ": ")
    print(list(cos_sim_val.keys())[:3])
