# %%

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from common.util import ppmi
import numpy as np

# %%
# 2.3.1 파이썬으로 말뭉치 전처리 하기


def preprocess(text) -> list:
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            id_to_word[new_id] = word
            word_to_id[word] = new_id

    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word


# %%
text = 'You say goodbye and I say hello.'
# %%
corpus, word_to_id, id_to_word = preprocess(text)
# %%


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i  # left window_size
            right_idx = idx + i  # right window_size

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    # epsilon 값을 추가해,
    # 0으로 나누기 오류가 나는 것을 막아줌
    nx = x / np.sqrt(np.sum(x**2) + eps)  # x의 정규화
    ny = y / np.sqrt(np.sum(y**2) + eps)  # y의 정규화
    return np.dot(nx, ny)


# %%
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]

print(cos_similarity(c0, c1))

# %%
# %%

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)
W = uppmi(C)

np.set_printoptions(precision=3)
print(W)
# %%
# SVD
U, S, V = np.linalg.svd(W)


for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
# %%

# PTB 데이터셋 평가
import sys
sys.path.append('..')
from dataset import ptb
from common.util import create_co_matrix, ppmi

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

# 동시 발생 수 계산
C = create_co_matrix(corpus, vocab_size, window_size)

# PPMI 계산
W = ppmi(C, verbose=True)
