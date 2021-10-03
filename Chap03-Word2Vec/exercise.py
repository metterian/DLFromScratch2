# %%
import sys, os
# sys.path.append('..')

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from common.optimizer import SGD
from common import optimizer
import numpy as np
from common.layers import MatMul
from common.util import preprocess, create_contexts_target, convert_one_hot
from common.layers import MatMul, SoftmaxWithLoss
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import convert_one_hot, preprocess, create_contexts_target


# %%
# 샘플 맥락 데이터
c0 = np.array([1, 0, 0, 0, 0, 0, 0])  # you
c1 = np.array([0, 1, 0, 0, 0, 0, 0])  # goodbye


# Weight Initialization
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)


# Layer
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# Feed Forward
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)


# %%
# you와 goodbye 라는 맥락이 주어졌을때, 타깃단어의 생성확률
s
# %%

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

context, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
context = convert_one_hot(context, vocab_size)
# %%


class SimpleCBOW(object):
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 가중치와 기울기 모으기
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, context, target):
        h0 = self.in_layer0.forward(context[:, 0])
        h1 = self.in_layer1.forward(context[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1) -> None:
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None


# %%
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 3000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
# optimizer = SGD()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch,  batch_size)
trainer.plot()
# %%
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
# %%
