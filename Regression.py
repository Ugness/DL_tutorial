import numpy as np
import random
from matplotlib import pyplot as plt

# 초기 설정 y = 2 * x^2 + 4 에 유사한 데이터를 가지고 있다고 가정.
x = np.array([i for i in range(-100, 100)])
x = x / 20.
error = 2 * (np.array([random.random() for i in range(len(x))]) - 0.5)
y = 2 * np.multiply(x, x) + 4 + error

datas = np.stack([x, y], axis=1).astype(np.float64)
print(datas.shape)

# 우리는 target function이 2차함수임을 알고 있음. (가정)
# y = a*x^2 + b에서 a, b를 추정하고자 함.
a = random.random()
b = random.random()

# Gradient Descent 가정.
learning_rate = 0.00005

epoch = 10000
for i in range(epoch):
    for j, (input, target) in enumerate(datas):
        _y = a * input * input + b                  # 모델에 input을 넣음

        L2 = (target - _y) ** 2                             # Loss를 계산
        L2_grad_a = 2 * (target - _y) * (input * input)  # L2를 a에 대해 미분.
        L2_grad_b = 2 * (target - _y) * (1)              # L2를 b에 대해 미분.

        a += L2_grad_a * learning_rate              # Gradient Descent 적용
        b += L2_grad_b * learning_rate              # Gradient Descent 적용
        # print(L2_grad_a, L2_grad_b)
        # print('Iter : {:4d}, a : {:4f}, b : {:4f}'.format(j, a, b))
    print('Epoch : {:4d}, a : {:.4f}, b : {:.4f}'.format(i, a, b))

plt.title('Regression')
plt.plot(x, y)
plt.plot(x, a * x * x + b)
plt.show()
