#! python

from agent import build_dqn

print("Hello")

model = build_dqn(0.9, 9,  9)
print(model.summary())
pred = model.predict([[0 for i in range(9)]])
print(pred)