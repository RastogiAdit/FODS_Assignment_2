import numpy as np

def kl_divergence(p,q):
  if not (np.isclose(sum(p),1) and np.isclose(sum(q),1)):
    raise ValueError("The probabilities don't add up to 1")
  if any (x<0 for x in p) and any(x<0 for x in q):
    raise ValueError("Probabilities can't be negative")

  kl = 0
  for p_i,q_i in zip(p,q):
    if p_i>0:
      kl += p_i * np.log(p_i/q_i)

  return round(kl,4)

p_default = [0.1, 0.3, 0.4, 0.1, 0.1]
q_default = [0.2, 0.2, 0.3, 0.2, 0.1]

print("Do you want to enter your own values? (y/n)")
choice = input()
if choice == 'y':
    print ("Enter size of random variables")
    n = int(input())
    p = []
    q = []
    print("Enter values of p")
    for i in range(n):
        p.append(float(input()))
    print("Enter values of q")
    for i in range(n):
        q.append(float(input()))
    ans = kl_divergence(p, q)
else:
    ans = kl_divergence(p_default, q_default)

print("KL Divergence value is ",ans)