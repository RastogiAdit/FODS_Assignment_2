names = ["Adit", "Harry", "Jack"]

size_of_names = len(names)
print(size_of_names)
i = 0
while i < size_of_names:
    print(names[i])
    i += 1

for name in names:
    print(name, end=' ')

#for loops
print("\n")
for name in names:
    print(name)


range(5) # 0,1,2,3,4
range(2,5) # 2,3,4
numbers = range(5)
for item in numbers:
    print(item)


numbers = (1,2,3,2,2)
print(numbers.count(2))


X = [[]]
for i in range(n):
    a = []
    for j in range(m):
        a.append(int(input()))
    X.append(a)
y = []
for i in range(n):
    y.append(int(input()))

for i in range(n):
    for j in range(m):
        print(X[i][j], end=' ')
    print("\n")

for i in range(n):
    print(y[i], end=' ')


np.log(prob_plus)