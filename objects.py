s = 'my name is kuldee'
m= s.upper()
m2= s.split('s')
print(m2)

k1 = {'k1' :1 , 'k2': 3}
#d=k1.keys()
d=k1.items()
print(d)

lst = [1,2,3,4]
lst.append('new')

print(lst)
first = lst.pop(0)
print(lst)
print(first)
x = [(1,2),(3,4)]
c=x[0][1]
print(c)
for item in x: #tuples of list x 
    print(item)
for a,b in x: #tuple unpacking (each item )inside the for loop
    print(a)
    print(b)   