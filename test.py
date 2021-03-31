
class myclass:
    num =10
    char = 'a'
    def __init__(self,name):
        self.name =name
        print('hello by init')
        
    def sum1(self):
        print('hello from sum')
m1= myclass('kdeep')   
m1.sum1() 
m2=myclass('11')
print(m1.num,"\n" ,m2.char)       
print(m1.name,'\n',m2.name)
c = add.add_nos(7,8)
print(c)