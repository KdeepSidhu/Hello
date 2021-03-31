class kuldeep():
    brother = 'jagdeep'
    age = 10

    def __init__(self,name, standard):
        self.name = name
        self.standard = standard
        print('through self method :' )

    def sumofnos(self ,a,b):
        return a+b
        
if __name__ == "__main__":
    m1 = kuldeep('kdeep','24')
    m2 = kuldeep('kdeep','24')
    c = m1.sumofnos(3,4)
    print('sumis: %0.2f'% c )
    print(m1.brother , m1.age)
    print(m1.name , m1.standard)