import tool

print(dir(tool))
# print(dir(locals()))

tool.put_list(dir(locals()))

# if __name__=='__main__':
#     print()

class Aaaa:
    i=0
    def __init__(self):
        self.serial = Aaaa.i
        Aaaa.i+=1

a = Aaaa()
b = Aaaa()

print(a.i,a.serial,b.i,b.serial)

for a in [4,5,6,7]:
    print(a)