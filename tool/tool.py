import matplotlib.pyplot as plt

# todo
def grid(x,y):#plt风格
    h,w=3,3
    return h,w,(x-1)*w+y

def pltGrid(x,y):
    plt.subplot(*grid(x,y))


def put_list(li,width = 3):
    ws = width*[0]
    strLi = []
    print('\n')
    for i,v in enumerate(li):
        s = '\''+str(v)+'\''+'  '
        ws[i%width] = max(len(s),ws[i%width])
        strLi.append(s)
    for i,v in enumerate(strLi):
        print(v+" "*(ws[i%width]-len(v)) ,end='')
        if(i%width == width-1):
            print('')
