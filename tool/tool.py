import matplotlib.pyplot as plt

gridH,gridW = 3,3
def setGrid(h,w):#plt风格
    global gridH,gridW
    gridH,gridW = h,w

# def grid(x,y):#plt风格
#     return gridH,gridW,(x-1)*gridW+y

# def pltGrid(x,y):
#     plt.subplot(*grid(x,y))

def pltGrid(x,y,colspan=1,rowspan=1):
    plt.subplot2grid((gridH,gridW),(y-1,x-1),colspan=colspan,rowspan=rowspan)


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
