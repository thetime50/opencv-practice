import numpy as np
def getDctCoefficient(side):
    res = np.empty([side,side,side,side],dtype="float64") # v,u,y,x
    # u=0 v=0
    res[0][0] = 1/side
    # v==0
    for v in range(1,side):
        for y in range(0,side):
            res[0,v,:,y] = res[v,0,y,:] = np.sqrt(2)/side * np.cos((y+0.5)*v*np.pi/side)
    # for v in range(1,side):
    #     for u in range(1,side):
    #         pass
    xyIndex = [(y,x) for x in range(0,side) for y in range(0,side)]
    uvIndex = [(v,u) for v in range(1,side) for u in range(1,side)]
    def uvxyFun_(uv,xy):
        v = uv[0]
        u = uv[1]
        y = xy[0]
        x = xy[1]
        res[v,u,y,x] = res[v,0,y,x] * res[0,u,y,x] * side
    def uvFun_(uv):
        # v = uv[0]
        # u = uv[1]
        for xy in xyIndex:
            uvxyFun_(uv,xy)

    for uv in uvIndex:
        uvFun_(uv)
    return res

def dct2d(img,coe):
    res = np.empty(coe.shape[:2],dtype="float64")
    for v in range(len(coe)):
        for u in range(len(coe[0])):
            res[u][v] = np.sum(img*coe[u][v])
    return res
    # res = np.dot(img,coe)
    # res = np.dot(img,np.transpose(coe))
    # return res.reshape(8,8)


def dct2dBlock(img,side = 8):
    coe8 = getDctCoefficient(side)
    yblock = img.shape[0] // side
    xblock = img.shape[1] // side
    res = [[None for bx in range(xblock)] for by in range(yblock)]
    
    
    for by in range(0,yblock):
        y = by * side
        for bx in range(0,xblock):
            x = bx *side
            roi = img[y:y+side,x:x+side]
            res[by][bx] = dct2d(roi,coe8)
    return res

