import numpy as np
import jpeg
import matplotlib.pyplot as plt

side = 8
dctcoe = jpeg.getDctCoefficient(side)
dctcoeMin = np.min(dctcoe)
dctcoeMax = np.max(dctcoe)
print(f"min: {dctcoeMin}, max: {dctcoeMax}")
for v in range(side):
    for u in range(side):
        plt.subplot2grid([side,side],[u,v])
        plt.title(f"{u}-{v}",y=0.8)
        plt.axis("off")
        plt.imshow(dctcoe[v,u],"gray",vmin= dctcoeMin,vmax=dctcoeMax)
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
