import numpy as np

def u(p, q, H, X, Y, x_bar, y_bar):
    return np.sum(np.power((X-x_bar),p) * np.power((Y-y_bar),q) * H, dtype=np.float64)

def huMoments(H):
    np.asarray(H, dtype=np.float64)
    row, column = H.shape
    x, y = np.arange(0, column, 1), np.arange(0,row,1)
    X, Y = np.meshgrid(x, y, sparse=False, indexing='xy')
    X += 1
    np.asarray(X,dtype=np.float64)
    Y += 1
    np.asarray(X,dtype=np.float64)
    m = np.sum(H, dtype=np.float64)
    x_bar = np.sum(X*H, dtype=np.float64) / m
    y_bar = np.sum(Y*H, dtype=np.float64) / m
    u1 = u(0,2,H,X,Y,x_bar,y_bar)
    u2 = u(0,3,H,X,Y,x_bar,y_bar)
    u3 = u(1,1,H,X,Y,x_bar,y_bar)
    u4 = u(1,2,H,X,Y,x_bar,y_bar)
    u5 = u(2,0,H,X,Y,x_bar,y_bar)
    u6 = u(2,1,H,X,Y,x_bar,y_bar)
    u7 = u(3,0,H,X,Y,x_bar,y_bar)
    moments = [u5 + u1, np.power((u5 - u1),2) + 4 * np.power(u3,2), 
    np.power((u7 - 3 * u4),2) + np.power((3 * u6 - u2), 2), np.power((u7 + u4),2) + np.power((u6 + u2), 2),
    (u7 - 3 * u4) * (u7 + u4) * (np.power((u7 + u4),2) - 3 * np.power((u6 + u2), 2)) + (3 * u6 - u2) * (u6 + u2) * (3 * np.power((u7 + u4), 2) - np.power((u6 + u2), 2)),
    (u5 - u1) * (np.power((u7 + u4),2) - np.power((u6 + u2),2)) + 4 * u3 * (u7 + u4) * (u6 + u2), 
    (3 * u6 - u2) * (u7 + u4) * (np.power((u7 + u4),2) - 3 * np.power((u6 + u2),2)) - (u7 - 3 * u4) * (u6 + u2) * (3 * np.power((u7 + u4), 2) - np.power((u6 + u2), 2))]
    return moments

if __name__ == "__main__":
    MHI = np.load('allMHIs.npy')
    row, column, num = MHI.shape
    huVectors = np.zeros((num,7))
    for i in range(num): huVectors[i,:] = huMoments(MHI[:,:,i])
    np.save('huVectors.npy', huVectors)
