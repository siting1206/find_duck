import cv2
import os
import math
duck_arr = []
nonduck_arr = []

#add data to array
def readimg(paths,arr):
    for path in paths:
        img = cv2.imread(path)
        arr.append(img[0, 0])

#compute mean and variance
def cal_u_var(data_arr):
    sum_b = 0.0
    sum_g = 0.0
    sum_r = 0.0

    for x in range(len(data_arr)):  # 計算各個特徵的和
        sum_b += float(data_arr[x][0])
        sum_g += float(data_arr[x][1])
        sum_r += float(data_arr[x][2])
        # print(x[0],x[1],x[2])
    # compute mean
    ub = sum_b / len(data_arr)
    ug = sum_g / len(data_arr)
    ur = sum_r / len(data_arr)

    kb = 0.0
    kg = 0.0
    kr = 0.0
    # compute variance
    for x in range(len(data_arr)):
        kb += (float(data_arr[x][0]) - ub) ** 2
        kg += (float(data_arr[x][1]) - ug) ** 2
        kr += (float(data_arr[x][2]) - ur) ** 2
    vb = kb / 49
    vg = kg / 49
    vr = kr / 49
    return ub, ug, ur, vb, vg, vr

#compute N(x|μ,σ)
def cal_P_xi(u1,u2,u3,v1,v2,v3,line_data):
    p_x1=(1/(math.sqrt(2*math.pi)*math.sqrt(v1)))*math.exp(-(float(line_data[0])-u1)**2/(2*v1))
    p_x2=(1/(math.sqrt(2*math.pi)*math.sqrt(v2)))*math.exp(-(float(line_data[1])-u2)**2/(2*v2))
    p_x3=(1/(math.sqrt(2*math.pi)*math.sqrt(v3)))*math.exp(-(float(line_data[2])-u3)**2/(2*v3))
    return p_x1,p_x2,p_x3


path_duck = 'C:/Users/user/PycharmProjects/DUCK/duck_rec/duck'
path_nonduck = 'C:/Users/user/PycharmProjects/DUCK/duck_rec/nonduck'


paths_d = []
paths_nd = []

#find all path
for dirpath,dirnames,filenames in os.walk(path_duck):
        for filename in filenames:
            paths_d.append(os.path.join(dirpath,filename))

for dirpath,dirnames,filenames in os.walk(path_nonduck):
        for filename in filenames:
            paths_nd.append(os.path.join(dirpath,filename))

readimg(paths_d, duck_arr)
readimg(paths_nd, nonduck_arr)

#data's mean and variance
duck_ub, duck_ug, duck_ur, duck_vb, duck_vg, duck_vr = cal_u_var(duck_arr)
duck_nub, duck_nug, duck_nur, duck_nvb, duck_nvg, duck_nvr = cal_u_var(nonduck_arr)
print(duck_ub, duck_ug, duck_ur, duck_vb, duck_vg, duck_vr)
print(duck_nub, duck_nug, duck_nur, duck_nvb, duck_nvg, duck_nvr)

#Priori Probability
p1=0.5
p2=0.5

test_path = 'C:/Users/user/PycharmProjects/DUCK/duck_rec/test/full_duck.jpg'
test_img = cv2.imread(test_path)
height, width, channels = test_img.shape
#compute the propability of duck/nonduck of each pixel
for x in range(height):
    for y in range(width):
        #各特徵條件機率
        P_x1_duck,P_x2_duck,P_x3_duck=cal_P_xi(duck_ub, duck_ug, duck_ur, duck_vb, duck_vg, duck_vr, test_img[x][y])
        P_x1_nonduck,P_x2_nonduck,P_x3_nonduck=cal_P_xi(duck_nub, duck_nug, duck_nur, duck_nvb, duck_nvg, duck_nvr, test_img[x][y])

        #posteriori probability
        P_duck=p1*P_x1_duck*P_x2_duck*P_x3_duck
        P_non_duck=p2*P_x1_nonduck*P_x2_nonduck*P_x3_nonduck

        #If the probability of the duck >　nonduck, we change this pixel to white, if not, black.
        if P_duck>P_non_duck:
            test_img[x][y] = [255,255,255]
        else:
            test_img[x][y] = [0,0,0]

#save and show the changed image
cv2.imwrite('outcome.jpg', test_img)
cv2.imshow('outcome', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
