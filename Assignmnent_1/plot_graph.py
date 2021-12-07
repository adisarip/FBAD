from matplotlib import pyplot as plt

# xc = [32768,
# 65536,
# 131072,
# 262144,
# 524288,
# 1048576,
# 2097152,
# 4194304,
# 8388608,
# 16777216,
# 33554432,
# 67108864,
# 134217724]
xc = [str(1<<(11+ i)) for i in range(10) ]
# yc = [0.757626,
# 1.09857,
# 2.18333,
# 4.44422,
# 8.97795,
# 19.0334,
# 38.0451,
# 74.8772,
# 152.357,
# 303.656] 
# yp = [0.864931,
# 1.85606,
# 3.58039,
# 7.16698,
# 13.4871,
# 28.9076,
# 54.8245,
# 109.119,
# 215.357,
# 425.91]
# cmtime = [
# 19.65,
# 19.145,
# 20.617,
# 19.627,
# 21.363,
# 20.988,
# 23.206,
# 23.548,
# 24.045,
# 26.696,
# 30.811,
# 35.68,
# 44.241,
# 69.911,
# 114.401,
# 147.247
# ] 
# cmtime = [i*1000 for i in cmtime ]
# yp  = [ yp[i] + cmtime[i] for i in range(len(cmtime))] 
# print(yp)

# plt.plot(xc , yp , color='r'  ,label="FPGA time" , lw=1.5 , aa=True)

# plt.plot(xc , yc , color='b'  ,label="CPU time" ,lw=1.5,aa=True )

# plt.xscale("log")
# plt.yscale('log')
# plt.legend()
for i in xc:
    print(i)
data =[
    9.03251,
8.41835,
8.7281,
8.72055,
9.26808,
8.64824,
9.12001,
9.16434,
9.28689,
9.39165
]
plt.plot(xc, data , c='r' ,aa=True , lw=1.5)
plt.title("FPGA throughput")
plt.xlabel("array size") 
plt.ylabel("Throughput (Mb/s)")
 
plt.show()