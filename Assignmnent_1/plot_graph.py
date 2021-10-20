from matplotlib import pyplot as plt

xc = [1024,
2048,
4096,
8192,
16384,
32768,
65536,
131072,
262144,
524288,
1048576,
2097152,
4194304,
8388608,
16777216,
33554432]
xc = [str(i) for i in xc ]
yc = [2,
2,
4,
7,
14,
47,
58,
114,
225,
455,
1011,
2244,
4816,
10855,
21400,
39931] 
yp = [105,
89,
133,
120,
145,
154,
217,
301,
447,
817,
1541,
2951,
5820,
11541,
22981,
45926]
cmtime = [
19.65,
19.145,
20.617,
19.627,
21.363,
20.988,
23.206,
23.548,
24.045,
26.696,
30.811,
35.68,
44.241,
69.911,
114.401,
147.247
] 
cmtime = [i*1000 for i in cmtime ]
yp  = [ yp[i] + cmtime[i] for i in range(len(cmtime))] 
# print(yp)

plt.plot(xc , yp , color='r'  ,label="FPGA"  )

plt.plot(xc , yc , color='b'  ,label="CPU"  )

# plt.xscale("log")
# plt.yscale('log')
plt.legend()
plt.xlabel("array size") 
plt.ylabel("time taken (us)") 
plt.show()