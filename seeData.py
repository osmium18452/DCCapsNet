import scipy.io as scio

# dic=scio.loadmat("data/Salinas_corrected.mat")["salinas_corrected"]
# dic=scio.loadmat("data/SalinasA_corrected.mat")["salinasA_corrected"]
dic=scio.loadmat("data/SalinasA_gt.mat")["salinasA_gt"]
print(dic)