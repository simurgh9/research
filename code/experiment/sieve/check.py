# -*- compile-command: "sage -python check.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *
from sage.modules.free_module_integer import IntegerLattice


files = [
    "paper40_237413_212828.csv",
    "paper50_576969_522364.csv",
    "paper60_1221209_1113319.csv",
    "paper70_2260926_2071820.csv",
    "paper80_3643144_3351770.csv",
    "paper90_6028468_5564035.csv"
]
PATH = '../../../saved/paper/'
PATH += files[5]


with open(PATH, 'r') as fl:
    L = [[int(x) for x in row.split(',')] for row in fl]

    
A = matrix(L).T
L = IntegerLattice(A)

v = '''
[-1274750  -997050  1181823   773396  -302989  1021105  -832360
  1182776  -312867  -174117   317947   883697   373340  -316104
  -397756   419206    73680  -667315   454859   689877  -134181
 -1197429   473405  -364472  -355255  1683744  -677044   374389
  -807033    26475  -565694  -215815   519553  -640761    14804
  -370077  -376172   567315   523484   364332   455241   -44063
  -693753  1461059  -610413  -246155   987269   437653  1192015
   -43549  -625410  -289837   176436 -1108498  -432860   -96227
   986405  1156446  -498938  1000947   669940   -13431  1070199
  -684089 -1206556  -179856   226652 -1225474  -441155 -1059597
  -436858   631266  -266591   288994   191763 -1848394   741634
  -269559   -13924   256653  -151401   284913  1150423   853757
 -1278581   925603   487154  -818629   106488  1247839]
'''
v = vector(tuple(int(x) for x in v[2:-2].split()))
print(v)
print(v in L)
