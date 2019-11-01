import numpy as np
import ast, re
import spline
"""Convert harmonic bond to spline and write out new ff file"""
#--Inputs---
FFfile = 'xp0.09_wrapped_CGMap_12_Spline_NoP_table2_ff.dat'
BondName = 'Bond'
cut = 20.
N = 20
OutFF = FFfile.split('ff.dat')[0] + 'SplineBond_cut{}_NKnots{}_ff.dat'.format(int(cut),N)
#----------

f = open(FFfile, 'r')
str = f.read()
f.close()

bondDict = {} #dictionary of bond paramters
potential_str = [val for val in str.split('>>> POTENTIAL ') if len(val) > 0]

#reading ff file and add bond parameters in to a dictionary
for potential in potential_str:
    s = potential.split('{') #split potential name from params
    if  BondName in s[0]:
        s = s[-1] #only take the parameters
        params = [' '.join(val.rsplit()) for val in re.split('}|,',s) if len(val)>0] 
        for i, param in enumerate(params):
            dict_val = ast.literal_eval('{'+param+'}')
            bondDict.update(dict_val)

print ('Bond parameters: {}'.format(bondDict))
k = bondDict['FConst']
r0 = bondDict['Dist0']

#convert to spline
rs = np.linspace(0,cut,N) 
knots = [0.] * N
vals = []
for r in rs:
	vals.append(k*(r-r0)**2)
s1 = spline.Spline(cut, knots)
s1.fitCoeff(rs, val)
knots = s1.knots.tolist()

print('%i knots between r = 0 and %5.2f:' %(N,cut))
print(knots)
print('Writing new forcefield file %s' %OutFF)

#write out forcefield file
for i, val in enumerate(potential_str):
    if BondName in val: #find index of the bond param string and remove it
        potential_str[i] = ''
SplineStr = []
s = BondName
s += "\n{"
s += "'Knots' : {} ".format(knots)
s += "}\n"
SplineStr.append(s)
SplineStr.extend(potential_str)
str = SplineStr
str = [val for val in str if len(val) > 0]
str = '>>> POTENTIAL '.join(str)
str = '>>> POTENTIAL ' + str
ff = open(OutFF,'w')
ff.write(str)
