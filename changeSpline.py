import numpy as np
import spline
import ast, re
"""Convert to new knot values and  write out new ff file"""
#--Inputs---

FFfile = 'CG_run_OptSpline_SplineBond_Final_ff.dat'
PotName = 'Spline'
#info of old knots
cut0 = 10.
#info of new knots
cut = 10.
N = 15

OutFF = FFfile.split('ff.dat')[0] + 'cut{}_NKnots{}_ff.dat'.format(int(cut),N)
#----------

rs = np.linspace(0,cut,N)
knots = []

f = open(FFfile, 'r')
str = f.read()
f.close()

ParamDict = {} #dictionary of paramters
potential_str = [val for val in str.split('>>> POTENTIAL ') if len(val) > 0]

#reading ff file and extracting knot values
for potential in potential_str:
    s = potential.split('{') #split potential name from params
    if  PotName in s[0]:
        s = s[-1] #only take the parameters
        params = [s] #[' '.join(val.rsplit()) for val in re.split('}|,',s) if len(val)>0] 
        print ('params')
        print(params)
        for i, param in enumerate(params):
            print('param')
            print(param)
            dict_val = ast.literal_eval('{'+param)
            ParamDict.update(dict_val)

knots0 = ParamDict['Knots']
print ('Knots for {}:'.format(PotName))
print (knots0)

#getting potential values at new knots number
s0 = spline.Spline(cut0, knots0)
val = []
for r in rs:
    val.append(s0.Val(r))

knots = N * [0.]
s1 = spline.Spline(cut, knots)
s1.fitCoeff(rs, val)
knots = s1.knots.tolist()
print ("New knots: \n{}".format(knots))


#write out forcefield file
for i, val in enumerate(potential_str):
    if PotName in val: #find index of the old param string and remove it
        potential_str[i] = ''
SplineStr = []
s = PotName
s += "\n{"
s += "'Knots' : {} ".format(knots)
s += "}\n"
potential_str[i] = s
#SplineStr.append(s)
#SplineStr.extend(potential_str)
#str = SplineStr
str = [val for val in potential_str if len(val) > 0]
str = '>>> POTENTIAL '.join(str)
str = '>>> POTENTIAL ' + str
ff = open(OutFF,'w')
ff.write(str)
