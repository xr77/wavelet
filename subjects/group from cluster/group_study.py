#!/usr/bin/env python
# coding: utf-8

# In[8]:

from scipy import stats
from mylib import MVPA
import numpy as np
from scipy.stats import ttest_ind

# In[9]:


subjects = ["s105","s107","s108","s109","s110", "s113","s114","s115","s116","s117","s118","s119","s120","s121","s122","s123","s126","s128"]
mask_fname = ["insula_mask.nii"]

results_all = []
for i in range(len(subjects)):
    for j in range(len(mask_fname)):
        results_all.append( MVPA(subjects[i],mask_fname[j]))
        
results_all_array = np.asarray(results_all)
print "******************************************************************"
print results_all_array
print "******************************************************************"

t, p = stats.ttest_1samp(results_all_array,.25,axis=0)

a = results_all_array[:,0]
b = results_all_array[:,3]
t2, p2 = ttest_ind(a, b, equal_var=False)
print "******************************************************************"
print("ttest_ind_KNN: t = %g  p = %g" % (t2, p2))
print "******************************************************************"
a = results_all_array[:,1]
b = results_all_array[:,4]
t2, p2 = ttest_ind(a, b, equal_var=False)
print "******************************************************************"
print("ttest_ind_SVM: t = %g  p = %g" % (t2, p2))
print "******************************************************************"
a = results_all_array[:,2]
b = results_all_array[:,5]
t2, p2 = ttest_ind(a, b, equal_var=False)
print "******************************************************************"
print("ttest_ind_GNB: t = %g  p = %g" % (t2, p2))
print "******************************************************************"
print
print "******************************************************************"
print 
print "This is my mean prediction accuracy:", np.mean(results_all,axis=0)
print "This is my resutls significance value:" , p
print 
print "******************************************************************"
print 


# In[ ]:




