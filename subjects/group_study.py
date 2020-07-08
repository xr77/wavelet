#!/usr/bin/env python
# coding: utf-8

# In[8]:


from mylib import MVPA


# In[9]:


subjects = ["s105","s107"]
mask_fname = ["IPS_mask.nii"]
for i in range(len(subjects)):
    for j in range(len(mask_fname)):
        results = MVPA(subjects[i],mask_fname[j])
        print
        print "***********************************"
        print 
        print "This is my results:", results
        print
        print "***********************************"
        print


# In[ ]:




