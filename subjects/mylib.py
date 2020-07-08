#!/usr/bin/env python
# coding: utf-8

# ## Project 1 pilot study single subject single ROI

# In[13]:


# load all the packages and libraries going to be used
from matplotlib import pyplot as plt
import numpy as np
from mvpa2.suite import *
import os.path as op
import sklearn
import seaborn as sns
from pywt import wavedecn
from scipy import stats
from mpl_toolkits import mplot3d


# In[14]:

def MVPA(subjects, mask_fname):
    # First, let's load fMRI data, 4 runs
    bold_fname = (subjects+'/'+subjects+'.nii/2.nii',subjects+'/'+subjects+'.nii/4.nii',subjects+'/'+subjects+'.nii/6.nii',subjects+'/'+subjects+'.nii/8.nii')

    chunks = np.zeros (320)
    chunks[:80]=1
    chunks[80:160]=2
    chunks[160:240]=3
    chunks[240:320]=4

    ds= fmri_dataset(bold_fname)
    print 'sample_number=',len(ds)  # check how many samples or volumes;
    print'feature_number=',ds.nfeatures # check how many feature are there;
    print'data_info=',ds.shape # check teh 2-dimensianl dataset info;


    # In[15]:


    # Load in the mask of the ROI
    ds = fmri_dataset (bold_fname, mask=mask_fname)
    print 'sample_number=',len(ds)  # check how many samples or volumes;
    print'feature_number=',ds.nfeatures # check how many feature being used;
    print'data_info=',ds.shape # check teh 2-dimensianl dataset info;


    # In[16]:


    # explore the dataset attributes
    print 'TR_index=', ds.sa.time_indices[:5]
    print 'Actual time of TR=', ds.sa.time_coords[:5]
    print 'ori_voxel_feature=', ds.fa.voxel_indices[:5] # The first five feature, we can see the spatial info is preseverd!
    print 'voxel_size=', ds.a.voxel_eldim
    print 'volumes_dim=', ds.a.voxel_dim


    # In[17]:


    ds.a.mapper # Since the the most important feature of this toolbox is everything
                # can be reversed back. why? Becaused everything has been recored! How cool 
                # is that!
    # stripped = ds.copy(deep=False, sa=['time_coords'],fa=[],a=[])
    # Print stripped
    # Having all these attributes being part of a dataset is often a useful thing to have, but in some cases (e.g. when it
    # comes to efficiency, and/or very large datasets) one might want to have a leaner dataset with just the information
    # that is really necessary. One way to achieve this, is to strip all unwanted attributes. The Dataset class’ copy()
    # method can help with that.


    # In[18]:


    np.mean(ds)
    sns.heatmap(ds)


    # In[19]:


    # Load in the condition label file
    conditions=np.loadtxt('regressor_shifted.csv',delimiter=',')
    # cond_labels_shifted = np.zeros(cond_labels.shape)
    # cond_labels_shifted[2:] = cond_labels[:-2]
    # return cond_labels_shifted


    # In[20]:


    ds = fmri_dataset (bold_fname, mask=mask_fname, targets= conditions, chunks=chunks)
    print ds.summary()


    # In[21]:


    fig = plt.figure(figsize= (7,7))
    ax = plt.axes(projection='3d')
    vol=ds.fa.voxel_indices
    x = vol[:,0]
    y = vol[:,1]
    z = vol[:,2]
    ax.scatter3D(x,y,z, c= np.ravel(ds[0,:]),cmap='coolwarm') #ravel to conpress; Let's look at the first TR with 92 TR
    # Good demonstration that spatial info can be preserved


    # 1 = words
    # 
    # 2 = faces
    # 
    # 3 = tools/shapes
    # 
    # 4 = numbers

    # In[22]:


    words = ds.targets == 1
    faces = ds.targets == 2
    shapes = ds.targets == 3
    numbers = ds.targets == 4


    # In[23]:


    words_allTR = ds[words,:]
    faces_allTR = ds[faces,:]
    shapes_allTR = ds[shapes,:]
    numbers_allTR = ds[numbers,:]
    words_mean = np.mean(words_allTR,axis=0)
    faces_mean = np.mean(faces_allTR,axis=0)
    shapes_mean = np.mean(shapes_allTR,axis=0)
    numbers_mean = np.mean(numbers_allTR,axis=0)


    # In[24]:


    vol=ds.fa.voxel_indices
    x = vol[:,0]
    y = vol[:,1]
    z = vol[:,2]

    # Here I ploted four objects, each one has many pictures, so we goanna look at the mean pic of each.
    cantlon = [words_mean,faces_mean,shapes_mean,numbers_mean]
    cantlon_titles = ["words_mean","faces_mean","shapes_mean","numbers_mean"]

    for i in range(len(cantlon)):
        fig = plt.figure(figsize= (7,7))
        ax = plt.axes(projection='3d')
        ax.scatter3D(x,y,z, c= np.ravel(cantlon[i]),cmap='coolwarm')
        ax.set_title (cantlon_titles[i],fontsize=15)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")



    # In[25]:


    # 3D project to 2D
    fig = plt.figure(figsize= (7,7))
    ax = plt.axes()
    vol=ds.fa.voxel_indices
    x = vol[:,0]
    y = vol[:,1]
    z = vol[:,2]
    ax.scatter(x,y, c= np.ravel(words_mean),cmap='coolwarm') #take all 
    ax.set_title ('Words Mean',fontsize=15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_zlabel("z")


    # In[26]:


    orig_data = ds.a.mapper.reverse(ds.samples)
    m = fmri_dataset(mask_fname)


    # In[27]:


    orig_mask = m.a.mapper.reverse(m.samples)


    # In[28]:


    orig_mask.shape


    # In[29]:


    orig_data.shape


    # In[30]:


    new_data = orig_mask * orig_data


    # In[31]:


    new_data.shape[0]


    # In[32]:


    sum_var_all=[]
    # loop over TR
    for i in range(new_data.shape[0]):
        TR=new_data[i,:,:,:]
        coeff = pywt.wavedecn(TR, 'haar')
        len(coeff)
        len(coeff[-1])

        sum_var=[]

        for j in range(len(coeff)-1):
            level=j+1
            # coeff[-level] is now a dictionary
            aad = coeff[-level]['aad']
            ada = coeff[-level]['ada']
            daa = coeff[-level]['daa']
            add = coeff[-level]['aad']
            dad = coeff[-level]['dad']
            dda = coeff[-level]['dda']
            ddd = coeff[-level]['ddd']
            sum_var.append(aad.var() +ada.var() +daa.var()+add.var() +dad.var() +dda.var()+ddd.var())
        sum_var_all.append(sum_var)
    #print 'sum var for all TRs=',sum_var_all # each numer describes the new feature for each level, then we have 5 new features based one 5 levels.


    # In[33]:


    pl.figure(figsize=(14, 6))
    pl.subplot(121)
    plot_samples_distance(ds, sortbyattr='chunks')
    pl.title('Distances: z-scored, detrended (sorted by chunks)')
    pl.subplot(122)
    plot_samples_distance(ds, sortbyattr='targets')
    pl.title('Distances: z-scored, detrended (sorted by targets)')


    # In[34]:


    #preprocessing and get rid of resting state
    poly_detrend(ds, polyord=1, chunks_attr='chunks')
    zscore(ds, param_est=('targets', [0]))
    ds = ds[ds.sa.targets != 0]


    # In[35]:


    pl.figure(figsize=(14, 6))
    pl.subplot(121)
    plot_samples_distance(ds, sortbyattr='chunks')
    pl.title('Distances: z-scored, detrended (sorted by chunks)')
    pl.subplot(122)
    plot_samples_distance(ds, sortbyattr='targets')
    pl.title('Distances: z-scored, detrended (sorted by targets)')


    # In[36]:


    clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_results = cvte(ds)
    KNN_orig = np.mean(cv_results)
    print 'Mean accuracy KNN_Orig=', KNN_orig
    print
    # test the significance
    t, p = stats.ttest_1samp(cv_results,.25)
    print 't_value=', t
    print 'p_value_orig=', p
    print
    print 'Accuracy for each run=', cv_results.samples


    # In[37]:


    # Try different classifers-SVM
    clf = LinearCSVMC()
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_results_svm= cvte(ds)
    SVM_orig = np.mean(cv_results_svm)
    print 'Mean accuracy SVM_orig=', SVM_orig
    print
    # Test the results significance
    t_svm, p_svm= stats.ttest_1samp(cv_results_svm,.25)
    print
    print 't_value_svm=', t_svm
    print 'p_value_svm=', p_svm


    # In[38]:


    # Try different classifers-GNB
    clf = GNB()
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_results_gnb= cvte(ds)
    GNB_orig = np.mean(cv_results_gnb)
    print 'Mean accuracy GNB_orig=',GNB_orig
    print
    # Test the results significance
    t_gnb, p_gnb= stats.ttest_1samp(cv_results_gnb,.25)
    print
    print 't_new_value_gnb=', t_gnb
    print 'p_new_value_gnb=', p_gnb


    # ## Define the function to capture the spreadness of our data. 

    # In[39]:


    ds = fmri_dataset(bold_fname, mask=mask_fname, targets=conditions, chunks=chunks)
    print ds.summary()


    # In[40]:


    def cal_R_sq (v_i,x_i,y_i,z_i):
        """Define the function that is gonna be used for the feature extration"""
        x_c = sum(np.abs(v_i)*x_i)/sum(np.abs(v_i))
        y_c = sum(np.abs(v_i)*y_i)/sum(np.abs(v_i))
        z_c = sum(np.abs(v_i)*z_i)/sum(np.abs(v_i))
        R_2 = sum(((x_i-x_c)**2+(y_i-y_c)**2)*np.abs(v_i))/sum(np.abs(v_i))
        return R_2


    # In[41]:


    # turn the list above into array
    new_five_feature = np.asarray(sum_var_all)
    data_five_feature = np.hstack([ds, new_five_feature])
    # Create a new dataset after adding in the additional label
    ds_new = dataset_wizard(data_five_feature, targets=conditions, chunks=chunks)
    ds_new.shape
    print ds.shape
    print new_five_feature.shape


    # In[42]:


    print new_five_feature


    # In[43]:


    pl.figure(figsize=(14, 6))
    pl.subplot(121)
    plot_samples_distance(ds_new, sortbyattr='chunks')
    pl.title('Distances: z-scored, detrended (sorted by chunks)')
    pl.subplot(122)
    plot_samples_distance(ds_new, sortbyattr='targets')
    pl.title('Distances: z-scored, detrended (sorted by targets)')

    #preprocessing on the dataset with new features
    poly_detrend(ds_new, polyord=1, chunks_attr='chunks')
    zscore(ds_new, param_est=('targets', [0]))
    ds_new = ds_new[ds_new.sa.targets != 0]

    pl.figure(figsize=(14, 6))
    pl.subplot(121)
    plot_samples_distance(ds_new, sortbyattr='chunks')
    pl.title('Distances: z-scored, detrended (sorted by chunks)')
    pl.subplot(122)
    plot_samples_distance(ds_new, sortbyattr='targets')
    pl.title('Distances: z-scored, detrended (sorted by targets)')


    # In[44]:


    # Classification based on new feature added
    clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_new_results = cvte(ds_new)
    KNN_new = np.mean(cv_new_results)
    print 'Mean Accuracy KNN_new=', KNN_new
    # Test the results significance
    print
    t_new, p_new = stats.ttest_1samp(cv_new_results,.25)
    print 't_new_value=', t_new
    print 'p_new_value=', p_new
    print
    print 'Accuracy for each run=', cv_new_results.samples


    # In[45]:


    # Try different classifers-SVM
    clf = LinearCSVMC()
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_results_svm= cvte(ds_new)
    SVM_new = np.mean(cv_results_svm)
    print 'Mean accuracy SVM=', SVM_new
    print
    # Test the results significance
    t_new_svm, p_new_svm= stats.ttest_1samp(cv_new_results,.25)
    print
    print 't_new_value_svm=', t_new_svm
    print 'p_new_value_svm=', p_new_svm


    # In[46]:


    # Try different classifers-GNB
    clf = GNB()
    cvte = CrossValidation(clf, NFoldPartitioner(),errorfx=lambda p, t: np.mean(p == t))
    cv_results_gnb= cvte(ds_new)
    GNB_new = np.mean(cv_results_gnb)
    print 'Mean accuracy GNB=', GNB_new
    print
    # Test the results significance
    t_new_gnb, p_new_gnb= stats.ttest_1samp(cv_results_gnb,.25)
    print
    print 't_new_value_gnb=', t_new_gnb
    print 'p_new_value_gnb=', p_new_gnb


    # In[47]:


    # a= fmri_dataset('IPS_mask.nii')
    # a.samples[:,a.fa['voxel_indices'][:,0] < 25] = 0
    # nimg = map2nifti(a)
    # nimg.to_filename('masktest.nii')


    # 

    # In[48]:


    a= fmri_dataset('IPS_mask.nii')
    #a.samples[:,a.fa['voxel_indices'][:,0] < 25] = 0
    #nimg = map2nifti(a)
    #nimg.to_filename('masktest.nii')
    # plt.hist(a.samples)


    # In[49]:


    a.samples;
    np.sum(a.samples == 2);
    np.sum(a.samples == 1);
    return  KNN_orig,SVM_orig,GNB_orig,KNN_new,SVM_new,GNB_new

