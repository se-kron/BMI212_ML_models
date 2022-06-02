import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold

'''
#########################################################################################
# BMI 212 MADC
# prep_data.py
#
# Code to load and preprocess the data for k-fold validation with feature selection
# See main function at the bottom of the file for approach
#
# Author: Sebastian Kronmueller (sekron)
#########################################################################################
'''

def print_statement(modality):
    ''' Helper function to print a modality statement
    '''
    print(' ')
    print('-------------------------------------------------------------------------------------------------------------')
    print(' ')
    print('Preprocessing Modality: ',modality)
    print(' ')
    print('-------------------------------------------------------------------------------------------------------------')
    print(' ')

def calc_ratios(labels,column):
    ''' Function to calculate ratios of control variables
    '''
    ratios = np.zeros(3)
    ratios[0] = np.sum(labels[column]==0)
    ratios[1] = np.sum(labels[column]==1)
    ratios[2] = np.sum(labels[column]==2)
    ratios /= len(labels[column])

    return ratios

def load_samples_into_df(samplenames):
    ''' Function to load samples from several files into a single dataframe
    '''
    # Load the first file to initiate the DataFrame
    sample = samplenames.pop()
    df = pd.read_csv('./FIXED_bed_files/'+sample+'.bed',index_col=0).reset_index(drop=True)
    df.rename(columns={'value_scaled':sample},inplace=True)
    filenumber = 1
    print(filenumber," file loaded - sample: ", sample)
    print(df.shape)

    # load remaining files and merge
    for sample in samplenames:
        sample_df = pd.read_csv('./FIXED_bed_files/'+sample+'.bed',index_col=0).reset_index(drop=True)
        df = df.merge(sample_df,how='inner',on=['chrom','start','end'])
        df.rename(columns={'value_scaled':sample},inplace=True)
        filenumber += 1
        print(filenumber," files loaded - sample: ", sample)
        #print(sample_df.head())
        print(sample_df.shape)
        #print(df.head())
        print(df.shape)

    print(df.shape)

    return df

def sort_labels_by_modality():
    ''' Function loading all experiment labels and creating separate label files
        based on available sample files only
    '''

    labels = pd.read_csv("experiment_labels.csv")

    # Load the existing files in the "bed" folder and just keep the labels for the files we actually have
    bedfilenames = [f.split('.')[0] for f in listdir('./FIXED_bed_files/') if isfile(join('./FIXED_bed_files/', f))]
    existing_labels = pd.DataFrame(bedfilenames, columns = ['file_accession'])
    labels = pd.merge(existing_labels,labels)
    print(labels)

    # Create AD label
    labels.loc[labels['label']=='control','AD_label']= 0
    labels.loc[labels['label']=='MCI','AD_label'] = 0
    labels.loc[labels['label']=='CI','AD_label'] = 0
    labels.loc[labels['label']=='AD','AD_label'] = 1

    print(labels)

    # Split the labels into modalities
    DNase_labels = labels.loc[labels['modality']=='DNase-seq']
    DNase_labels_MF46 = DNase_labels.loc[DNase_labels['tissue']=='middle frontal area 46']

    RNA_labels = labels.loc[labels['modality']=='RNA-seq']
    long_RNA_labels = labels.loc[labels['modality']=='long read RNA-seq']

    ChIP_labels = labels.loc[labels['modality']=='ChIP-seq']
    ChIP_CTCF = ChIP_labels.loc[ChIP_labels['target']=='CTCF']
    ChIP_H3K27ac = ChIP_labels.loc[ChIP_labels['target']=='H3K27ac']
    ChIP_H3K27me3 = ChIP_labels.loc[ChIP_labels['target']=='H3K27me3']
    ChIP_H3K4me3 = ChIP_labels.loc[ChIP_labels['target']=='H3K4me3']

    # Save the relevant labels in separate files
    DNase_labels_MF46 = DNase_labels_MF46[['file_accession','label','AD_label','sex','age']]
    DNase_labels_MF46.to_csv('DNase_labels.csv',index=False)
    RNA_labels = RNA_labels[['file_accession','label','AD_label','sex','age']]
    RNA_labels.to_csv('RNA_labels.csv',index=False)
    long_RNA_labels = long_RNA_labels[['file_accession','label','AD_label','sex','age']]
    long_RNA_labels.to_csv('long_RNA_labels.csv',index=False)
    ChIP_CTCF = ChIP_CTCF[['file_accession','label','AD_label','sex','age']]
    ChIP_CTCF.to_csv('ChIP_CTCF_labels.csv',index=False)
    ChIP_H3K27ac = ChIP_H3K27ac[['file_accession','label','AD_label','sex','age']]
    ChIP_H3K27ac.to_csv('ChIP_H3K27ac_labels.csv',index=False)
    ChIP_H3K27me3 = ChIP_H3K27me3[['file_accession','label','AD_label','sex','age']]
    ChIP_H3K27me3.to_csv('ChIP_H3K27me3_labels.csv',index=False)
    ChIP_H3K4me3 = ChIP_H3K4me3[['file_accession','label','AD_label','sex','age']]
    ChIP_H3K4me3.to_csv('ChIP_H3K4me3_labels.csv',index=False)

    # Print check to ensure no file was missed
    print("Check: ", len(DNase_labels)+len(RNA_labels)+len(long_RNA_labels)+len(ChIP_CTCF)+len(ChIP_H3K27ac)+len(ChIP_H3K27me3)+len(ChIP_H3K4me3))

def split_labels_and_control_for_age(modality):
    ''' Function to load and split the labels between disease and control samples
        Also controls for age by ensuring ratio of patients <80, 80-<90 and >=90 is
        consistent between disease and control samples used
    '''

    # Get the labels of the modality
    labels = pd.read_csv(modality+'_labels.csv')

    # Prep labels for age control
    labels.loc[labels['age']=='90 or above' ,'age'] = 90
    labels['age'] = labels['age'].astype(int)
    labels['AD_label'] = labels['AD_label'].astype(int)
    labels['age_control'] = 0
    labels.loc[labels['age']>=80,'age_control']= 1
    labels.loc[labels['age']==90,'age_control']= 2

    # Prep labels for multiclass. 'control' = 0, 'MCI' = 1, 'CI' = 2, 'AD' = 3
    labels['mc_label'] = 0
    labels.loc[labels['label']=='MCI','mc_label']= 1
    labels.loc[labels['label']=='CI','mc_label']= 2
    labels.loc[labels['label']=='AD','mc_label']= 3

    labels = labels[['file_accession','AD_label','mc_label','age_control']]

    # Split into disease and control groups
    labels_positive = labels.loc[labels['AD_label']==1]
    labels_negative = labels.loc[labels['AD_label']==0]

    # Control for age - adjust labels_negative so ratios are equal to labels_positive
    ratios_pos = calc_ratios(labels_positive,'age_control')
    ratios_neg = calc_ratios(labels_negative,'age_control')
    delta = ratios_pos-ratios_neg

    while(np.any(delta<0)):
        labels_0 = labels_negative.loc[labels['age_control']==0]
        labels_1 = labels_negative.loc[labels['age_control']==1]
        labels_2 = labels_negative.loc[labels['age_control']==2]
        if delta[0]<0:
            labels_0 = labels_0.iloc[:-1,:]
        if delta[1]<0:
            labels_1 = labels_1.iloc[:-1,:]
        if delta[2]<0:
            labels_2 = labels_2.iloc[:-1,:]
        labels_negative = pd.concat([labels_0,labels_1,labels_2])
        ratios_pos = calc_ratios(labels_positive,'age_control')
        ratios_neg = calc_ratios(labels_negative,'age_control')
        delta = ratios_pos-ratios_neg

    print(labels_positive)
    print(labels_negative)
    print(ratios_pos)
    print(ratios_neg)

    print('Labels Prepared ----------------------------------------------------------------------------')

    return labels_positive, labels_negative


def generate_cv_data(modality, labels_positive, labels_negative, pvalue=True):
    ''' Function to generate folds for cross validation including feature selection
        If pvalue is true, feature selection will be performed by a ttest between disease / control on
        all features and selecting the lowest pvalues.
        If pvalue is false, feature selection will be performed by highest variance instead
    '''

    # Load data
    samplenames_positive = labels_positive['file_accession'].to_list()
    samplenames_negative = labels_negative['file_accession'].to_list()
    samplenames = samplenames_positive + samplenames_negative
    df = load_samples_into_df(samplenames)

    labels = pd.concat([labels_positive,labels_negative])
    labels = labels.set_index('file_accession')

    row_labels = df[['chrom','start','end']]
    df = df.drop(['chrom','start','end'],axis=1).transpose()

    # Merge labels / data
    df = pd.merge(df, labels,left_index=True,right_index=True)
    print(df)

    # Generate cross validation splits
    y = df['AD_label'].to_numpy()
    num_samples = len(y)
    n_splits = 5
    cv = StratifiedKFold(n_splits)

    split_indices = cv.split(np.zeros(num_samples),y) # Note that we are using a placeholder for X per the docs

    for i, (train, test) in enumerate(split_indices):
        print('Preparing fold ',str(i+1))
        print('Train indices:')
        print(train)
        print('Test indices:')
        print(test)

        fold_df = df.copy() # This might not be necessary, but ensures that there is no leakage across folds

        # Calculate features on training data only
        train_df = fold_df.iloc[train,:]
        if pvalue:
            # Split training data into positive and negative samples
            pos_train_df = train_df.loc[train_df['AD_label']==1]
            neg_train_df = train_df.loc[train_df['AD_label']==0]
            print(pos_train_df.iloc[:,:-3])
            print(neg_train_df.iloc[:,:-3])

            # Calculate pvalues on training data for this fold
            statistic, pvalues = ttest_ind(pos_train_df.iloc[:,:-3].to_numpy(dtype=np.float64),
                                         neg_train_df.iloc[:,:-3].to_numpy(dtype=np.float64),
                                         axis=0,
                                         nan_policy='omit')
        else:
            # Calculate variance of features
            variances = train_df.iloc[:,:-3].var(axis=0).to_numpy()

        # Reconstruct complete dataframe, sort by pvalues, split in test and train and save separately
        fold_labels = fold_df[['AD_label','mc_label','age_control']]
        fold_labels['test_data'] = 0
        fold_labels['test_data'].iloc[test] = 1
        fold_df = fold_df.iloc[:,:-3].transpose()
        fold_df[['chrom','start','end']] = row_labels

        if pvalue:
            fold_df['statistic'] = statistic
            fold_df['pvalue'] = pvalues
            fold_df = fold_df.sort_values('pvalue')
            frames = fold_df[['chrom','start','end','statistic','pvalue']]
            fold_df = fold_df.drop(['chrom','start','end','statistic','pvalue'],axis=1)
        else:
            fold_df['variance'] = variances
            fold_df = fold_df.sort_values('variance', ascending = False)
            frames = fold_df[['chrom','start','end','variance']]
            fold_df = fold_df.drop(['chrom','start','end','variance'],axis=1)

        fold_df = fold_df.head(10000)

        print(fold_df)
        print(frames)
        print(fold_labels)
        if pvalue:
            folder = './cv_data_pvalue_fixed/'
        else:
            folder = './cv_data_variance/'

        fold_df.to_csv(folder + modality + '_fold_' + str(i) + '_data.csv')
        frames.to_csv(folder + modality + '_fold_' + str(i) + '_frame_info.csv')
        fold_labels.to_csv(folder + modality + '_fold_' + str(i) + '_labels.csv')

def main():

    # Create labels (uncomment to use)
    # sort_labels_by_modality()

    modalities = ['ChIP_H3K27ac', 'ChIP_H3K27me3', 'ChIP_H3K4me3', 'ChIP_CTCF', 'DNase']

    # Create data with pvalue prioritization based on test data only in 5-fold train/test split
    for modality in modalities:
        print_statement(modality)
        labels_positive, labels_negative = split_labels_and_control_for_age(modality)
        generate_cv_data(modality, labels_positive, labels_negative, pvalue=True)

    print('Done.')

if __name__ == "__main__":
    main()
