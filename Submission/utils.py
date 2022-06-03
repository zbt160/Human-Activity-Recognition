
import numpy as np
import h5py,os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def get_complete_data(data_dict,activities_arr,increment = 50,time_step = 250):
    # Combines the data by activity and labels them
    #output = (num_examples, timestep,117),labels
    ln = activities_arr.shape[0]
    label_combined = []
    subject_combined = []
    sub = 0
    first = True

    for sub in range(len(data_dict)):
        
        for act_num in activities_arr:
            print(sub,act_num)
            for inst in range(len(data_dict[sub][act_num])):
                brk = 1
                start = 0
                end = start+time_step
                if(data_dict[sub][act_num][inst].shape[0]< time_step):
                    continue
                while(brk):
                    if(end>data_dict[sub][act_num][inst].shape[0]):
                        end = data_dict[sub][act_num][inst].shape[0]
                        start = end-time_step
                        brk = 0

                    data_append = data_dict[sub][act_num][inst][start:end,:]
    #                 print(act_num,data_append.shape,data_dict[sub][act_num][inst].shape[0],end)
                    if(first == True):
                        data_combined = data_append[np.newaxis] 
                        first = False
                    else:
                        data_combined = np.append(data_combined,data_append[np.newaxis],axis = 0)
                    label_combined.append(act_num)
                    subject_combined.append(sub)
                    start = start+ increment
                    end = start+time_step
        first = True
        if(sub == 0):
            all_data_combined  = data_combined
        else:
            all_data_combined = np.append(all_data_combined,data_combined,axis = 0)
    return all_data_combined,label_combined,subject_combined



def get_complete_data2(data_dict,activities_arr,data_dir,filename,increment = 50,time_step = 250):
    # Combines the data by activity and labels them
    #output = (num_examples, timestep,117),labels
    ln = activities_arr.shape[0]
    label_combined = []
    subject_combined = []
    sub = 0
    first = True

    data_file = h5py.File(os.path.join(data_dir,filename), 'a') 
    for sub in range(len(data_dict)):
        
        for act_num in activities_arr:
            print(sub,act_num)
            for inst in range(len(data_dict[sub][act_num])):
                brk = 1
                start = 0
                end = start+time_step
                if(data_dict[sub][act_num][inst].shape[0]< time_step):
                    continue
                while(brk):
                    if(end>data_dict[sub][act_num][inst].shape[0]):
                        end = data_dict[sub][act_num][inst].shape[0]
                        start = end-time_step
                        brk = 0

                    data_append = data_dict[sub][act_num][inst][start:end,:]
                    # print(data_append[np.newaxis].shape)
                    if(first == True):
                        Xnew = data_file.create_dataset('X', data=data_append[np.newaxis],maxshape=(None,250,117))
                        first = False
                    else:
                        append_data(Xnew,data_append[np.newaxis])
                        
                    label_combined.append(act_num)
                    subject_combined.append(sub)
                    start = start+ increment
                    end = start+time_step
        
    return label_combined,subject_combined





def append_data(ds, data):
    curRows = ds.shape[0]
    newRows = data.shape[0]
    # print("append",ds.shape,curRows,newRows)
    ds.resize(curRows+newRows, axis=0)
    ds[curRows:, ...] = data


def classwise_weights(src_act_array,trg_act_array,labels_src,labels_trg_supp,A):
    O = np.zeros((src_act_array.shape[0],trg_act_array.shape[0]))
    for p in range(src_act_array.shape[0]):
        sp = np.where(labels_src ==src_act_array[p])[0]
        for q in range(trg_act_array.shape[0]):
            sq = np.where(labels_trg_supp ==trg_act_array[q])[0]
            sm = 0
            for i in sp:
                for j in sq:
                    sm = sm + A[i,j]
            O[p,q] = sm
    return O

def soft_normalization(classwiseWeights):
    W = classwiseWeights

    for q in range(W.shape[1]):
        W[:,q] = W[:,q]/np.sum(W[:,q])
    return W

def hard_normalization(classwiseWeights):

    W  =np.zeros((classwiseWeights.shape))
    for q in range(classwiseWeights.shape[1]):
        indice = np.argmax(classwiseWeights[:,q])
        W[indice,q] = 1
    return W


def convert_one_hot(arr):

    values = arr
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


# Shuffle the indices 
def shuffle_indices(indices,labels):
    ln = indices.shape[0]
    temp_array = np.arange(0,ln)
    np.random.shuffle(temp_array)
    new_indices = np.zeros(ln)
    labels_new = np.zeros(ln)
    for i in range(ln):
        new_indices[i] = indices[int(temp_array[i])]
        labels_new[i] = labels[int(temp_array[i])]
    return new_indices,labels_new