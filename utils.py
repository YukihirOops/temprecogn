import random
import os
import torch
import numpy as np
import pandas as pd
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def load_metanalysis(type_meta,metanqt,metanst):
    if type_meta == "NQT":
        hand = pd.read_csv(metanqt+"not_normalize_hand_NQT.csv"
                ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        calculation = pd.read_csv(metanqt+"not_normalize_calculation_NQT.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        listening = pd.read_csv(metanqt+"not_normalize_listening_NQT.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        mentalizing = pd.read_csv(metanqt+"not_normalize_mentalizing_NQT.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        metanalysis = list()
        metanalysis.extend([hand[~hand.isnull()],
        calculation[~calculation.isnull()],
        listening[~listening.isnull()],
        mentalizing[~mentalizing.isnull()]])
        metanalysis = np.vstack(metanalysis)
        metanalysis = torch.tensor(metanalysis).to("cuda")
        return metanalysis
        
    if type_meta == "NST":
        hand = pd.read_csv(metanst+"not_normalize_hand_NST.csv"
            ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        calculation = pd.read_csv(metanst+"not_normalize_calculation_NST.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        listening = pd.read_csv(metanst+"not_normalize_listening_NST.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        mentalizing = pd.read_csv(metanst+"not_normalize_mentalizing_NST.csv"
                    ,sep='\t',na_values='                      NaN',header=None)[0].fillna(np.nan).astype(np.float32)
        
        
        metanalysis = list()
        metanalysis.extend([hand[~hand.isnull()],
        calculation[~calculation.isnull()],
        listening[~listening.isnull()],
        mentalizing[~mentalizing.isnull()]])
        metanalysis = np.vstack(metanalysis)
        metanalysis = torch.tensor(metanalysis).to("cuda")
        return metanalysis
    if type_meta == "12":
        calculation = pd.read_csv(metanqt+"new/calculation.csv",header=None).astype(np.float32).iloc[:,0]
        default = pd.read_csv(metanqt+"new/default.csv",header=None).astype(np.float32).iloc[:,0]
        emotional = pd.read_csv(metanqt+"new/emotional.csv",header=None).astype(np.float32).iloc[:,0]
        language = pd.read_csv(metanqt+"new/language.csv",header=None).astype(np.float32).iloc[:,0]
        listening = pd.read_csv(metanqt+"new/listening.csv",header=None).astype(np.float32).iloc[:,0]
        mentalizing = pd.read_csv(metanqt+"new/mentalizing.csv",header=None).astype(np.float32).iloc[:,0]
        motor = pd.read_csv(metanqt+"new/motor.csv",header=None).astype(np.float32).iloc[:,0]
        reading = pd.read_csv(metanqt+"new/reading.csv",header=None).astype(np.float32).iloc[:,0]
        somatosensory = pd.read_csv(metanqt+"new/somatosensory.csv",header=None).astype(np.float32).iloc[:,0]
        task = pd.read_csv(metanqt+"new/task.csv",header=None).astype(np.float32).iloc[:,0]
        visual = pd.read_csv(metanqt+"new/visual.csv",header=None).astype(np.float32).iloc[:,0]
        working_memory = pd.read_csv(metanqt+"new/working_memory.csv",header=None).astype(np.float32).iloc[:,0]
        
        
        metanalysis = list()
        metanalysis.extend([calculation,
        default,
        emotional,
        language,
        listening,mentalizing,motor,reading,somatosensory,task,visual,working_memory])
        metanalysis = np.vstack(metanalysis)
        metanalysis = torch.tensor(metanalysis).to("cuda")
        return metanalysis
def load_subject_path(datapath):
    sublect_list = os.listdir(datapath)


    use_subject_num = pd.read_csv(datapath+"rest_run1234_random515_1.csv",sep='\t',header=None).iloc[0,3:34]
    group1_sub = pd.read_csv(datapath+"rest_run1234_random515_1.csv",sep='\t',header=None).values

    group2_sub = pd.read_csv(datapath+"rest_run1234_random515_2.csv",sep='\t',header=None).values
    group1_sub = pd.read_csv(datapath+"rest_run1234_random515_1.csv",sep='\t',header=None).values

    group2_sub = pd.read_csv(datapath+"rest_run1234_random515_2.csv",sep='\t',header=None).values
    group1_sub = pd.DataFrame(sublect_list)[0][:2086].str[:6].isin(group1_sub[0].astype(str))
    group2_sub = pd.DataFrame(sublect_list)[0][:2086].str[:6].isin(group2_sub[0].astype(str))
    sublect_list = pd.DataFrame(sublect_list)[0][:2086]
    group1_sub_path = sublect_list[group1_sub]
    group2_sub_path = sublect_list[group2_sub]
    
    return group1_sub_path,group2_sub_path













