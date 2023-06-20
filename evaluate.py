def evaluate1(participant_answers, correct_answers):
    assert len(participant_answers) == len(correct_answers), "wanted to evaluate two different sized answers"
    TP = TN = FP = FN = 0
    for i in range(len(participant_answers)):
        if correct_answers[i] == 1 and participant_answers[i] == 1:
            TP += 1
        if correct_answers[i] == 1 and participant_answers[i] == 0:
            FN += 1
        if correct_answers[i] == 0 and participant_answers[i] == 0:
            TN += 1
        if correct_answers[i] == 0 and participant_answers[i] == 1:
            FP += 1
    accuracy = (TP + TN) / (TN + TP + FP + FN + 0.0000000000000000000001)
    precision = TP / (TP + FP + 0.0000000000000000000001)   
    recall = TP / (TP + FN + 0.0000000000000000000001)
    balancedf = 2 * precision * recall / (precision + recall + 0.0000000000000000000001)

    return {'precision' : precision, 
            'recall' : recall, 
            'balancedf' : balancedf, 
            'accuracy' : accuracy}

def evaluate2(participant_answers, correct_answers):
    assert len(participant_answers) == len(correct_answers), "wanted to evaluate two different sized answers"
    
    TP0=0
    TP1=0
    TP2=0
    TN0=0
    TN1=0
    TN2=0
    FN0=0
    FN1=0
    FN2=0
    FP0=0
    FP1=0
    FP2=0

    for index in range(len(participant_answers)):
        if correct_answers[index]==0 and participant_answers[index]==0:
            TP0 += 1
        
        if correct_answers[index]!=0 and participant_answers[index]==0: 
            FP0 += 1
        
        if correct_answers[index]==0 and participant_answers[index]!=0: 
            FN0 += 1
        
        if correct_answers[index]!=0 and participant_answers[index]!=0: 
            TN0 += 1
        
        if correct_answers[index]==1 and participant_answers[index]==1:
            TP1 += 1
        
        if correct_answers[index]!=1 and participant_answers[index]==1: 
            FP1 += 1
        
        if correct_answers[index]==1 and participant_answers[index]!=1: 
            FN1 += 1
        
        if correct_answers[index]!=1 and participant_answers[index]!=1: 
            TN1 += 1
        
        if correct_answers[index]==2 and participant_answers[index]==2:
            TP2 += 1
        
        if correct_answers[index]!=2 and participant_answers[index]==2: 
            FP2 += 1
        
        if correct_answers[index]==2 and participant_answers[index]!=2: 
            FN2 += 1
        
        if correct_answers[index]!=2 and participant_answers[index]!=2: 
            TN2 += 1

    # micro averages
    microAveragePrecision = (TP0+TP1+TP2)/(TP0+TP1+TP2+FP0+FP1+FP2+0.0000000000000000000001)
    microAverageRecall = (TP0+TP1+TP2)/(TP0+TP1+TP2+FN0+FN1+FN2+0.0000000000000000000001)
    microAverageFscore = 2*microAveragePrecision*microAverageRecall/(microAveragePrecision+microAverageRecall+0.0000000000000000000001)

    accuracy0=(TP0+TN0)/(TN0+TP0+FP0+FN0+0.0000000000000000000001)
    precision0=TP0/(TP0+FP0+0.0000000000000000000001)
    recall0=TP0/(TP0+FN0+0.0000000000000000000001)
    # balancedf0=2*precision0*recall0/(precision0+recall0+0.0000000000000000000001)

    accuracy1=(TP1+TN1)/(TN1+TP1+FP1+FN1+0.0000000000000000000001)
    precision1=TP1/(TP1+FP1+0.0000000000000000000001)
    recall1=TP1/(TP1+FN1+0.0000000000000000000001)
    # balancedf1=2*precision1*recall1/(precision1+recall1+0.0000000000000000000001)

    accuracy2=(TP2+TN2)/(TN2+TP2+FP2+FN2+0.0000000000000000000001)
    precision2=TP2/(TP2+FP2+0.0000000000000000000001)
    recall2=TP2/(TP2+FN2+0.0000000000000000000001)
    # balancedf2=2*precision2*recall2/(precision2+recall2+0.0000000000000000000001)

    macroAveragePrecision = (precision0+precision1+precision2)/3
    macroAverageRecall = (recall0+recall1+recall2)/3
    macroAverageFscore = 2*macroAveragePrecision*macroAverageRecall/(macroAveragePrecision+macroAverageRecall+0.0000000000000000000001)

    return {'microAverageFscore' : microAverageFscore, 'macroAverageFscore' : macroAverageFscore}
