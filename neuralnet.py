#!/usr/bin/python
import sys
import math
import random
#random.seed(10)

W = [] # list of all weights
W0 = 0.1 
SGD = []
class_tuple = []
accurate = 0
inaccurate = 0
output = []
parent_training_dataset = []

def read_data(data_to_parse):
    global first_value
    global second_value
    global class_tuple
    file_txt = open(data_to_parse)
    dataset = [[]]
    attr_names = []
    lines = [line.rstrip('\n') for line in file_txt]
    for item in lines:       
        if item.startswith("@relation"):
            relation_name = item[10:]
        if ((item.startswith("@attribute")) and ("Class" not in item)):
            attr_mem = item[11:].split(" ")
            attr_names.append(attr_mem[0])
        if ((item.startswith("@attribute")) and ("Class" in item)):
            item = item.strip("}")
            item = item.split(" ")
            class_tuple = [item[3].replace(",",""),item[4]]
        elif item.startswith("@data"):
            index_of_data = lines.index("@data")
            data = lines[(index_of_data+1):]
            for i in range(len(data)):
                dataset.append(data[i].split(","))
            del dataset[0]
            for row in dataset:
                for index in range(len(row)-1):
                    row[index] = float(row[index])
    return dataset
    
def sum_attr(W, x_attributes):
    sum = 0.0
    for index in range(len(x_attributes)):
        sum +=(W[index]*1.0) * (x_attributes[index]*1.0)
    return sum

def ConstructNeuralNetworkSGD(learning_rate, epoch, dataset, fold, flag):
    global W0
    global SGD
    global W
    global parent_training_dataset

    if (flag == "train"):
        for index in range(len(dataset[0])-1): #W will have as many entries as the number of attributes
            W.append(0.1)
            SGD.append(0.0)
        W0 = 0.1
        while epoch != 0:
            epoch -= 1
            for row in dataset: 
                if (row[-1] == class_tuple[1]):
                    y = 1
                else:
                    y = 0
                x_attributes = row[0 : -1]
                net_value = W0 + sum_attr(W, x_attributes)
                print net_value
		activation_value = 1.0 / ( 1 + math.exp(-net_value))
                #error = 0.5*math.pow((y-activation_value),2)
                for index in range(len(x_attributes)): #Updating SGD and weights
                    gradient = -1.0 * (y-activation_value) * (activation_value) * (1-activation_value) * (x_attributes[index])
                    SGD[index] = gradient
                    W[index] += -1.0 * learning_rate * gradient
                W0 += -1.0 * learning_rate * -1.0 * (y-activation_value) * (activation_value) * (1-activation_value) * 1
                #print W0
             
    elif (flag == "test"):
        global accurate
        global inaccurate
        accurate = 0
        inaccurate = 0
        for row in dataset:
                index = parent_training_dataset.index(row)
                x_attributes = row[0:-1]
                net_value = W0 + sum_attr(W, x_attributes)
                activation_value = 1.0 / ( 1 + math.exp(-net_value))
                if (row[-1] == class_tuple[1]):
                    y = 1
                else:
                    y = 0
                if (activation_value > 0.5):
                    output[index] = [fold, class_tuple[1], row[-1], activation_value]
                    if (y == 1):
                        accurate += 1
                    else:
                        inaccurate += 1
                else:
                    output[index] =[fold,class_tuple[0], row[-1], activation_value]
                    if (y == 0):
                        accurate += 1
                    else:
                        inaccurate += 1 
        percentage_accuracy = (accurate*100.0)/(accurate+inaccurate)
        #print "Percentage Accuracy: " + str(percentage_accuracy) + "\nAccurate: " + str(accurate) + " Innacurate: " + str(inaccurate)        
        return percentage_accuracy
    
def stratified_cross_validation(parent_training_dataset, n, learning_rate, epoch):
    positive_training_set = []
    negative_training_set = []
    stratified_sample = []
    global W
    global SGD
    
    for row in parent_training_dataset:
        if (row[-1] == class_tuple[1]):
            positive_training_set.append(row)
        else:
            negative_training_set.append(row)
    random.shuffle(positive_training_set)
    random.shuffle(negative_training_set)
    pos_count_in_each_sample = len(positive_training_set)/n
    neg_count_in_each_sample = len(negative_training_set)/n

    for i in range(n):
        stratified_sample.append([])
        
    for i in range(n):
        stratified_sample[i] = positive_training_set[0:pos_count_in_each_sample] + negative_training_set[0:neg_count_in_each_sample]
        positive_training_set = positive_training_set[pos_count_in_each_sample:]
        negative_training_set = negative_training_set[neg_count_in_each_sample:]
    
    for i in range(max(len(positive_training_set),len(negative_training_set))):
        if (len(positive_training_set) !=0):
            stratified_sample[i].append(positive_training_set.pop())
        if (len(negative_training_set) != 0):
            stratified_sample[i].append(negative_training_set.pop())

    for i in range(len(stratified_sample)):
        random.shuffle(stratified_sample[i])

    sum = 0
    for i in range(len(stratified_sample)):
        W = []
        SGD = []
        testing_dataset = stratified_sample.pop()
        training_dataset = []
        for row in stratified_sample:
            training_dataset += row
        random.shuffle(training_dataset)
        stratified_sample.insert(0, testing_dataset)
        ConstructNeuralNetworkSGD(learning_rate, epoch, training_dataset, "", "train") #testing
        sum += ConstructNeuralNetworkSGD(learning_rate, epoch, testing_dataset, i+1, "test") #training
    #print sum/n

def plot_roc_curve(list_for_roc):
    TP = 0
    FP = 0
    last_TP = 0
    num_neg = 0
    num_pos = 0
    FPR = 0
    TPR = 0
    coord_list = []
    m = len(list_for_roc)
    list_for_roc = sorted(list_for_roc, key=lambda item: -item[1])
    for row in list_for_roc:
        if (row[0] == class_tuple[1]):
            num_pos += 1
        elif (row[0] == class_tuple[0]):
            num_neg += 1
    if (list_for_roc[0][0] == class_tuple[1]):
        TP += 1
    else:
        FP += 1
    for i in range(1,m):
        if((list_for_roc[i][1] != list_for_roc[i-1][1]) and (list_for_roc[i][0] == class_tuple[0]) and (TP > last_TP)):
            FPR = FP*1.0/num_neg
            TPR = TP*1.0/num_pos
            coord_list.append([FPR,TPR])
            last_TP = TP
        if (list_for_roc[i][0] == class_tuple[1]):
            TP += 1
        else:
            FP += 1
    FPR = FP*1.0/num_neg
    TPR = TP*1.0/num_pos
    coord_list.append([FPR,TPR])
    return coord_list

if __name__ == '__main__':
    #training set format is [[<feature vector>, classification value], ..n training instances.. ]
    #testing if Neural Network works for AND gate
    
    output = []
    testing_dataset = []

    #training_dataset = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
    #testing_dataset = [[0,0],[0,1],[1,0],[1,1]]
    '''
    n = 10
    learning_rate = 0.1
    epoch = 100
    train_file_name = "sonar.arff"
    '''
    
    train_file_name = sys.argv[1]
    n = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    epoch = int(sys.argv[4])                                     
       
    
    file_txt = open(train_file_name)
    read_data(train_file_name)
    parent_training_dataset = read_data(train_file_name) #parse the data 
    
    for index in range(len(parent_training_dataset)):
        output.append("")
    
    if (n==1):
        testing_dataset = parent_training_dataset    
        ConstructNeuralNetworkSGD(learning_rate, epoch, parent_training_dataset, n, "train") #testing
        ConstructNeuralNetworkSGD(learning_rate, epoch, testing_dataset, n, "test") #training  
    else : 
        #Stratified Cross Validation
        stratified_cross_validation(parent_training_dataset, n, learning_rate, epoch)
    ''' 
    #Output of Program
    #row = [fold, predicted class, actual class, activation/confidence value]
    print_output = output
    for row in print_output:
        row = str(row).strip("[")
        row = str(row).strip("]")
        row = str(row).replace("'","")
        row = str(row).replace(",","")
        print row
    '''
    '''
    #ROC Curve
    list_for_roc = []
    coord_list = []
    for row in output:
        list_for_roc.append([row[2],row[3]])
    coord_list = plot_roc_curve(list_for_roc)
    x = []
    y = []
    print len(coord_list)
    print "Start graph"
    for row in coord_list:
        print row[0] 
    print "Start y"
    for row in coord_list:
        print row[1] 
            
        #x.append(row[0])
        #y.append(row[1])
    #print x
    #print y
    '''
