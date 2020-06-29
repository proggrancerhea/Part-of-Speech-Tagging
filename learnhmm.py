import math
import numpy as np
import csv
import sys

def index_to_stuff(file):
    value,  tag  = 0, {}
    with open(file) as use:
        for line in use:
            k = line.strip()
            v = value
            tag[k] = v
            value += 1
    return tag

# print (word_index_dict)
def prior_matrix(file, index_of_tag):
    rows = len(file)
    # print (rows)
    # print (file)
    prior = np.zeros(len(index_of_tag), dtype=float)
    prior_add = np.ones_like(prior)
    for i in range(rows):
        # print (file[i][0])
        start_state = file[i][0].split("_")
        # print (start_state[1])
        # print (tag_index.keys())
        tag_num = index_of_tag.get(start_state[1])
        # print (tag_num)
        # print (prior[tag_num])
        prior[tag_num] += 1
    # print (prior)
    prior += prior_add
    # print (prior)
    sum_of_matrix = np.sum(prior)
    # print (sum_of_matrix)
    return (prior/sum_of_matrix)
        # print (sum_of_matrix)

def transition_matrix(index_of_tag, train):
    rows_dict = len(index_of_tag)
    a = np.zeros(( rows_dict ,  rows_dict ), dtype= float)
    a_add = np.ones_like(a)
    rows = len(train)
    for i in range(rows):
        for j in range(len(train[i])-1):
                time, time_plus = train[i][j].split('_'), train[i][j+1].split('_')
                time_index, time_plus_index = index_of_tag.get(time[1]), index_of_tag.get(time_plus[1])
                a[time_index, time_plus_index] += 1
    a += a_add
    for i in range( rows_dict ):
        sum_of_matrix = np.sum(a[i])
        a[i] = a[i]/sum_of_matrix
    return a

def emission_matrix(index_of_tag, index_of_words, train):
    rows_tag_dict, rows_words_dict, rows = len(index_of_tag), len(index_of_words), len(train)
    b = np.zeros((rows_tag_dict, rows_words_dict), dtype=  float)
    b_add = np.ones_like(b)
    for i in range(rows):
        for j in range(len(train[i])):
            probability = train[i][j].split('_')
            tag, observation = index_of_tag.get(probability[1]), index_of_words.get(probability[0])
            b[tag, observation] +=1
    b += b_add
    for i in range(len(b)):
        sum_of_matrix = np.sum(b[i])
        b[i] = b[i]/(sum_of_matrix)
    return b

def main():
    #
    train_input = sys.argv[1]
    index_to_word_input = sys.argv[2]
    index_to_tag_input = sys.argv[3]
    hmmprior_output = sys.argv[4]
    hmmemit_output = sys.argv[5]
    hmmtrans_output = sys.argv[6]
    # train_input = "toytrain.txt"
    # index_to_word_input = 'toy_index_to_word.txt'
    # index_to_tag_input = 'toy_index_to_tag.txt'
    # hmmprior_output = 'toy_hmmprior_me.txt'
    # hmmemit_output = 'toy_hmmemit_me.txt'
    # hmmtrans_output = 'toy_hmmtrans_me.txt'
    prior_out = open(hmmprior_output, "w+")
    trans_out = open(hmmtrans_output, 'w+')
    emit_out = open(hmmemit_output, "w+")
    tag_index_dict = (index_to_stuff(index_to_tag_input))
    word_index_dict = (index_to_stuff(index_to_word_input))
    with open(train_input) as f:
        train = list(csv.reader(f, delimiter=' ', quotechar='|'))
    priorMatrix = prior_matrix(train, tag_index_dict)
    transMatrix = transition_matrix(tag_index_dict, train)
    emitMatrix = emission_matrix(tag_index_dict, word_index_dict, train)
    taglength = len(tag_index_dict)
    wordlength = len(word_index_dict)
    # print (wordlength)
    # print (transMatrix)
    # print (priorMatrix)
    # print (emitMatrix)
    for i in range(taglength):
        prior_out.write(str(priorMatrix[i])+ "\n")
    for i in range(taglength):
        for j in range(taglength):
                if j!= taglength-1:
                    trans_out.write(str(transMatrix[i][j]) + " ")
                else:
                    trans_out.write(str(transMatrix[i][j]))
        trans_out.write("\n")
    for i in range(taglength):
            for j in range(wordlength):
                if j!= wordlength:
                    emit_out.write(str(emitMatrix[i][j]) + " ")
                else:
                    trans_out.write(str(emitMatrix[i][j]))
            emit_out.write("\n")

    # print (transMatrix)



    # print (prior_matrix(train, tag_index_dict))
    # print (transition_matrix(tag_index_dict, train))
    # print (emission_matrix(tag_index_dict, word_index_dict, train))


if __name__ == "__main__":
    main()