import numpy as np
import csv
import math
import sys


def main():

    test_input = sys.argv[1]
    # test_input = "testwords.txt"
    with open(test_input) as csv_file:
        test = list(csv.reader(csv_file, delimiter=' ', quotechar='|'))
    # print(test)


    pi = np.loadtxt(sys.argv[4],dtype='float64')
    pi = (pi)
    # pi = np.loadtxt("hmmprior_me.txt", dtype='float64')
    a = np.loadtxt(sys.argv[6],dtype='float64')
    a = (a)
    b = np.loadtxt(sys.argv[5],dtype='float64')
    # b = np.loadtxt("hmmemit_me.txt", dtype='float64')
    # a = np.loadtxt("hmmtrans_me.txt", dtype='float64')
    b = (b)
    metric_out = sys.argv[8]
    predict_out = sys.argv[7]
    # metric_out = "metrix.txt"
    # predict_out = "predict_me.txt"
    predict = open(predict_out, "w")
    metric = open(metric_out, "w")

    def index_to_stuff(file):
        value, length_tag, tag = 0, 0, {}
        with open(file) as use:
            for line in use:
                # k, v = line.strip().split('')
                k = line.strip()
                v = value
                tag[k] = v
                value += 1
                length_tag += 1
        return tag

    index_to_word_input = sys.argv[2]
    index_to_tag_input = sys.argv[3]
    tags_list = open(index_to_tag_input).readlines()
    word_list = open(index_to_word_input).readlines()
    # index_to_word_input = 'index_to_word.txt'
    # index_to_tag_input = 'index_to_tag.txt'
    tag_index_dict = (index_to_stuff(index_to_tag_input))
    # print(tag_index_dict)
    word_index_dict = (index_to_stuff(index_to_word_input))
    # print (word_index_dict)
    tag_list, get_tag = {}, {}
    for m in range(len(tags_list)):
        tag = tags_list[m].strip()
        tag_list[tag] = m
        get_tag[m] = tag
        # print (get_tag[m])
    def initialise(d1, d2, row_ones):
        if row_ones == 0:
            alpha = np.zeros((d1, d2), dtype=float)
            for j in range((d2)):
                alpha[0][j] += 1
            return alpha
        else:
            for j in range((d2)):
                beta = np.zeros((d1, d2), dtype=float)
                beta[d1][j] += 1
            return beta

    correct_pred, totallength = 0, 0
    num_of_samples = len(test)
    num_of_classes = len(pi)
    log_likelihood = 0
    for i in range(num_of_samples):
        sample = test[i]
        num_of_observations = len(sample)

        alpha = np.zeros((num_of_observations, num_of_classes), dtype=float)
        beta = np.zeros((num_of_observations, num_of_classes), dtype=float)
        for k in range(num_of_classes):  # ALPHA
            index = (word_index_dict.get(test[i][0].split('_')[0]))  # get index
            alpha[0][k] = (pi[k] * b[k][index])  # fill first row
            beta[len(test[i]) - 1][k] = 1
        for t in range(1, len(test[i])):
            word_index = word_index_dict.get(test[i][t].split('_')[0])  # get index for next row
            for j in range(len(pi)):
                alpha_t_minus_1 = alpha[t - 1, :]
                a_j = a[:, j]
                b_jxt = b[j, word_index]
                alpha[t, j] = b_jxt * np.sum(np.dot(alpha_t_minus_1, a_j))

                # for k_change in range(len(pi)):
                #     alpha[t][k] += (alpha[t-1][k_change]*b[k][word_index]*a[k_change][k]) #final alpha computation

        # print (alpha)

        for t in reversed(range(len(test[i]) - 1)):
            word_index_one = word_index_dict.get(test[i][t + 1].split('_')[0])
            for j in range(len(pi)):
                # summing over all k=1 to j
                beta_tplusOne = beta[t + 1][:]
                a_jk = a[j][:]
                b_kxtplus1 = b[:, word_index_one]
                # print(beta_tplusOne)
                # print(a_jk)
                # print(b_kxtplus1)
                beta[t][j] = np.sum(beta_tplusOne * a_jk * b_kxtplus1)
            # beta[t, :] = beta[t, :]/ np.sum(beta[t,:])
        # print(beta)
        pred_labels = []
        for m in range(num_of_observations):
            # print (m)
            probabilities = alpha[m, :] * beta[m, :]
            # print(probabilities)
            # print (np.shape(probabilities))
            predicted_tag = np.argmax(probabilities)
            pred_labels.append(predicted_tag)
            # print(pred_labels)

            if pred_labels[m] == tag_index_dict.get(test[i][m].split('_')[1]):
                # print(tag_index_dict.get(test[i][m].split('_')[1]))
                # print("pred labels are" + str(pred_labels[m]))
                correct_pred += 1
        totallength += len(pred_labels)
        # print((tag_index_dict.get(test[i][m].split('_')[1])))
        # print(tag_index_dict.get(test[i][m].split('_')[1]))
        # print(pred_labels)
        # for x in range(len(pred_labels)):
        predicted_words, given_words = [], []
        for labels in pred_labels:
            predicted_words.append(get_tag[labels])
            # print (predicted_words)

        for w in range(len(predicted_words)):
            # print (given_words[w])
            predict.write(str(sample[w].split("_")[0]) + "_" + str(predicted_words[w]))
            if w != (len(predicted_words) - 1):
                predict.write(" ")
        if i != num_of_samples:
            predict.write('\n')

            # print (predicted_words)
            # given_words = get_word[labels]

        log_likelihood += np.log(np.sum(alpha[-1, :]))
    predict.close()
    Average = (log_likelihood / num_of_samples)
    Accuracy = (correct_pred / totallength)
    # print (Accuracy)
    metric.write("Average Log-Likelihood: " + str(Average) + "\n")
    metric.write("Accuracy: " + str(Accuracy))
if __name__ == "__main__":
    main()

