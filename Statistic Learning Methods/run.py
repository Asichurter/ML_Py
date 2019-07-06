from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
from K_NN import KNN
import matplotlib.pyplot as plt

#code for completing UIP DTI course assignment1

data = [[['Y', False, False, 'S'], False],
        [['Y', False, False, 'G'], False],
        [['Y', True, False, 'G'], True],
        [['Y', True, True, 'S'], True],
        [['Y', False, False, 'S'], False],
        [['M', False, False, 'S'], False],
        [['M', False, False, 'G'], False],
        [['M', True, True, 'G'], True],
        [['M', False, True, 'VG'], True],
        [['M', False, True, 'VG'], True],
        [['O', False, True, 'VG'], True],
        [['O', False, True, 'G'], True],
        [['O', True, False, 'G'], True],
        [['O', True, False, 'VG'], True],
        [['O', False, False, 'S'], False]]

disease_data = [[['Y', 'U', True, True], False],
                [['Y', 'U', True, False], False],
                [['M', 'U', True, True], True],
                [['O', 'M', True, True], True],
                [['O', 'O', False, True], True],
                [['O', 'O', False, False], False],
                [['M', 'O', False, False], True],
                [['Y', 'M', True, True], False],
                [['Y', 'O', False, True], True],
                [['O', 'M', False, True], True],
                [['Y', 'M', False, False], True],
                [['M', 'M', True, False], True],
                [['M', 'U', False, True], True],
                [['O', 'M', True, False], False]]

question1_data = [[['M', False], True],
                  [['S', False], False],
                  [['D', False], True],
                  [['S', True], False],
                  [['S', False], False],
                  [['M', True], False],
                  [['D', True], True],
                  [['S', True], False]]\

que1_test_data = [['M', True],
                  ['S', True],
                  ['D', False]]

question2_data = [[[True,True,True,True,True],'S'],
                  [[True,False,True,False,False],'S'],
                  [[False,False,True,False,False],'S'],
                  [[False,False,False,True,True],'A'],
                  [[False,False,False,True,False],'A'],
                  [[True,False,False,True,True],'A'],
                  [[True,True,False,True,False],'C'],
                  [[False,True,False,True,False],'C'],
                  [[False,True,False,True,True],'C'],
                  [[True,True,False,True,True],'C']]

qua2_test_data = [[True, False, True, True, True]]

question3_data = [[[7,8],'A'],
                  [[3,3],'B'],
                  [[12,13],'C'],
                  [[6,5],'A'],
                  [[2,3],'B'],
                  [[11,13],'C'],
                  [[11,23],'D'],
                  [[13,21],'D'],
                  [[8,9],'A'],
                  [[12,24],'D'],
                  [[12,11],'C'],
                  [[3,4],'B'],
                  [[10,12],'C'],
                  [[6,7],'A']]

qua3_test_data = [[12,20],
                  [8,9],
                  [13,14],
                  [7,5],
                  [9,11]]

que1_tree = DecisionTree([['M', 'S', 'D'], [True, False]], [True, False], question1_data, criteria='ID3')
#que1_tree.print_tree()


#predicts = que1_tree.predict(que1_test_data)
qua2_naive = NaiveBayes([[True,False],[True,False],[True,False],[True,False],[True,False]], ['S','A','C'], 0)
qua2_naive.train(question2_data)
#predicts = qua2_naive.predict_groups(qua2_test_data)

qua3_knn = KNN(question3_data, k=3)
predicts = qua3_knn.predict_groups(qua3_test_data, metric='Man')

print(predicts)
plt.plot([x[0][0] for x in question3_data if x[1]=='A'],[x[0][1] for x in question3_data if x[1]=='A'], 'o', color='red', label='class A train')
plt.plot([x[0][0] for x in question3_data if x[1]=='B'],[x[0][1] for x in question3_data if x[1]=='B'], 'o', color='green', label='class B train')
plt.plot([x[0][0] for x in question3_data if x[1]=='C'],[x[0][1] for x in question3_data if x[1]=='C'], 'o', color='orange', label='class C train')
plt.plot([x[0][0] for x in question3_data if x[1]=='D'],[x[0][1] for x in question3_data if x[1]=='D'], 'o', color='blue', label='class D train')
plt.plot([qua3_test_data[i][0] for i in range(5) if predicts[i]=='A'],[qua3_test_data[i][1] for i in range(5) if predicts[i]=='A'],
         'x', color='red', label='class A test')
plt.plot([qua3_test_data[i][0] for i in range(5) if predicts[i]=='B'],[qua3_test_data[i][1] for i in range(5) if predicts[i]=='B'],
         'x', color='green', label='class B test')
plt.plot([qua3_test_data[i][0] for i in range(5) if predicts[i]=='C'],[qua3_test_data[i][1] for i in range(5) if predicts[i]=='C'],
         'x', color='orange', label='class C test')
plt.plot([qua3_test_data[i][0] for i in range(5) if predicts[i]=='D'],[qua3_test_data[i][1] for i in range(5) if predicts[i]=='D'],
         'x', color='blue', label='class D test')
plt.legend()
plt.show()
