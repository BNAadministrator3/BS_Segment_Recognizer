import numpy as np

# ensemble = [80.19,81.79,74.28,78.38,73.5]
# lstm = [59.13,59.76,60.89,59.75,51.19]
# lstmno = [58.88, 58.14, 60.2, 59.19, 51.31]
# fc = [58,56.88,60.14,59.62,52.56]
# print(np.mean(fc))
# print(np.std(fc))


from C1_network_evaluation import foldermanager

h5folder = '/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/files_h5/subjects_20'
commonfolger = foldermanager(5, h5folder)
hierachy = (('spec+regular','spec+residual','spec+inception'),(2,4,6,8,10))
commonfolger.creation_desginfolder('cnn', hierachy)