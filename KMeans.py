import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.groups = []

    def train(self, x):
        if not self.groups:
            for i in range(self.k):
                self.groups.append(x[np.random.randint(0, len(x))])

        self.groups = np.array(self.groups, dtype=np.float)

        groups_x_instances = {}
        old_groups_x_instances = {}
        while 1:
            groups_x_instances.clear()
            for idx, instance in enumerate(x):
                group_index = 0
                instance_distance = 1e10000
                for i in range(len(self.groups)):
                    curr_distance = sum((instance - self.groups[i]) ** 2)
                    if curr_distance < instance_distance:
                        group_index = i
                        instance_distance = curr_distance

                if group_index not in groups_x_instances:
                    groups_x_instances[group_index] = [instance]
                else:
                    groups_x_instances[group_index].append(instance)

            if old_groups_x_instances.keys() == groups_x_instances.keys():
                equal_groups = []
                for index in old_groups_x_instances:
                    if np.array_equal(old_groups_x_instances[index], groups_x_instances[index]):
                        equal_groups.append(True)
                    else:
                        equal_groups.append(False)

                if all(equal_groups):
                    break

            old_groups_x_instances = groups_x_instances.copy()

            for i in range(len(self.groups)):
                for j in range(len(self.groups[i])):
                    dim_sum = 0
                    for value in groups_x_instances[i]:
                        dim_sum += value[j]

                    self.groups[i][j] = dim_sum/len(groups_x_instances[i])


if __name__ == '__main__':
    instances = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [2, 1, 1],
        [3, 2, 2],
        [3, 4, 3],
        [4, 4, 4]
    ])

    km = KMeans(2)
    km.train(instances)
