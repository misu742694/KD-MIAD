import pandas as pd
import numpy as np
from constants import *

class DataReader:
    """
    The class to read data set from the given file
    """
    def __init__(self, data_set=DEFAULT_SET, label_column=LABEL_COL, batch_size=BATCH_SIZE,
                 distribution=DEFAULT_DISTRIBUTION, reserved=0, dirichlet_alpha=DIRICHLET_ALPHA):
        """
        Load the data from the given data path
        :param path: the path of csv file to load data
        :param label_column: the column index of csv file to store the labels
        :param label_size: The number of overall classes in the given data set
        """
        # load the csv file
        self.dirichlet_alpha = dirichlet_alpha
        if data_set == PURCHASE100:
            path = PURCHASE100_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)

        elif data_set == TEXAS100:
            path = TEXAS100_PATH
            self.data = np.load(path)
            self.labels = self.data['labels']
            self.data = self.data['features']
            self.labels = np.argmax(self.labels, axis=1)
            self.labels = torch.tensor(self.labels, dtype=torch.int64).to(DEVICE)
            features  = torch.tensor(self.data, dtype=torch.float).to(DEVICE)
            dataset_size = len(features)
            train_size = int(1 * dataset_size)
            indices = torch.randperm(dataset_size)

            self.data=features[indices[:train_size]]
            self.labels=self.labels[indices[:train_size]]


        self.data = self.data.to(DEVICE)
        self.labels = self.labels.to(DEVICE)


        # if there is no reserved data samples defined, then set the reserved data samples to 0
        try:
            reserved = RESERVED_SAMPLE
        except NameError:
            reserved = 0

        # if there is no FLtrust set defined, then set it to 0
        try:
            fl_trust_samples = FL_TRUST_SET
        except NameError:
            fl_trust_samples = 0

        # initialize the training and testing batches indices
        self.train_set = []
        self.test_set = []
        self.test_set1=[]
        self.batch_size=BATCH_SIZE
        self.train_set_last_batch = None
        self.test_set_last_batch =None
        overall_size = self.labels.size(0)

        if distribution is None:
            # divide data samples into batches, drop the last bit of data samples to make sure each batch is full sized
            overall_size -= 16
            rand_perm = torch.randperm(self.labels.size(0)).to(DEVICE)
            self.reserve_set = rand_perm[overall_size:]
            print("cover dataset size is {}".format(reserved))
            #initialize the fl trust set
            overall_size -= fl_trust_samples
            self.fl_trust = rand_perm[overall_size+reserved:]
            print("FL TRUST dataset size is {}".format(fl_trust_samples))
            all_size = overall_size
            overall_size -= overall_size % batch_size
            rand_perm_last = rand_perm[overall_size:all_size]
            rand_perm = rand_perm[:overall_size]
            print("cover dataset size is {}".format(reserved))
            self.last_batch_indices = self.reserve_set
            self.batch_indices = rand_perm.reshape((-1, batch_size)).to(DEVICE)
            self.train_test_split()
            print("Data set " + DEFAULT_SET +
                  " has been loaded, overall {} records, batch size = {}, testing batches: {}, training batches: {}"
                  .format(overall_size, batch_size, self.test_set.size(0), self.train_set.size(0)))
        elif distribution == DIRICHLET:
            self.perform_dirichlet_partition(alpha=self.dirichlet_alpha, batch_size=batch_size)
        elif distribution == LABEL_PARTITION:
            self.perform_label_partition(labels_per_client=10)

    def perform_dirichlet_partition(self, alpha=0.5, batch_size=BATCH_SIZE):
        num_classes = torch.max(self.labels).item() + 1
        num_clients = NUMBER_OF_PARTICIPANTS
        class_indices = [torch.where(self.labels == y)[0] for y in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c, idx in enumerate(class_indices):
            idx = idx[torch.randperm(idx.size(0))]
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            proportions = (np.cumsum(proportions) * idx.size(0)).astype(int)[:-1]
            split_indices = torch.split(idx, list(np.diff(np.concatenate(([0], proportions, [idx.size(0)])))))
            for i, split in enumerate(split_indices):
                client_indices[i].extend(split.tolist())
        for i in range(num_clients):
            client_data = torch.tensor(client_indices[i], dtype=torch.long).to(DEVICE)
            client_data = client_data[torch.randperm(client_data.size(0))]
            client_data = client_data[:(len(client_data) // batch_size) * batch_size] 
            batches = client_data.view(-1, batch_size)
            num_batches = batches.size(0)
            split_point = num_batches // 2
            self.train_set.append(batches[:split_point]) 
            self.test_set1.append(batches[split_point:]) 
        self.test_set = torch.cat(self.test_set1, dim=0).to(DEVICE)

    def perform_label_partition(self, labels_per_client=2, batch_size=BATCH_SIZE):
        num_classes = torch.max(self.labels).item() + 1
        num_clients = NUMBER_OF_PARTICIPANTS
        class_indices = [torch.where(self.labels == y)[0] for y in range(num_classes)]
        self.train_set = []
        self.test_set1 = []
        class_pool = list(range(num_classes))
        client_class_map = []
        for _ in range(num_clients):
            chosen = np.random.choice(class_pool, labels_per_client, replace=False)
            client_class_map.append(set(chosen))
        client_label_sets = [set() for _ in range(num_clients)]
        client_indices = [[] for _ in range(num_clients)]
        for class_id, idx in enumerate(class_indices):
            idx = idx[torch.randperm(idx.size(0))]
            owners = [i for i in range(num_clients) if class_id in client_class_map[i]]
            if len(owners) == 0:
                continue
            split_sizes = [len(idx) // len(owners)] * len(owners)
            split_sizes[-1] += len(idx) - sum(split_sizes)
            splits = torch.split(idx, split_sizes)
            for i, owner in enumerate(owners):
                client_indices[owner].extend(splits[i].tolist())
                client_label_sets[owner].add(class_id) 
        for i in range(num_clients):
            client_data = torch.tensor(client_indices[i], dtype=torch.long).to(DEVICE)
            if len(client_data) < batch_size:
                print(f"[Warning] Client {i} has too few samples ({len(client_data)})")
                continue
            client_data = client_data[torch.randperm(client_data.size(0))]
            usable_len = (len(client_data) // batch_size) * batch_size
            client_data = client_data[:usable_len]
            if usable_len == 0:
                print(f"[Skip] Client {i} has no full batch after filtering")
                continue
            batches = client_data.view(-1, batch_size)
            num_batches = batches.size(0)
            if num_batches < 2:
                print(f"[Skip] Client {i} has <2 batches, skipping")
                continue
            split_point = num_batches // 2
            self.train_set.append(batches[:split_point])
            self.test_set1.append(batches[split_point:])
        if len(self.test_set1) > 0:
            self.test_set = torch.cat(self.test_set1, dim=0).to(DEVICE)
        else:
            self.test_set = torch.empty(0, batch_size).to(DEVICE)
        print("\n=== Client Label Summary ===")
        for i, label_set in enumerate(client_label_sets):
            print(f"Client {i}: Labels = {sorted(label_set)} (count = {len(label_set)})")
        print(f"\nPartition complete: {len(self.train_set)} clients with training data.")
        print(self.train_set)

    def train_test_split(self, ratio=TRAIN_TEST_RATIO, batch_training=BATCH_TRAINING):
        """
        Split the data set into training set and test set according to the given ratio
        :param ratio: tuple (float, float) the ratio of train set and test set
        :param batch_training: True to train by batch, False will not
        :return: None
        """
        if batch_training:
            train_count = round(self.batch_indices.size(0) * ratio[0] / sum(ratio))
            last_count = 8
            self.train_set = self.batch_indices[:train_count].to(DEVICE)
            self.test_set = self.batch_indices[train_count:].to(DEVICE)
            self.train_set_last_batch = self.last_batch_indices[:last_count].to(DEVICE)
            self.test_set_last_batch = self.last_batch_indices[last_count:].to(DEVICE)
        else:
            train_count = round(self.data.size(0) * ratio[0] / sum(ratio))
            rand_perm = torch.randperm(self.data.size(0)).to(DEVICE)
            self.train_set = rand_perm[:train_count].to(DEVICE)
            self.test_set = rand_perm[train_count:].to(DEVICE)

    def get_train_set(self, participant_index=0):
        """
        Get the indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        if DEFAULT_DISTRIBUTION == None:
            batches_per_participant = self.train_set.size(0) // NUMBER_OF_PARTICIPANTS
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            return self.train_set[lower_bound: upper_bound]
        elif DEFAULT_DISTRIBUTION == DIRICHLET:
            #print(participant_index)
            #print(self.train_set[participant_index])
            return self.train_set[participant_index]
        elif DEFAULT_DISTRIBUTION == LABEL_PARTITION:
            #print(participant_index)
            #print(self.train_set[participant_index])
            return self.train_set[participant_index]


    def get_test_set(self, participant_index=0):
        """
        Get the indices for each test batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
        """
        if DEFAULT_DISTRIBUTION == None:
            batches_per_participant = self.test_set.size(0) // NUMBER_OF_PARTICIPANTS
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            return self.test_set[lower_bound: upper_bound]
        elif DEFAULT_DISTRIBUTION == DIRICHLET:
            #print(self.test_set1[participant_index])
            return self.test_set1[participant_index]
        elif DEFAULT_DISTRIBUTION == LABEL_PARTITION:
            #print(self.test_set1[participant_index])
            return self.test_set1[participant_index]



    def get_last_train_batch(self):
        """
        Get the last batch of training data
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.train_set_last_batch

    def get_last_test_batch(self):
        """
        Get the last batch of testing data
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        self.test_set_last_batch = self.test_set_last_batch.reshape((-1))
        return self.test_set_last_batch

    def get_batch(self, batch_indices):
        """
        Get the batch of data according to given batch indices
        :param batch_indices: tensor[BATCH_SIZE], the indices of a particular batch
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.data[batch_indices], self.labels[batch_indices]

    def get_honest_node_member(self,participant_index = 0):
        """
        Get the member sample indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        member_list = []
        train_flatten = self.train_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(participant_index):
                member_eachx, member_eachy = self.get_batch(train_flatten[j])
                member_list.append(train_flatten[j])
        member_total = torch.tensor(member_list).to(DEVICE)
        return member_total

    def get_honest_node_nonmember(self,participant_index = 0):
        """
        Get the non-member sample indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        self.train_set = torch.concat(self.train_set,self.reserve_set)

    def get_black_box_batch(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate normal batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        :return: tuple (tensor, tensor, tensor) the tensor representing the data and labels
        """
        member_count = round(attack_batch_size * member_rate)
        non_member_count = attack_batch_size - member_count
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        non_member_indices = test_flatten[2:][torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        result = result[torch.randperm(len(result))].to(DEVICE)
        return result, member_indices, non_member_indices


    def get_black_box_batch_fixed(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training, e.g. 2 member samples in same class and rest of the samples are non-member samples
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        :return: tuple (tensor, tensor, tensor) the tensor representing the data and labels
        """
        participant_member_count = 2
        member_list = []
        non_member_count = attack_batch_size - participant_member_count
        participant_nonmember_count = non_member_count // NUMBER_OF_PARTICIPANTS
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(0):
                if len(member_list) < participant_member_count:
                    member_eachx, member_eachy = self.get_batch(train_flatten[j])
                    if int(member_eachy) == 2:
                        member_list.append(train_flatten[j])

                else:
                    break
        member_indices = torch.tensor(member_list).to(DEVICE)
        member_class_list = []
        member_x,member_y = self.get_batch(member_indices)
        for i in member_y:
            member_class_list.append(i)
        same_class_list = []
        diff = 0
        for index,i in enumerate(test_flatten):
            test_x, test_y = self.get_batch(i)
            if test_y not in member_class_list:
                same_class_list.append(i)
        diff_class_test_flatten = torch.tensor(same_class_list)
        non_member_indices = diff_class_test_flatten[torch.randperm((len(diff_class_test_flatten)))[:non_member_count]].to(DEVICE)
        non_member_x,nonmember_y = self.get_batch(non_member_indices)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        return result, member_indices, non_member_indices


    def get_black_box_batch_fixed_balance_class(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training, e.g. 2 member samples in different class and rest of the samples are non-member samples
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        :return: tuple (tensor, tensor, tensor) the tensor representing the data and labels
        """
        participant_member_count = 2
        member_list = []
        non_member_count = attack_batch_size - participant_member_count
        participant_nonmember_count = non_member_count // NUMBER_OF_PARTICIPANTS
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(0):
                if len(member_list) < participant_member_count:
                    member_eachx, member_eachy = self.get_batch(train_flatten[j])
                    if int(member_eachy) == 2:
                        member_list.append(train_flatten[j])

                else:
                    break

        member_indices = torch.tensor(member_list).to(DEVICE)
        member_class_list = []
        member_x,member_y = self.get_batch(member_indices)
        for i in member_y:
            member_class_list.append(i)
        non_member_indices = test_flatten[torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        non_member_x,nonmember_y = self.get_batch(non_member_indices)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        return result, member_indices, non_member_indices

    def del_samples(self,index,train_set,last_batch):
        """
        Delete the samples from the training set
        :param index The index of the sample to be deleted
        :param train_set The training set
        :param last_batch The last batch of the training set
        ：return: tuple (tensor, tensor) the tensor representing the data and labels
        """

        flatten_set = train_set.flatten()
        if last_batch!= None:
            flatten_set = torch.cat((flatten_set,last_batch.flatten()),0)
        flatten_set = flatten_set.cpu().numpy().tolist()
        flatten_set.remove(index)
        over_train_size = len(flatten_set)
        over_full_size = over_train_size- over_train_size % BATCH_SIZE
        full_indeices = torch.tensor(flatten_set[:over_full_size]).to(DEVICE)
        train_set = full_indeices.reshape((-1, BATCH_SIZE)).to(DEVICE)
        last_indeices = torch.tensor(flatten_set[over_full_size:over_train_size]).to(DEVICE)
        last_batch = last_indeices


        return train_set,last_batch
