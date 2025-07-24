import torch

import time
from models import *
from constants import *
import pandas as pd
import numpy as np
import copy, os, random
import matplotlib.pyplot as plt

class Organizer():
    def __init__(self, train_epoch=TRAIN_EPOCH):
        self.set_random_seed()
        self.reader = DataReader()
        self.target = TargetModel(self.reader)
        self.bar_recorder = 0
        self.last_acc = 0
        
    def exit_bar(self, acc, threshold, bar):
        if acc - self.last_acc <= threshold:
            self.bar_recorder += 1
        else:
            self.bar_recorder = 0
        self.last_acc = acc
        return self.bar_recorder > bar

    def set_random_seed(self, seed = GLOBAL_SEED):
        random.seed(seed)#random.seed(9)生成一个固定的随机数
        np.random.seed(seed)
        #print(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)#控制随机性
        torch.manual_seed(seed)#设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。每次重新运行都是固定的
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def plot_attack_performance(self, attack_recorder):
        epoch = attack_recorder["epoch"]
        acc = attack_recorder["acc"]
        #precise = attack_recorder["precise"]
        #print(attack_recorder)
        precise = attack_recorder["precision"]

        plt.figure(figsize=(20,10))

        plt.subplot(1,2,1)
        plt.plot(epoch, acc)
        #plt.vlines(train_epoch, 0, max(non_attacked_non_member_loss)+0.2, colors = "r", linestyles = "dashed")
        plt.title('Attack Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')

        plt.subplot(1,2,2)
        plt.plot(epoch, precise)
        #plt.vlines(train_epoch, 0, max(non_attacked_member_acc)+0.2, colors = "r", linestyles = "dashed")
        plt.title('Attack Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        
        plt.show()

    def centralized_training(self):
        self.target.init_parameters()
        max_acc = 0
        best_model = self.target.model.state_dict()
        for i in range(MAX_EPOCH):
            print("Starting epoch {}...".format(i))
            loss, acc = self.target.normal_epoch()
            print("Training loss = {}, training accuracy = {}".format(loss, acc))
            loss, acc = self.target.test_outcome()
            print("Test loss = {}, test accuracy = {}".format(loss, acc))
            if acc > max_acc:
                max_acc = acc
                best_model = copy.deepcopy(self.target.model.state_dict())
        torch.save(best_model, EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Centralized_best_model")

    def federated_training_basic(self, record_model=False, record_process=True):
        """
        A code sample to start a federated learning setting using this package
        使用此包启动联合学习设置的代码示例
        """
        # Initialize data frame for recording purpose为记录目的初始化数据帧
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "loss", "accuracy"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size使用给定的参数大小初始化聚合器
        aggregator = Aggregator(self.target.get_flatten_parameters(), robust_mechanism=DEFAULT_AGR)
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()

        # provide the global model as baseline model to Fang defense
        if DEFAULT_AGR == FANG:
            aggregator.agr_model_acquire(global_model)
        if DEFAULT_AGR == CONTRIBUTION_AWARE:
            aggregator.agr_model_acquire(global_model)

        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc)
        print("Global model initiated, loss={}, acc={}".format(loss, acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc)
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        max_global=0

        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()

            
            """          
            if j%10==0:
                print("j=" + str(j))
                if j <100:
                    torch.save(global_model.model.state_dict(), "./save/location/ADP-30/agg1/0" + str(j) + ".pth")
                else:
                    torch.save(global_model.model.state_dict(),"./save/location/ADP-30/agg1/"+str(j)+".pth")
            """
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()



                """
                if j<10:
                    print("j="+str(j))
                    if j <10:
                        torch.save(participants[i].model.state_dict(), "./save/location/DPSGD/part" + str(i) + "/0" + str(j) + ".pth")
                    else:
                        torch.save(participants[i].model.state_dict(),"./save/location/DPSGD/part"+str(i)+"/"+str(j)+".pth")
                """
                # Printing and recording 

                loss, acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, loss, acc)
                print("Epoch {} Participant {}, loss={}, acc={}".format(j + 1, i, loss, acc))
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            # Printing and recording
            loss, acc = global_model.test_outcome()
            if max_global<acc:
                max_global=acc

            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", loss, acc)
            print("Epoch {} Global model, loss={}, acc={}".format(j + 1, loss, acc))
        print("Max_global_acc is {}".format(max_global))

       # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().cpu().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().cpu().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Train.csv")






    def federated_training_with_poison(self, base_instance=None, base_label=None, target_instance=None, target_label=None):

        aggregator = Aggregator(self.target.get_flatten_parameters(), robust_mechanism=DEFAULT_AGR)
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()
        print("Global model initiated, loss={}, acc={}".format(loss, acc))

        if DEFAULT_AGR == FANG:
            aggregator.agr_model_acquire(global_model)
        if DEFAULT_AGR == CONTRIBUTION_AWARE:
            aggregator.agr_model_acquire(global_model)



        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        max_global = 0
        max_global_epoch=0
        result_list = []
        for epoch in range(50):
            count=0
            # 获取全局模型的参数
            global_parameters = global_model.get_flatten_parameters()
            if epoch > 15 and epoch % 3 == 0:
                print(epoch)
                participants[0].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[0].share_gradient_poison(epoch, base_instance, base_label, target_instance, attack=True)

                # Printing and recording
                loss, acc = participants[0].test_outcome()

                print("Epoch {} Participant {}, loss={}, acc={}".format(epoch + 1, 0, loss, acc))
                for i in range(1, NUMBER_OF_PARTICIPANTS):
                    participants[i].collect_parameters(global_parameters)
                    participants[i].share_gradient_poison(epoch, 0)

                    # Printing and recording
                    loss, acc = participants[i].test_outcome()

                    print("Epoch {} Participant {}, loss={}, acc={}".format(epoch + 1, i, loss, acc))

            else:

                # 遍历每个参与者
                for i in range(NUMBER_OF_PARTICIPANTS):

                    # 将全局模型参数加载到参与者的教师模型
                    participants[i].collect_parameters(global_parameters)
                    participants[i].share_gradient()

                    # 测试参与者模型性能
                    loss, acc = participants[i].test_outcome()
                    print(f"Epoch {epoch + 1}, Participant {i}, Loss: {loss}, Accuracy: {acc}")

            # 全局模型聚合更新
            global_model.apply_gradient()

            # 测试全局模型性能
            loss, acc = global_model.test_outcome()

            with torch.no_grad():
                outputs = global_model.model(target_instance)
                _, preds = torch.max(outputs, 1)
                print("+++++++++++++++++++++++++++++++")
                print(target_label)
                print(preds)
                print(base_label)
                print("+++++++++++++++++++++++++++++++")

                # if preds==base_label:
                #    count+=1
                for i in range(len(preds)):

                    percentages = torch.nn.Softmax(dim=1)(outputs)[i]
                    if i < len(preds)/2:
                        if preds[i] != target_label[0]:
                            count += 1
                        else:
                            if percentages[target_label[0]] < 0.9:
                                count += 1
                    else:
                        if percentages[target_label[0]] > 0.9:
                            count += 1
                    #print(f'[Predicted Confidence] ', end='')
                    #print(percentages)
                print(f'Accuracy={float(count / len(preds))}')
                result_list.append(float(count / len(preds)))
                print(result_list)
                print(max(result_list))
            if max_global < acc:
                max_global = acc
                max_global_epoch = epoch
            print(f"Global Epoch {epoch + 1}, Loss: {loss}, Accuracy: {acc}")

        print(f"Max_epoch is {max_global_epoch}, Max_global_acc is {max_global}")
        return result_list,max(result_list)

    def federated_training_with_distillation(self):
        total_start = time.time()
        aggregator = Aggregator(self.target.get_flatten_parameters(), robust_mechanism=DEFAULT_AGR)
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()
        print("Global model initiated, loss={}, acc={}".format(loss, acc))

        if DEFAULT_AGR == FANG:
            aggregator.agr_model_acquire(global_model)
        if DEFAULT_AGR == CONTRIBUTION_AWARE:
            aggregator.agr_model_acquire(global_model)



        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):

            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))

        max_global = 0
        max_global_epoch=0

        loss_list = []
        acc_list = []
        total_training_time = 0.0
        per_epoch_time_list = []
        per_participant_time = []

        for epoch in range(MAX_EPOCH):
            epoch_start = time.time()
            # 获取全局模型的参数
            global_parameters = global_model.get_flatten_parameters()


            if epoch>0 and epoch%10==0:
                print("epoch=" + str(epoch))
                if epoch <100:
                    torch.save(global_model.model.state_dict(), "./save/"+DEFAULT_SET+"/NONE/agg1/0" + str(epoch) + ".pth")
                    print("done")
                else:
                    torch.save(global_model.model.state_dict(),"./save/"+DEFAULT_SET+"/NONE/agg1/"+str(epoch)+".pth")

            # 遍历每个参与者
            for i in range(NUMBER_OF_PARTICIPANTS):
                participant_start = time.time()
                # 将全局模型参数加载到参与者的教师模型
                participants[i].collect_parameters(global_parameters)
                # 微调教师模型
                #participants[i].train_teacher(epochs=1, global_params=global_params)
                # 训练学生模型（基于教师模型的输出）
                #participants[i].train_student(distill_epochs=3)

                # 共享学生模型的梯度
                participants[i].share_gradient()

                # 测试参与者模型性能
                loss, acc = participants[i].test_outcome()
                print(f"Epoch {epoch + 1}, Participant {i}, Loss: {loss}, Accuracy: {acc}")
                participant_end = time.time()
                per_participant_time.append((epoch, i, participant_end - participant_start))

            agg_start = time.time()
            # 全局模型聚合更新
            global_model.apply_gradient()
            agg_end = time.time()
            agg_time = agg_end - agg_start

            # 测试全局模型性能
            loss, acc = global_model.test_outcome()
            loss_list.append(loss)
            acc_list.append(acc)
            if max_global < acc:
                max_global = acc
                max_global_epoch = epoch
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            total_training_time += epoch_duration
            per_epoch_time_list.append(epoch_duration)
            #print(f"Global Epoch {epoch + 1}, Loss: {loss}, Accuracy: {acc}")
            print(
                f"Global Epoch {epoch + 1}, Loss: {loss}, Accuracy: {acc}, Aggregation Time: {agg_time:.4f}s, Epoch Time: {epoch_duration:.4f}s")

        # 训练完成后保存最终目标模型
        torch.save(global_model.model.state_dict(), "./save/" + DEFAULT_SET + "/NONE/final/final_model.pth")
        print("Final global model saved.")

        print(f"Max_epoch is {max_global_epoch}, Max_global_acc is {max_global}")
        print(acc_list)
        print(loss_list)
        total_end = time.time()
        total_time = total_end - total_start

        print(f"Max_epoch is {max_global_epoch}, Max_global_acc is {max_global}")
        print(acc_list)
        print(loss_list)
        print(f"Total training time: {total_time:.4f} seconds")
        print(f"Average epoch time: {total_training_time / MAX_EPOCH:.4f} seconds")

        # 可选输出：每个客户端平均训练时间
        per_client_total = {i: 0.0 for i in range(NUMBER_OF_PARTICIPANTS)}
        for epoch_id, client_id, t in per_participant_time:
            per_client_total[client_id] += t
        for cid in sorted(per_client_total):
            avg_time = per_client_total[cid] / MAX_EPOCH
            print(f"Client {cid} average time per epoch: {avg_time:.4f} seconds")







