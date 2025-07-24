from constants import *
import numpy as np
import copy

class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        :param sample_gradients: The tensor size of a sample gradient
        :param robust_mechanism: the robust mechanism used to aggregate the gradients
        """
        self.sample_gradients = sample_gradients.to(DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.robust = RobustMechanism(robust_mechanism)

        # AGR related parameters
        self.agr_model = None #Global model Fang and FLtrust required

    def reset(self):
        """
        Reset the aggregator to 0 before next round of aggregation
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size())
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor,source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        :param gradient: The gradient calculated by a participant
        :param souce: The source of the gradient, used for AGR verification
        :param indices: the indices of the gradient, used for AGR verification

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
                self.counter_by_indices = self.counter_by_indices.to(device)
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients.append(gradient * sample_count)
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        :param reset: Whether to reset the aggregator after getting the outcome
        :param by_indices: Whether to aggregate by indices
        """
        if by_indices:
            result = sum(self.collected_gradients) / self.counter_by_indices
        else:
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
        if reset:
            self.reset()
        return result

    def agr_model_acquire(self, model):
        """
        Make use of the given model for AGR verification
        :param model: The model used for AGR verification
        """
        self.agr_model = model
        self.robust.agr_model_acquire(model)


class RobustMechanism:
    """
    The robust aggregator applied in the aggregator
    """
    #predefined the list of participants indices and status in AGR
    appearence_list = [0,1,2,3,4]
    status_list = []

    def __init__(self, robust_mechanism):
        self.type = robust_mechanism
        if robust_mechanism is None:
            self.function = self.naive_average
        elif robust_mechanism == TRMEAN:
            self.function = self.trmean
        elif robust_mechanism ==  MULTI_KRUM:
            self.function = self.multi_krum
        elif robust_mechanism == MEDIAN:
            self.function = self.median
        elif robust_mechanism == FANG:
            self.function = self.Fang_defense
        elif robust_mechanism == CONTRIBUTION_AWARE:
            self.function = self.contribution_aware

        elif robust_mechanism == FL_TRUST:
            self.function = self.fl_trust
        self.agr_model = None


    def agr_model_acquire(self, global_model: torch.nn.Module):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        :param model: The model used for LRR and ERR verification
        """
        self.agr_model = global_model

    def contribution_aware(self, input_gradients: torch.Tensor, validation_loader, malicious_user: int = 0):
        """
        按参与者对全局模型准确率的贡献程度加权聚合
        :param input_gradients: 所有参与者的梯度
        :param validation_loader: 用于评估的全局验证集加载器
        :param malicious_user: 恶意参与者的数量（未使用）
        :return: 加权聚合后的全局梯度
        """
        ratio=0.9
        if not hasattr(self, "last_aggregated_gradient"):
            # 初始化上一个 epoch 的梯度为全局平均梯度
            self.last_aggregated_gradient = torch.mean(input_gradients, dim=0)
        print("********************************")
        print(self.last_aggregated_gradient)
        print("********************************")

        base_model = self.agr_model # 当前全局模型
        print(base_model)
        base_model_grad = torch.mean(input_gradients, dim=0)  # 计算全局平均梯度

        # 测试基准准确率

        test_loss, base_acc = base_model.test_outcome()
        #test_loss, base_acc = base_model.test_outcome_vae(0.035)
        #test_loss, base_acc = base_model.test_outcome_ratio(0.2)
        print(f"Base acc: ={base_acc:.4f}")

        # 计算每个参与者的贡献
        contributions = []
        for i, grad in enumerate(input_gradients):
            # 创建一个临时模型，应用参与者的梯度
            temp_model = copy.deepcopy(base_model)
            self.apply_gradient(temp_model, grad)

            # 测试临时模型的准确率

            test_loss, new_acc = temp_model.test_outcome()
            #test_loss, new_acc = temp_model.test_outcome_vae(0.035)
            #test_loss, new_acc = temp_model.test_outcome_ratio(0.2)
            print(f"New acc: ={new_acc:.4f}")
            contribution = new_acc - base_acc
            contributions.append(contribution)

        contributions = torch.tensor(contributions, device=input_gradients.device)
        print("contributions is ")
        print(contributions)

        # 只保留正贡献的参与者
        positive_contributions = torch.where(contributions > 0, contributions, torch.zeros_like(contributions))

        # 如果所有贡献都是非正，直接返回全局平均梯度
        if torch.sum(positive_contributions) == 0:
            print("No positive contributions detected. Using naive average.")
            _, indices = torch.topk(contributions, 1, largest=False)
            print(indices)
            mask = torch.ones_like(contributions, dtype=torch.bool)
            mask[indices] = False
            filtered_gradients = input_gradients[mask]
            filtered_contributions = contributions[mask]
            print("filtered")
            print(len(filtered_gradients))
            print(filtered_contributions)

            weights = filtered_contributions / torch.sum(filtered_contributions)
            weighted_gradients = filtered_gradients * weights[:, None]
            aggregated_gradient = torch.sum(weighted_gradients, dim=0)


            #print(self.last_aggregated_gradient)
            return self.last_aggregated_gradient-0.11*aggregated_gradient

        # 对正贡献的参与者进行归一化
        weights = positive_contributions / torch.sum(positive_contributions)
        print("weight is ")
        print(weights)

        # 加权聚合梯度
        weighted_gradients = input_gradients * weights[:, None]
        aggregated_gradient = torch.sum(weighted_gradients, dim=0)

        # 保存当前聚合的梯度
        self.last_aggregated_gradient = aggregated_gradient
        print("new gradient")
        print(self.last_aggregated_gradient)

        return aggregated_gradient



    def apply_gradient(self, glo_model, gradient):
        """
        将梯度应用到模型
        :param model: 模型
        :param gradient: 梯度
        """
        start_idx = 0
        for param in glo_model.model.parameters():
            length = param.numel()
            update = gradient[start_idx:start_idx + length].view(param.size())
            param.data.add_(update)
            start_idx += length

    def naive_average(self, input_gradients: torch.Tensor):
        """
        The naive aggregator
        :param input_gradients: The gradients collected from participants
        :return: The average of the gradients
        """
        return torch.mean(input_gradients, 0)

    def trmean(self, input_gradients, malicious_user: int):
        """
        The trimmed mean
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        sorted_updates,sorted_results  = torch.sort(input_gradients, 0)
        print(sorted_results)
        if malicious_user*2 < len(input_gradients):
            print(sorted_results[malicious_user: -malicious_user])
            return torch.mean(sorted_updates[malicious_user: -malicious_user], 0)
        else:
            return torch.mean(sorted_updates, 0)

    def median(self, input_gradients: torch.Tensor,number_of_malicious):
        """
        The median AGR
        :param input_gradients: The gradients collected from participants
        :return: The median of the gradients
        """
        return torch.median(input_gradients, 0).values

    def multi_krum(self, all_updates, n_attackers):
        """
        The multi-krum method 
        :param all_updates: The gradients collected from participants
        :param n_attackers: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        multi_k =  (self.type == MULTI_KRUM)
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
                (candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break


        print("Selected candicates = {}".format(np.array(candidate_indices)))
        RobustMechanism.appearence_list = candidate_indices
        return torch.mean(candidates, dim=0)


    def Fang_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The LRR and ERR mechanism proposed in Fang defense
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        # Get the baseline loss and accuracy without removing any of the inputs
        all_avg = torch.mean(input_gradients, 0)
        base_loss, base_acc = self.agr_model.test_gradients(all_avg)
        loss_impact = []
        err_impact = []
        # Get all the loss value and accuracy without ith input
        RobustMechanism.status_list = []
        for i in range(len(input_gradients)):
            avg_without_i = (sum(input_gradients[:i]) + sum(input_gradients[i+1:])) / (input_gradients.size(0) - 1)
            ith_loss, ith_acc = self.agr_model.test_gradients(avg_without_i)
            loss_impact.append(torch.tensor(base_loss - ith_loss))
            err_impact.append(torch.tensor(ith_acc - base_acc))
            RobustMechanism.status_list.append((i,ith_acc,ith_loss))
        loss_impact = torch.hstack(loss_impact)
        err_impact = torch.hstack(err_impact)
        loss_rank = torch.argsort(loss_impact, dim=-1)
        acc_rank = torch.argsort(err_impact, dim=-1)
        result = []
        for i in range(len(input_gradients)):
            if i in loss_rank[:-malicious_user] and i in acc_rank[:-malicious_user]:
                result.append(i)
        RobustMechanism.appearence_list = result
        return torch.mean(input_gradients[result], dim=0)


    def getter(self, gradients, malicious_user=NUMBER_OF_ADVERSARY):
        """
        The getter method applying the robust AGR
        :param gradients: The gradients collected from all participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after adding the malicious gradient
        """
        gradients = torch.vstack(gradients)
        return self.function(gradients, malicious_user)

    def fl_trust(self,input_gradients,malicious_user):
        replica = input_gradients.clone()
        grad_zero = self.agr_model.get_gradzero()
        print(grad_zero)
        grad_zero = grad_zero.unsqueeze(0)
        cos = torch.nn.CosineSimilarity(eps = 1e-5)
        relu = torch.nn.ReLU()
        norm = grad_zero.norm()
        scores = []
        for i in input_gradients:
            i = i.unsqueeze(0)
            score = cos(i, grad_zero)
            scores.append(score)
        scores = torch.tensor([item for item in scores]).to(DEVICE)
        print(scores)
        scores = relu(scores)

        grad = torch.nn.functional.normalize(replica)* norm
        grad =(grad.transpose(0, 1)*scores).transpose(0,1)
        grad =torch.sum(grad, dim=0)/scores.sum()
        return grad

