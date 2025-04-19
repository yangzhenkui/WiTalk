import os
import re
import json
import torch
import torch.nn as nn
from tqdm import tqdm  # 导入 tqdm 进度条模块
import numpy as np
from strategy.evaluation.softnms import softnms_v2
from strategy.evaluation.eval_detection import ANETdetection
import datetime

class Tester(object):
    def __init__(self,
                config,
                test_dataset,
                model
                ):
        super(Tester, self).__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.checkpoint_path = config['path']['result_path']

        self.clip_length = config['dataset']['clip_length']
        self.num_classes = config['model']['num_classes']

        self.top_k = config['testing']['top_k']
        self.conf_thresh = config['testing']['conf_thresh']
        self.nms_thresh = config['testing']['nms_thresh']
        self.nms_sigma = config['testing']['nms_sigma']

        self.eval_gt = test_dataset.eval_gt
        print("eval_gt: ", self.eval_gt)
        self.id_to_action = test_dataset.id_to_action

        
    def get_all_checkpoints(self):
        all_files = os.listdir(self.checkpoint_path)
        pattern = re.compile(r".*-epoch-(\d+)\.pt$")
        valid_files = [(file, int(match.group(1))) for file in all_files if (match := pattern.match(file))]
        return [file for file, _ in sorted(valid_files, key=lambda x: x[1])] if valid_files else []



    def _to_var(self, data):
        for key, value in data.items():
            data[key] = value.unsqueeze(0).cuda()  # Directly move tensor to device
        return data


    def testing(self):

        checkpoint_files = self.get_all_checkpoints()

        if not checkpoint_files:
            print("No checkpoint files found in", self.checkpoint_path)
            return
        
        for pt_file_name in checkpoint_files:
            self.model.eval().cuda()  # 切换到 eval 模式，并将模型移到 GPU 上
            score_func = nn.Softmax(dim=-1)  # 使用 Softmax 将分类得分转换为概率

            test_files = list(self.test_dataset.dataset())  # 确保数据集可以多次遍历
        
            print(f"\nTesting model: {pt_file_name}")
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, pt_file_name)), strict=False)  # 加载模型参数
            result_dict = {}

            for file_name, data_iterator in tqdm(test_files, desc="Testing Progress", unit="file"):
                # 生成labels
                # print("file_name:", file_name)
                

                output = [[] for _ in range(self.num_classes)]  # 初始化每个类别的输出结果
                res = torch.zeros(self.num_classes, self.top_k, 3)  # 用于存储 Soft-NMS 处理后的 top-k 结果

                for clip, segment in data_iterator:
                    clip = self._to_var(clip)
                    with torch.no_grad():  # 禁用梯度计算
                        output_dict = self.model(clip)

                    loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors'][0]

                    decoded_segments = torch.cat(
                        [priors[:, :1] * self.clip_length - loc[:, :1],  # 左边界
                        priors[:, :1] * self.clip_length + loc[:, 1:]], dim=-1)  # 右边界
                    decoded_segments.clamp_(min=0, max=self.clip_length)  # 裁剪到合法范围

                    conf = score_func(conf)  # 使用 Softmax 计算分类概率
                    conf = conf.view(-1, self.num_classes).transpose(1, 0)  # 转换形状
                    conf_scores = conf.clone()  # 复制分类结果

                    # 筛选满足置信度阈值的检测结果
                    for cl in range(0, self.num_classes):  # 遍历每个类别
                        c_mask = conf_scores[cl] > self.conf_thresh  # 筛选置信度高的结果
                        scores = conf_scores[cl][c_mask]
                        if scores.size(0) == 0:  # 如果没有满足阈值的结果，跳过
                            continue
                        l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                        segments = decoded_segments[l_mask].view(-1, 2)
                        segments = segments + segment[0]  # 转换为时间区间
                        segments = torch.cat([segments, scores.unsqueeze(1)], -1)  # 拼接区间和分数
                        output[cl].append(segments)  # 保存到输出列表

                # 对每个类别应用 Soft-NMS
                sum_count = 0
                for cl in range(0, self.num_classes):
                    if len(output[cl]) == 0:
                        continue
                    tmp = torch.cat(output[cl], 0)  # 合并所有片段
                    tmp, count = softnms_v2(tmp, sigma=self.nms_sigma, top_k=self.top_k)  # 进行 Soft-NMS
                    res[cl, :count] = tmp  # 保存处理后的 top-k 结果
                    sum_count += count

                sum_count = min(sum_count, self.top_k)  # 限制最大数量
                flt = res.contiguous().view(-1, 3)
                flt = flt.view(self.num_classes, -1, 3)  # 重新组织结果

                # 生成 JSON 格式的结果
                proposal_list = []
                for cl in range(0, self.num_classes):  # 遍历每个类别
                    class_name = self.id_to_action[str(cl)]  # 获取类别名称
                    tmp = flt[cl].contiguous()
                    tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)  # 筛选有效结果
                    if tmp.size(0) == 0:
                        continue
                    # tmp = tmp.detach().cpu().numpy()
                    tmp = torch.tensor(tmp)  # 转换为 Tensor
                    for i in range(tmp.shape[0]):
                        tmp_proposal = {
                            'label': class_name,
                            'score': float(tmp[i, 2]),
                            'segment': [float(tmp[i, 0]), float(tmp[i, 1])]
                        }
                        proposal_list.append(tmp_proposal)

                result_dict[file_name] = proposal_list  # 保存视频结果

            # 保存最终结果为 JSON 文件
            output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
            json_name = "checkpoint_" + str(pt_file_name) + ".json"
            with open(os.path.join(self.checkpoint_path, json_name), "w") as out:
                json.dump(output_dict, out)
            self.eval_pr = os.path.join(self.checkpoint_path, json_name)

            self.eval()

    def eval(self):
        """
        Evaluate model performance and save a report to self.checkpoint_path directory.
        """
        # Define tIoU thresholds
        tious = np.linspace(0.5, 0.95, 10)

        # Initialize ANETdetection
        anet_detection = ANETdetection(
            ground_truth_filename=self.eval_gt,
            prediction_filename=self.eval_pr,
            subset='test',
            tiou_thresholds=tious
        )

        # Perform evaluation
        mAPs, average_mAP, ap = anet_detection.evaluate()

        

        # Prepare report content
        report_lines = []
        report_lines.append("Evaluation Report")
        report_lines.append("===================")
        report_lines.append(f"Evaluation Ground Truth: {self.eval_gt}")
        report_lines.append(f"Evaluation Predictions: {self.eval_pr}")
        report_lines.append("\nResults:")
        for (tiou, mAP) in zip(tious, mAPs):
            report_lines.append(f"mAP at tIoU {tiou:.2f}: {mAP:.4f}")
        report_lines.append(f"\nAverage mAP: {average_mAP:.4f}")
        # Convert report content to string
        report_content = "\n".join(report_lines)

        # Define report file path
        report_filename = os.path.join(self.checkpoint_path, "evaluation_report.txt")

        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 文件内容
        report_content = f"\nGenerated at: {current_time}\n\n{report_content}"

        # Save report to file
        try:
            with open(report_filename, "a") as report_file:
                report_file.write(report_content)
            print(f"Evaluation report saved to: {report_filename}")
        except Exception as e:
            print(f"Error saving evaluation report: {e}")


# if __name__ == '__main__':

#     from global_config import config
#     from model import wifiTAD, wifiTAD_config

#     model_config = wifiTAD_config(config['model']['model_set'])
#     model = wifiTAD(model_config)

#     dataset = WWADLDatasetTestSingle(dataset_dir='/root/shared-nvme/dataset/imu_30_3')
#     test = Tester(config,dataset, model)
#     test.testing()