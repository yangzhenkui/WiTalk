
import torch
from torch.utils.data import Dataset
from dataset.wwadl import WWADLDatasetSingle
from model.embedding import TextEmbedding


imu_name_to_id = {
    'gl': 0,
    'lh': 1,
    'rh': 2,
    'lp': 3,
    'rp': 4,
}

action_labels = {'0': 'Stretching', '1': 'Pouring Water', '2': 'Writing', '3': 'Cutting Fruit', '4': 'Eating Fruit', '5': 'Taking Medicine', '6': 'Drinking Water', '7': 'Sitting Down', '8': 'Turning On/Off Eye Protection Lamp', '9': 'Opening/Closing Curtains', '10': 'Opening/Closing Windows', '11': 'Typing', '12': 'Opening Envelope', '13': 'Throwing Garbage', '14': 'Picking Fruit', '15': 'Picking Up Items', '16': 'Answering Phone', '17': 'Using Mouse', '18': 'Wiping Table', '19': 'Writing on Blackboard', '20': 'Washing Hands', '21': 'Using Phone', '22': 'Reading', '23': 'Watering Plants', '24': 'Walking to Bed', '25': 'Walking to Chair', '26': 'Walking to Cabinet', '27': 'Walking to Window', '28': 'Walking to Blackboard', '29': 'Getting Out of Bed', '30': 'Standing Up', '31': 'Lying Down', '32': 'Standing Still', '33': 'Lying Still'}

# 将比例转换为实际时间
def convert_to_seconds(segment, total_duration=30):
    start_ratio, end_ratio, action_id = segment
    start_time = start_ratio * total_duration
    end_time = end_ratio * total_duration
    return start_time, end_time, action_id

# 生成描述
def generate_description(action_segments, action_labels, length=6):
    description_list = []
    
    for segment in action_segments:
        start_time, end_time, action_id = convert_to_seconds(segment)
        action_name = action_labels[str(int(action_id))]
        description_list.append(f"The user did the action of {action_name}, with a start time of {start_time:.1f} and an end time of {end_time:.1f} seconds")
    while len(description_list) < length:
        description_list.append('')

    return description_list




class WWADLDatasetMutiAll(Dataset):
    def __init__(self, dataset_dir, split="train", receivers_to_keep=None):
        """
        初始化 WWADL 数据集。
        :param dataset_dir: 数据集所在目录路径。
        :param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        if receivers_to_keep is None:
            # receivers_to_keep = {
            #     'imu': [0, 1, 2, 3, 4],
            #     'wifi': True,
            #     'airpods': True
            # }
            receivers_to_keep = {
                'imu': [],
                'wifi': True,
                'airpods': False
            }
        else:
            receivers_to_keep = {
                'imu': [imu_name_to_id[receiver] for receiver in receivers_to_keep['imu']] if receivers_to_keep['imu'] else None,
                'wifi': receivers_to_keep['wifi'],
                'airpods': receivers_to_keep['airpods']
            }
        self.imu_dataset = None
        self.wifi_dataset = None
        self.airpods_dataset = None
        self.labels = None
        if receivers_to_keep['imu']:
            self.imu_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='imu', device_keep_list=receivers_to_keep['imu'])
            self.labels = self.imu_dataset.labels
        if receivers_to_keep['wifi']:
            self.wifi_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='wifi')
            if self.labels is None:
                self.labels = self.wifi_dataset.labels
        if receivers_to_keep['airpods']:
            self.airpods_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='airpods')
            if self.labels is None:
                self.labels = self.airpods_dataset.labels
        
        self.data_len = None

    def shape(self):
        shape_info = ''
        if self.imu_dataset is not None:
            self.data_len = self.imu_dataset.shape()[0]
            shape_info += f'{self.imu_dataset.shape()}_'
        if self.wifi_dataset is not None:
            self.data_len = self.wifi_dataset.shape()[0]
            shape_info += f'{self.wifi_dataset.shape()}_'
        if self.airpods_dataset is not None:
            self.data_len = self.airpods_dataset.shape()[0]
            shape_info += f'{self.airpods_dataset.shape()}'
        return self.data_len, shape_info

    def __len__(self):
        """
        返回数据集的样本数。
        """
        if self.data_len is None:
            self.shape()
        return self.data_len

    def __getitem__(self, idx):
        data = {}
        label = None
        if self.imu_dataset is not None:
            imu_data, imu_label = self.imu_dataset[idx]
            data['imu'] = imu_data['imu']
            label = imu_label
        if self.wifi_dataset is not None:
            wifi_data, wifi_label = self.wifi_dataset[idx]
            data['wifi'] = wifi_data['wifi']
            if label is None:
                label = wifi_label
        if self.airpods_dataset is not None:
            airpods_data, airpods_label = self.airpods_dataset[idx]
            if self.imu_dataset is not None:
                data['imu'] = torch.cat((data['imu'], airpods_data['airpods']), dim=0)
            else:
                data['imu'] = airpods_data['airpods']
            if label is None:
                label = airpods_label
        # 补充文本特征
        # data['text'] = TextEmbedding(generate_description(label, action_labels))
        return data, label

