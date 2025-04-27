# Talk is Not Always Cheap: Promoting Wireless Sensing Models with Text Prompts

Authors: [Zhenkui Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+Z), [Zeyi Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang,+Z), [Ge Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+G), [Han Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding,+H), [Tony Xiao Han](https://arxiv.org/search/cs?searchtype=author&query=Han,+T+X), [Fei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+F)

## 📝Abstract

Wireless signal-based human sensing technologies, such as WiFi, millimeter-wave (mmWave) radar, and Radio Frequency Identification (RFID), enable the detection and interpretation of human presence, posture, and activities, thereby providing critical support for applications in public security, healthcare, and smart environments. These technologies exhibit notable advantages due to their non-contact operation and environmental adaptability; however, existing systems often fail to leverage the textual information inherent in datasets. To address this, we propose an innovative text-enhanced wireless sensing framework, WiTalk, that seamlessly integrates semantic knowledge through three hierarchical prompt strategies-label-only, brief description, and detailed action description-without requiring architectural modifications or incurring additional data costs. We rigorously validate this framework across three public benchmark datasets: XRF55 for human action recognition (HAR), and WiFiTAL and XRFV2 for WiFi temporal action localization (TAL). Experimental results demonstrate significant performance improvements: on XRF55, accuracy for WiFi, RFID, and mmWave increases by 3.9%, 2.59%, and 0.46%, respectively; on WiFiTAL, the average performance of WiFiTAD improves by 4.98%; and on XRFV2, the mean average precision gains across various methods range from 4.02% to 13.68%. Our codes have been included in [this https URL](https://github.com/yangzhenkui/WiTalk).[[paper](https://arxiv.org/abs/2504.14621)]

![image-20250427160139705](https://zkyang.oss-cn-hangzhou.aliyuncs.com/imgimage-20250427160139705.png)

## 🔖Summary

- We propose a method to enhance wireless sensing models using text labels from existing datasets, incurring no additional cost.
- We designed three hierarchical text prompt strategies with progressive semantic richness and systematically validated their performance in Wi-Fi, RFID, and mmWave action recognition and temporal action localization tasks across three large-scale public datasets.

## **📊** Dataset Link

XRF55 DataSet: https://aiotgroup.github.io/XRF55/

XRFV2 DataSet: https://github.com/aiotgroup/XRFV2

WiFiTAL DataSet：https://github.com/AVC2-UESTC/WiFiTAD

## 🏃‍♂️ Running the Code

### XRF55

Based on the modalities to be tested (WiFi, mmWave, RFID), modify the data loading directory in load_data, configure the text encoding model and prompt strategy, and set parameters such as learning rate and batch size.

```python
python main.py
```

### XRFV2

1. Modify the paths in `basic_config.json` to match your system setup. Simultaneously update the following configurations in mamba_gpu.py to match the project path settings

   ```python
   project_path = '/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2'
   dataset_root_path = '/root/shared-nvme/dataset/XRFV2'
   causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
   mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
   python_path = '/root/.conda/envs/mamba/bin/python'
   sys.path.append(project_path)
   ```

2. Adjust the parameters in **model_str_list** to specify the model to be used for training. 

   ```python
   model_str_list = [
           ('mamba', 16, 80, {'layer': 8, 'i':2}),
           # ('Transformer', 16, 80, {'layer': 8}),
           # ('Transformer', 16, 80, {'layer': 8, 'embed_type': 'Norm'}),
           # ('wifiTAD', 16, 80, {'layer': 8}),
       ]
   ```

   Once updated, execute the following command:

```python
python mamba_gpu.py
```

3. After training, update the model save path in test_model_list within **test_run_all.py**, 

   ```python
     test_model_list = [XXXXX]
   ```

   then run the following command：

```python
python script/test_run_all.py
```

The configuration of the Mamba environment can reference: https://blog.csdn.net/yyywxk/article/details/136071016

### Notes

1. To enable WiFi single-modality, simply comment out the following code in **TAD_single.py**

```python
def forward(self, input):
        x = input[self.modality]
        B, C, L = x.size()

        # to use WiFi single modality, simply comment out the text-related code.
        # x_text = self.cal_text_features_2d()
        # x = self.fusion(x_text, x)

        x = self.embedding(x)
        feats = self.backbone(x)

        out_offsets, out_cls_logits = self.head(feats)
        priors = torch.cat(self.priors, 0).to(x.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors 
        }
```

2. The difference between **trainer_dp_all.py** and **train_dp.py** lies primarily in model saving: the former saves all models, while the latter saves only the final round's model. Similarly, **test_run_all.py** tests all models, whereas **test_run.py** tests only the final round's model.

### WiFiTAL

1. Update the **WiFiTAD_text.yaml** configuration file by specifying the encoding model and prompt strategy using the embed_model_name and embed_type parameters.
2. Once the modifications are complete, run the following command to execute:

```sh
bash TAD/train_tools/tools_text.sh 0 simple clip-vit-large-patch14
```

## Citation & Acknowledgment

If you find the paper and its code uesful to your research, please use the following BibTex entry.

```latex
@misc{yang2025talkcheappromotingwireless,
      title={Talk is Not Always Cheap: Promoting Wireless Sensing Models with Text Prompts}, 
      author={Zhenkui Yang and Zeyi Huang and Ge Wang and Han Ding and Tony Xiao Han and Fei Wang},
      year={2025},
      eprint={2504.14621},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.14621}, 
}
```

The code builds upon XRF55, WiFiTAD, and XRFV2 by incorporating a text branch, and we sincerely thank them for their contributions.