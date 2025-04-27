import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from TAD.dataset.load_csi import get_video_info, get_class_index_map
from TAD.model.tad_model import wifitad
from TAD.model.tad_model_embeding import wifitad_text
from TAD.evaluation.softnms import softnms_v2
from TAD.config import config
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
embed_type = config['embed_type']
checkpoint = config['testing']['checkpoint_path'] + "/" + embed_type + "/"
output_path = config['testing']['output_path'] + embed_type
embed_model_name = config['embed_model_name']
print("output_path: ", output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)



    
if __name__ == '__main__':
    
    for epoch in range(20, max_epoch + 1):
        print("epoch: ",epoch)
        checkpoint_path =  checkpoint + 'checkpoint-'+ str(epoch) +'.ckpt'
        video_infos = get_video_info(config['dataset']['testing']['csi_info_path'])
        originidx_to_idx, idx_to_class = get_class_index_map()

        npy_data_path = config['dataset']['testing']['csi_data_path']

        net = wifitad_text(embed_type=embed_type, model_key=embed_model_name)
        print(checkpoint_path)
        net.load_state_dict(torch.load(checkpoint_path), strict=False)
        net.eval().cuda()
        #os.remove(checkpoint_path)
        score_func = nn.Softmax(dim=-1)

        result_dict = {}
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]

            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
            data = np.transpose(data)
            data = torch.from_numpy(data)

            output = []
            for cl in range(num_classes):
                output.append([])
            res = torch.zeros(num_classes, top_k, 3)

            for offset in offsetlist:
                clip = data[:, offset: offset + clip_length]
                clip = clip.float()
                clip = clip / 40.0
                if clip.size(1) < clip_length:
                    tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                                    96, 96]).float()
                    clip = torch.cat([clip, tmp], dim=1)
                clip = clip.unsqueeze(0).cuda()
                with torch.no_grad():
                    output_dict = net(clip)
                loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors'][0]
                decoded_segments = torch.cat(
                    [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=clip_length)

                conf = score_func(conf)
                conf = conf.view(-1, num_classes).transpose(1, 0)
                conf_scores = conf.clone()

                for cl in range(1, num_classes):
                    c_mask = conf_scores[cl] > conf_thresh
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                    segments = decoded_segments[l_mask].view(-1, 2)
                    segments = (segments + offset) / sample_fps
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                    output[cl].append(segments)
            sum_count = 0
            for cl in range(1, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(1, num_classes):
                class_name = idx_to_class[cl]
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                            float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name] = proposal_list

        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
        json_name = "checkpoint" + str(epoch) + ".json"
        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)





# import time

# def forward_one_epoch(net, clips, targets, training=True):
#     start_time = time.time()
#     clips = clips.cuda()
#     targets = [t.cuda() for t in targets]
#     print(f"Data to GPU: {time.time() - start_time:.3f}s")
    
#     inference_time = time.time()
#     if training:
#         output_dict = net(clips)
#     else:
#         with torch.no_grad():
#             output_dict = net(clips)
#     print(f"Inference: {time.time() - inference_time:.3f}s")
    
#     return output_dict

# if __name__ == '__main__':
#     for epoch in range(3, max_epoch + 1):
#         print("epoch: ", epoch)
#         checkpoint_path = checkpoint + 'checkpoint-' + str(epoch) + '.ckpt'
#         video_infos = get_video_info(config['dataset']['testing']['csi_info_path'])
#         originidx_to_idx, idx_to_class = get_class_index_map()
#         npy_data_path = config['dataset']['testing']['csi_data_path']

#         net = TriDetWithTextFusion(in_channels=config['model']['in_channels'], n_embd=256)
#         net.load_state_dict(torch.load(checkpoint_path))
#         net.eval().cuda()
#         score_func = nn.Softmax(dim=-1)

#         result_dict = {}
#         for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
#             sample_count = video_infos[video_name]['sample_count']
#             sample_fps = video_infos[video_name]['sample_fps']
#             offsetlist = list(range(0, sample_count - clip_length + 1, stride)) + \
#                          ([sample_count - clip_length] if (sample_count - clip_length) % stride else [])
            
#             data_load_time = time.time()
#             data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
#             data = np.transpose(data)
#             data = torch.from_numpy(data)
#             print(f"Data loading: {time.time() - data_load_time:.3f}s")

#             output = [[] for _ in range(num_classes)]
#             res = torch.zeros(num_classes, top_k, 3)
            
#             clip_time_total = 0
#             inference_time_total = 0
#             postproc_time_total = 0
            
#             for offset in offsetlist:
#                 clip_start = time.time()
#                 clip = data[:, offset: offset + clip_length].float() / 40.0
#                 if clip.size(1) < clip_length:
#                     tmp = torch.zeros([clip.size(0), clip_length - clip.size(1), 96, 96]).float()
#                     clip = torch.cat([clip, tmp], dim=1)
#                 clip = clip.unsqueeze(0).cuda()
#                 clip_time = time.time() - clip_start
#                 clip_time_total += clip_time

#                 inference_start = time.time()
#                 with torch.no_grad():
#                     output_dict = net(clip)
#                 inference_time = time.time() - inference_start
#                 inference_time_total += inference_time

#                 postproc_start = time.time()
#                 loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors'][0]
#                 decoded_segments = torch.cat(
#                     [priors[:, :1] * clip_length - loc[:, :1],
#                      priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
#                 decoded_segments.clamp_(min=0, max=clip_length)
#                 conf = score_func(conf).view(-1, num_classes).transpose(1, 0)
#                 conf_scores = conf.clone()

#                 for cl in range(1, num_classes):
#                     c_mask = conf_scores[cl] > conf_thresh
#                     scores = conf_scores[cl][c_mask]
#                     if scores.size(0) == 0:
#                         continue
#                     l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
#                     segments = decoded_segments[l_mask].view(-1, 2)
#                     segments = (segments + offset) / sample_fps
#                     segments = torch.cat([segments, scores.unsqueeze(1)], -1)
#                     output[cl].append(segments)
#                 postproc_time_total += time.time() - postproc_start

#             print(f"Video {video_name}:")
#             print(f"  Clip prep: {clip_time_total:.3f}s")
#             print(f"  Inference: {inference_time_total:.3f}s")
#             print(f"  Postproc: {postproc_time_total:.3f}s")

#             nms_start = time.time()
#             sum_count = 0
#             for cl in range(1, num_classes):
#                 if len(output[cl]) == 0:
#                     continue
#                 tmp = torch.cat(output[cl], 0)
#                 # print("tmp shape:", tmp.shape)
#                 tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
#                 res[cl, :count] = tmp
#                 sum_count += count
#             print(f"NMS: {time.time() - nms_start:.3f}s")

#             json_start = time.time()
#             sum_count = min(sum_count, top_k)
#             flt = res.contiguous().view(-1, 3).view(num_classes, -1, 3)
#             proposal_list = []
#             for cl in range(1, num_classes):
#                 class_name = idx_to_class[cl]
#                 tmp = flt[cl][(flt[cl][:, 2] > 0)].detach().cpu().numpy()
#                 if tmp.size == 0:
#                     continue
#                 for i in range(tmp.shape[0]):
#                     tmp_proposal = {
#                         'label': class_name,
#                         'score': float(tmp[i, 2]),
#                         'segment': [float(tmp[i, 0]), float(tmp[i, 1])]
#                     }
#                     proposal_list.append(tmp_proposal)
#             result_dict[video_name] = proposal_list
#             print(f"JSON prep: {time.time() - json_start:.3f}s")

#         json_write_start = time.time()
#         output_dict = {"version": "THUMOS14", "results": result_dict, "external_data": {}}
#         with open(os.path.join(output_path, f"checkpoint{epoch}.json"), "w") as out:
#             json.dump(output_dict, out)
#         print(f"JSON write: {time.time() - json_write_start:.3f}s")
