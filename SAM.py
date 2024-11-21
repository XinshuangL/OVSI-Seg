import threading
import copy
import numpy as np
from transformers import SamModel, SamProcessor
import torch
import torch.nn.functional as F

def mask2prob_map(mask):
    probability_map = copy.deepcopy(mask)
    probability_map = probability_map + 1e-6
    probability_map = probability_map / np.sum(probability_map)    
    return probability_map

def sample_points(probability_map, N):
    points = []

    def random_sample_(points):
        flat_map = probability_map.flatten()
        sampled_idx = np.random.choice(len(flat_map), p=flat_map)
        y, x = divmod(sampled_idx, probability_map.shape[1])
        points.append((x, y))
    threads = []
    for _ in range(N):
        execute_thread = threading.Thread(target=random_sample_, args=(points,))
        threads.append(execute_thread)
        execute_thread.start()
    for execute_thread in threads:
        execute_thread.join()
    return points

def mask2points(mask, N, mode='uniform', allow_empty=True):    
    points = []
    labels = []

    if mode == 'random':        
        FG_probability_map = mask2prob_map(mask)
        FG_points = sample_points(FG_probability_map, N)
        FG_labels = [1 for _ in range(len(FG_points))]
        BG_probability_map = mask2prob_map(1-mask)
        BG_points = sample_points(BG_probability_map, N)
        BG_labels = [0 for _ in range(len(BG_points))]

        points, labels = FG_points + BG_points, FG_labels + BG_labels

    elif mode == 'uniform':
        ratio = 0.1
        try:
            box = mask2box(mask)
            x_m, y_m, x_M, y_M = box
            delta_x = x_M - x_m
            delta_y = y_M - y_m
            x_m = max(0, x_m - delta_x * ratio)
            y_m = max(0, y_m - delta_y * ratio)
            x_M = min(mask.shape[1] - 1, x_M + delta_x * ratio)
            y_M = min(mask.shape[0] - 1, y_M + delta_y * ratio)

            for x_i in range(N):
                for y_i in range(N):
                    x = round(x_i * (x_M - x_m) / (N - 1) + x_m)
                    y = round(y_i * (y_M - y_m) / (N - 1) + y_m)
                    points.append((x, y))
                    labels.append(1 if mask[y][x] > 0.5 else 0)
        except:
            if not allow_empty:
                raise NotImplementedError
            
            x_m, y_m, x_M, y_M = 0, 0, mask.shape[1] - 1, mask.shape[0] - 1

            for x_i in range(N):
                for y_i in range(N):
                    x = round(x_i * (x_M - x_m) / (N - 1) + x_m)
                    y = round(y_i * (y_M - y_m) / (N - 1) + y_m)
                    points.append((x, y))
                    labels.append(1 if (x_i+y_i)%2==0 else 0)

    else:
        raise NotImplementedError
    
    return points, labels

def mask2box(mask, threshold=0.1):
    mask = copy.deepcopy(mask).cpu().numpy()
    mask[mask < threshold] = 0
    y_ids, x_ids = np.where(mask > 0)
    return [x_ids.min(), y_ids.min(),
            x_ids.max(), y_ids.max()]

def dilate(mask, r):
    mask = F.pad(mask.unsqueeze(0).unsqueeze(0), pad=[r, r, r, r])
    return F.max_pool2d(mask, kernel_size=2*r+1, stride=1, padding=0)[0][0]

def erode(mask, r):
    return 1 - dilate(1 - mask, r)

def contrastive_sampling(pred_mask, reference_mask_pt, N, ratio=0.01):
    H, W = reference_mask_pt.shape
    L = round((H + W) / 2 * ratio)
    sampling_mask = (pred_mask - reference_mask_pt).abs()
    sampling_mask[sampling_mask < 0.1] = 0
    sampling_mask = erode(sampling_mask, L).cpu().numpy()

    prob_map = mask2prob_map(sampling_mask)
    points = sample_points(prob_map, N)
    labels = [1 if reference_mask_pt[point[1]][point[0]] > 0.5 else 0 for point in points]
    return points, labels

def get_model():
    model = SamModel.from_pretrained("facebook/sam-vit-huge").cuda()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    return model, processor

def get_embedding(image, model, processor):
    inputs = processor(image, return_tensors="pt").to(model.device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    return image_embeddings

def decode(image, image_embeddings, model, processor, points, labels):
    inputs = processor(image, 
                    input_points=[points], 
                    input_labels=[labels], 
                    return_tensors="pt").to(model.device)
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores
    return masks[0][0], scores[0][0]

def get_best_mask(image, image_embeddings, model, processor, points, labels):
    masks, scores = decode(image, image_embeddings, model, processor, points, labels)
    best_mask = masks[torch.argmax(scores)]
    mask = torch.zeros_like(best_mask, dtype=torch.float32)
    mask[best_mask] = 1
    return mask.to(model.device), max(scores)

def contrastive_prediction(image, image_embeddings, reference_mask, model, processor, initial_N=4, contrastive_N=10, allow_empty=True):
    points, labels = mask2points(reference_mask, initial_N, mode='uniform', allow_empty=allow_empty)
    reference_mask_pt = torch.tensor(reference_mask).to(model.device)
    mask, score = get_best_mask(image, image_embeddings, model, processor, points, labels)

    if reference_mask_pt.sum() > 0.5:
        best_score = score
        best_mask = copy.deepcopy(mask)
        for _ in range(contrastive_N):
            try:
                new_points, new_labels = contrastive_sampling(mask, reference_mask_pt, 1)
                points.extend(new_points)
                labels.extend(new_labels)
                mask, score = get_best_mask(image, image_embeddings, model, processor, points, labels)

                if score > best_score:
                    best_score = score
                    best_mask = copy.deepcopy(mask)
            except:
                break
        score = best_score
        mask = best_mask
    else:
        print('reference_mask is empty')
        
    return mask, score, points, labels

def convert(image, reference_mask, model, processor, initial_N, contrastive_N, allow_empty=True):
    image_embeddings = get_embedding(image, model, processor)
    mask, score, points, labels = contrastive_prediction(image, image_embeddings, reference_mask, model, processor, initial_N=initial_N, contrastive_N=contrastive_N, allow_empty=allow_empty)
    return mask, score

