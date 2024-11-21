from SAM import *

def logits2mask(mask):
    assert len(mask.shape) == 3
    max_map = mask.max(dim=0, keepdim=True)[0]
    mask = (mask == max_map).float()
    return mask

def semantic_convert(image, reference_mask, model, processor, initial_N, contrastive_N, allow_empty=True):
    assert len(reference_mask.shape) == 3
    # the channel 0 corresponds to "other"
    reference_mask_rest = reference_mask[1:]
    C = reference_mask_rest.shape[0]
    logits = torch.zeros_like(reference_mask, device=model.device, dtype=torch.float32) + 0.5
    score_sum = 0
    for c in range(C):
        cur_reference_mask = reference_mask_rest[c]
        cur_mask, cur_score = convert(image, cur_reference_mask, model, processor, initial_N, contrastive_N, allow_empty=allow_empty)
        logits[c+1] = cur_mask * cur_score
        score_sum += cur_score
    return logits2mask(logits), score_sum / C

def advisor2executor(img_PIL, logits, SAM_model, SAM_processor, allow_empty=False, use_contrastive=True):
    logits_image_shape = F.interpolate(logits, img_PIL.size[::-1])
    mask_image_shape = logits2mask(logits_image_shape[0])
    SAM_mask, score = semantic_convert(img_PIL, mask_image_shape, SAM_model, SAM_processor, 8, (5 if use_contrastive else 0), allow_empty=allow_empty)
    mask = F.interpolate(SAM_mask.unsqueeze(0), logits.shape[2:])
    return mask, score

def safe_squeeze0(x):
    assert len(x) == 1
    x = x[0]
    return x

def executor2advisor(mask_executor, image_features, text_features, imshape, segment_fn, ratio=0.5):
    _, C, H, W = mask_executor.shape

    image_features = safe_squeeze0(image_features)
    text_features = safe_squeeze0(text_features)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features_2D = image_features.permute(1, 0).view(1, 512, imshape[2], imshape[3])
    image_features_2D = F.interpolate(image_features_2D, (H, W))[0]

    feature_list = []
    for c in range(C):
        cur_mask = mask_executor[0, c:c+1]
        feature = (image_features_2D * cur_mask).sum(dim=(1,2)) / cur_mask.sum()
        feature_list.append(feature.unsqueeze(0))
    features = torch.cat(feature_list, dim=0)
    features = features / features.norm(dim=-1, keepdim=True)

    combined_features = features * ratio + text_features * (1 - ratio)
    combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

    logits_advisor = segment_fn([image_features], [combined_features], imshape)
    return logits_advisor

def discussion(img_PIL, logits_advisor, SAM_model, SAM_processor, image_features, text_features, imshape, segment_fn, repeat_times=1, adaptive=True, ratio=0.5, use_contrastive=True):
    best_score = -1
    count = 0
    for _ in range(repeat_times):
        mask_executor, score = advisor2executor(img_PIL, logits_advisor, SAM_model, SAM_processor, use_contrastive=use_contrastive)
        if (score < (best_score * 0.5 + 0.5)) and adaptive:
            break
        
        logits_merged = executor2advisor(mask_executor, image_features, text_features, imshape, segment_fn, ratio=ratio)

        logits_advisor = logits_merged

        final_mask = mask_executor
        best_score = score
        count += 1
    mask_executor, score = advisor2executor(img_PIL, logits_advisor, SAM_model, SAM_processor, use_contrastive=use_contrastive)
    if (score >= (best_score * 0.5 + 0.5)) or (not adaptive):
        final_mask = mask_executor
        best_score = score
        count += 1
    return final_mask
