from lseg import *
from ovsi import *
from SAM import *
import random
import os
import PIL.Image as Image
import cv2
import json

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = Options().parse()

    set_seeds(args.seed)
    backbone = get_lseg_model(args)
    SAM_model, SAM_processor = get_model()

    repeat_times = 2
    use_contrastive = True

    exp_name = f'{repeat_times}_{use_contrastive}'
    os.makedirs('exp_results/', exist_ok=True)

    OVSI_model = OVSI(SAM_model, SAM_processor, backbone, None, use_contrastive=use_contrastive)

    args.benchmark = args.dataset
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    save_folder = f'exp_results/{exp_name}/{args.dataset}_{args.fold}/'
    os.makedirs(save_folder, exist_ok=True)

    results = {
        'detail': {}
    }

    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = utils.to_cuda(batch)
            image = batch['query_img']
            gt = batch['query_mask']
            class_info = batch['class_id']
            name = batch['query_name'][0].split('/')[-1].split('/')[0]

            if args.dataset == 'coco':
                image_PIL = Image.open(os.path.join(dataloader.dataset.base_path, batch['query_name'][0])).convert('RGB')
            elif args.dataset == 'pascal':
                image_PIL = Image.open(os.path.join(dataloader.dataset.img_path, batch['query_name'][0] + '.jpg')).convert('RGB')
            else:
                image_PIL = Image.open(batch['query_name'][0]).convert('RGB')

            pred_mask = OVSI_model(image, class_info, image_PIL, repeat_times)

            process_results(average_meter, pred_mask, gt, class_info, args, idx, batch, dataloader)

            iou = (pred_mask[0,1] * gt[0]).sum() / (1 - (1 - pred_mask[0,1]) * (1 - gt[0])).sum()

            cv2.imwrite(f'{save_folder}/{name}.png', pred_mask[0,1].cpu().numpy() * 255)
            cv2.imwrite(f'{save_folder}/{name}_gt.png', gt[0].cpu().numpy() * 255)

            results['detail'][name] = float(iou.cpu())

    test_miou, test_fb_iou = average_meter.compute_iou()
    results['test_miou'] = float(test_miou)
    results['test_fb_iou'] = float(test_fb_iou)

    print('Finished. fold: ', args.fold, test_miou, test_fb_iou)
    
    result_path = f'exp_results/{exp_name}.json'
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results[f'{args.dataset}_{args.fold}'] = results

    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=4)

