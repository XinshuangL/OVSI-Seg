from interactive_seg import *

class OVSI:
    def __init__(self, SAM_model, SAM_processor, backbone, use_contrastive=True):
        self.SAM_model = SAM_model
        self.SAM_processor = SAM_processor
        self.backbone = backbone
        self.use_contrastive = use_contrastive
        self.SAM_model.eval()
        self.backbone.eval()

    def __call__(self, image, class_info, image_PIL, repeat_times=1, adaptive=True, baseline=False):
        image_features, imshape = self.backbone.get_image_features(image)
        text_features = self.backbone.get_text_features(class_info)
        logits_advisor = self.backbone.segment(image_features, text_features, imshape)

        if baseline:
            return logits_advisor

        pred_mask = discussion(image_PIL, logits_advisor, self.SAM_model, self.SAM_processor, image_features, text_features, imshape, self.backbone.segment, repeat_times=repeat_times, adaptive=adaptive, use_contrastive=self.use_contrastive)
        
        return pred_mask
    