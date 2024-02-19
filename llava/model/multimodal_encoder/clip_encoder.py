import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig



class ClipVisionModel(torch.nn.Module):
    def __init__(self, model,  normalize, all_tokens=False, proj=True):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.proj = model.proj
        if all_tokens:
            self.model.output_tokens = True
        if not proj:
            self.model.proj = None

    def forward(self, vision_, output_normalize=False):
        embedding = self.model(self.normalize(vision_))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)

        if self.model.output_tokens:
            # flatten and concatenate all tokens
            return torch.hstack([embedding[0].flatten(1), embedding[1].flatten(1)])
        else:
            return embedding


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, non_llava=False, pretrained_ckpt=None, device='cuda'):

        self.non_llava = non_llava
        
        if non_llava:
            import open_clip
            print("using open_clip")
            model_orig, _, image_processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            vision_model = model_orig.visual
            if pretrained_ckpt != 'openai':
                vision_model.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu'))
            # self.image_processor = image_processor
            # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')  # 224
            # model_orig = vision_model
            # llava operates on the second to last layer output, so we remove the last layer
            vision_model.transformer.resblocks = vision_model.transformer.resblocks[:-1]; print("removing last layer of vision model")
            model_orig = ClipVisionModel(
                model=vision_model,
                normalize=lambda x: x,  # images have to be normalized, e.g. as handled by the llava model wrapper
                all_tokens=True, proj=False
            )
            self.vision_tower = model_orig        
            self.vision_tower.device = device

        else:
            print("using huggingface clip")
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        # couldn;t find a quicker way of doing this, should work for now but suboptimal
        # self.vision_tower.dtype = CLIPVisionModel.from_pretrained(self.vision_tower_name).dtype
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        
        if self.non_llava:
            image_features = image_forward_outs#.hidden_states[1:]
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    def forward(self, images):
            if type(images) is list:
                image_features = []
                for image in images:
                    if self.non_llava:
                        image_forward_out = self.vision_tower(image.to(device=self.device).unsqueeze(0)).reshape(images.shape[0], -1, 1024)
                    else:
                        image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                if self.non_llava:
                    image_forward_outs = self.vision_tower(images.to(device=self.device)).reshape(images.shape[0], -1, 1024)
                else:
                    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)


    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
