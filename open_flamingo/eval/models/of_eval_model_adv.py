import os.path
from typing import List

from PIL import Image
import torch
import torch.nn.functional as F

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms
from contextlib import suppress
from open_flamingo.eval.models.utils import unwrap_model, get_label
from torchvision.transforms import transforms


# adversarial eval model
# adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/models/open_flamingo.py

class EvalModelAdv(BaseEvalModel):
    """OpenFlamingo adversarial model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args, adversarial):
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
            and "precision" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, vision_encoder_pretrained, and precision arguments to be specified"

        self.device = (
            model_args["device"]
            if ("device" in model_args and model_args["device"] >= 0)
            else "cpu"
        )
        self.model_args = model_args
        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

        if model_args["vision_encoder_pretrained"] != "openai":
            # load openai weights first - as we save only the visual weights, it doesn't work to load the full model
            vision_encoder_pretrained_ = "openai"
        else:
            vision_encoder_pretrained_ = model_args["vision_encoder_pretrained"]

        (
            self.model,
            image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            model_args["vision_encoder_path"],
            vision_encoder_pretrained_,
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
            compute_all_grads=adversarial,
        )
        self.image_processor_no_norm = transforms.Compose(image_processor.transforms[:-1])
        self.normalizer = image_processor.transforms[-1]
        del image_processor  # make sure we don't use it by accident
        self.adversarial = adversarial
        # image processor (9B model, probably same for others):
            # Compose(
            #   Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
            #   CenterCrop(size=(224, 224))
            #   <function _convert_to_rgb at 0x7fb90724ee80>
            #   ToTensor()
            # )

        if model_args["vision_encoder_pretrained"] != "openai":
            print("Loading non-openai vision encoder weights")
            self.model.vision_encoder.load_state_dict(torch.load(model_args["vision_encoder_pretrained"], map_location=self.device))


        checkpoint = torch.load(model_args["checkpoint_path"], map_location=self.device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device, dtype=self.cast_dtype)
        self.model.eval()
        self.tokenizer.padding_side = "left"

    def _prepare_images(self, batch: List[List[torch.Tensor]], preprocessor=None) -> torch.Tensor:
        """Preprocess images and stack them. Returns unnormed images.

        Args:
            batch: A list of lists of images.
            preprocessor: If specified, use this preprocessor instead of the default.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor_no_norm(image) if not preprocessor else preprocessor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: torch.Tensor,
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            with self.autocast():
                # x_vis = self._prepare_images(batch_images).to(
                #         self.device, dtype=self.cast_dtype, non_blocking=True
                #     )
                x_vis = batch_images.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    )
                x_vis = self.normalizer(x_vis)
                outputs = unwrap_model(self.model).generate(
                    x_vis,
                    input_ids.to(self.device, non_blocking=True),
                    attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_logits(
        self,
        lang_x: torch.Tensor,
        vision_x_unnorm: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
        labels: torch.Tensor = None,
    ):
        with torch.inference_mode(not self.adversarial):
            with self.autocast():
                outputs = self.model(
                    vision_x=self.normalizer(vision_x_unnorm),
                    lang_x=lang_x,
                    labels=labels,
                    attention_mask=attention_mask.bool(),
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=(past_key_values is not None),
                )
        return outputs

    def __call__(self, vision_x_unnorm):
        assert self.lang_x is not None
        assert self.attention_mask is not None
        assert self.labels is not None
        outputs = self.get_logits(
            self.lang_x,
            vision_x_unnorm=vision_x_unnorm,
            attention_mask=self.attention_mask,
            past_key_values=self.past_key_values,
            clear_conditioned_layers=True,
            labels=None  # labels are considered below
        )
        logits = outputs.logits
        loss_expanded = compute_loss(logits, self.labels)
        return loss_expanded
        # return outputs.loss

    def set_inputs(
        self,
        batch_text: List[str],
        past_key_values: torch.Tensor = None,
        to_device: bool = False,
    ):
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        self.lang_x = encodings["input_ids"]
        labels = get_label(lang_x=self.lang_x, tokenizer=self.tokenizer, mode="colon")
        self.labels = labels
        self.attention_mask = encodings["attention_mask"]
        self.past_key_values = past_key_values
        if to_device:
            self.lang_x = self.lang_x.to(self.device)
            self.attention_mask = self.attention_mask.to(self.device)
            self.labels = self.labels.to(self.device)
            if self.past_key_values is not None:
                self.past_key_values = self.past_key_values.to(self.device)


    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None) -> str:
        if answer and ":" in answer:
            answer = answer.replace(":", "")
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        if caption and ":" in caption:
            caption = caption.replace(":", "")
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

def compute_loss(logits, labels):
    bs = logits.shape[0]
    labels = torch.roll(labels, shifts=-1)
    labels[:, -1] = -100
    loss_expanded = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1),
        reduction='none'
    )
    loss_expanded = loss_expanded.view(bs, -1).sum(-1)
    return loss_expanded

def get_cast_dtype(precision: str):
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision in ["fp16", "float16"]:
        cast_dtype = torch.float16
    elif precision in ["fp32", "float32", "amp_bf16"]:
        cast_dtype = None
    else:
        raise ValueError(f"Unknown precision {precision}")
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress