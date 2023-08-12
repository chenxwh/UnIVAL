# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import Optional
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchaudio
from fairseq import utils, tasks, options
from fairseq import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from tasks.mm_tasks.caption import CaptionTask
from tasks.mm_tasks.refcoco import RefcocoTask
from tasks.mm_tasks.vqa_gen import VqaGenTask

from utils.zero_shot_utils import zero_shot_step
from data.video_utils import VIDEO_READER_FUNCS
from data.audio_utils import (
    get_audio_features,
    int16_to_float32,
    float32_to_int16,
    AUDIO_CFG,
)

from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    answer: Optional[str]
    output: Optional[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Register tasks
        tasks.register_task("caption", CaptionTask)
        tasks.register_task("refcoco", RefcocoTask)
        tasks.register_task("vqa_gen", VqaGenTask)
        tasks.register_task("video_caption", CaptionTask)
        tasks.register_task("audio_caption", CaptionTask)

        # Load ckpt & config for Image Captioning
        checkpoint_path_caption = (
            "checkpoints/unival_caption_stage_1/checkpoint_best_test.pt"
        )

        caption_overrides = {
            "eval_cider": False,
            "beam": 5,
            "max_len_b": 22,
            "no_repeat_ngram_size": 3,
            "seed": 7,
            "unnormalized": False,
            "bpe_dir": "utils/BPE",
            "video_model_path": None,
            "resnet_model_path": None,
        }

        (
            caption_models,
            caption_cfg,
            caption_task,
        ) = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(checkpoint_path_caption), arg_overrides=caption_overrides
        )

        # Load ckpt & config for Video Captioning
        checkpoint_path_video = (
            "checkpoints/unival_video_caption_stage_1/checkpoint_best.pt"
        )

        (
            video_caption_models,
            self.video_caption_cfg,
            video_caption_task,
        ) = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(checkpoint_path_video), arg_overrides=caption_overrides
        )

        # Load ckpt & config for Audio Captioning
        checkpoint_path_audio = "checkpoints/unival_audio_caption/checkpoint_best.pt"

        caption_overrides_audio = {
            "eval_cider": False,
            "beam": 5,
            "max_len_b": 22,
            "no_repeat_ngram_size": 3,
            "seed": 7,
            "unnormalized": False,
            "bpe_dir": "utils/BPE",
            "video_model_path": None,
            "resnet_model_path": None,
            "audio_model_path": None,
        }

        (
            audio_caption_models,
            self.audio_caption_cfg,
            audio_caption_task,
        ) = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(checkpoint_path_audio),
            arg_overrides=caption_overrides_audio,
        )

        # Load ckpt & config for Refcoco
        checkpoint_path_refcoco = "checkpoints/unival_refcocog/checkpoint_best.pt"

        refcoco_overrides = {
            "bpe_dir": "utils/BPE",
            "video_model_path": None,
            "resnet_model_path": None,
        }

        (
            refcoco_models,
            refcoco_cfg,
            refcoco_task,
        ) = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(checkpoint_path_refcoco), arg_overrides=refcoco_overrides
        )
        refcoco_cfg.common.seed = 7
        refcoco_cfg.generation.beam = 5
        refcoco_cfg.generation.min_len = 4
        refcoco_cfg.generation.max_len_a = 0
        refcoco_cfg.generation.max_len_b = 4
        refcoco_cfg.generation.no_repeat_ngram_size = 3

        # Load pretrained ckpt & config for VQA
        checkpoint_path_vqa = "checkpoints/unival_vqa/checkpoint_best.pt"

        overrides = {"video_model_path": None, "resnet_model_path": None}
        parser = options.get_generation_parser()
        input_args = [
            "",
            "--task=vqa_gen",
            "--beam=100",
            "--unnormalized",
            f"--path={checkpoint_path_vqa}",
            "--bpe-dir=utils/BPE",
        ]
        args = options.parse_args_and_arch(parser, input_args)
        vqa_cfg = convert_namespace_to_omegaconf(args)
        vqa_task = tasks.setup_task(vqa_cfg.task)
        vqa_models, vqa_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(vqa_cfg.common_eval.path),
            task=vqa_task,
            arg_overrides=overrides,
        )

        # Load pretrained ckpt & config for Generic Interface
        checkpoint_path_general = "checkpoints/unival_s2_hs/checkpoint1.pt"

        parser = options.get_generation_parser()
        input_args = [
            "",
            "--task=refcoco",
            "--beam=10",
            f"--path={checkpoint_path_general}",
            "--bpe-dir=utils/BPE",
            "--no-repeat-ngram-size=3",
            "--patch-image-size=384",
        ]
        args = options.parse_args_and_arch(parser, input_args)
        general_cfg = convert_namespace_to_omegaconf(args)
        self.general_task = tasks.setup_task(general_cfg.task)

        overrides = {"video_model_path": None, "resnet_model_path": None}

        general_models, general_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(general_cfg.common_eval.path),
            task=self.general_task,
            arg_overrides=overrides,
        )

        # move models to gpu
        use_fp16, use_cuda = True, True
        move2gpu(caption_models, caption_cfg, use_fp16, use_cuda)
        move2gpu(refcoco_models, refcoco_cfg, use_fp16, use_cuda)
        move2gpu(vqa_models, vqa_cfg, use_fp16, use_cuda)
        move2gpu(general_models, general_cfg, use_fp16, use_cuda)
        move2gpu(video_caption_models, general_cfg, use_fp16, use_cuda)
        move2gpu(audio_caption_models, general_cfg, use_fp16, use_cuda)

        # # Initialize generator
        caption_generator = caption_task.build_generator(
            caption_models, caption_cfg.generation
        )
        refcoco_generator = refcoco_task.build_generator(
            refcoco_models, refcoco_cfg.generation
        )
        vqa_generator = vqa_task.build_generator(vqa_models, vqa_cfg.generation)
        general_generator = self.general_task.build_generator(
            general_models, general_cfg.generation
        )
        video_caption_generator = caption_task.build_generator(
            video_caption_models, self.video_caption_cfg.generation
        )
        audio_caption_generator = caption_task.build_generator(
            audio_caption_models, self.audio_caption_cfg.generation
        )
        # Construct image transforms
        caption_transform = construct_transform(caption_cfg.task.patch_image_size)
        refcoco_transform = construct_transform(refcoco_cfg.task.patch_image_size)
        vqa_transform = construct_transform(vqa_cfg.task.patch_image_size)
        general_transform = construct_transform(general_cfg.task.patch_image_size)

        self.task = {
            "Image Captioning": caption_task,
            "Video Captioning": video_caption_task,
            "Audio Captioning": audio_caption_task,
            "Visual Question Answering": vqa_task,
            "Visual Grounding": refcoco_task,
            "General": self.general_task,
        }

        self.models = {
            "Image Captioning": caption_models,
            "Video Captioning": video_caption_models,
            "Audio Captioning": audio_caption_models,
            "Visual Question Answering": vqa_models,
            "Visual Grounding": refcoco_models,
            "General": general_models,
        }

        self.generator = {
            "Image Captioning": caption_generator,
            "Video Captioning": video_caption_generator,
            "Audio Captioning": audio_caption_generator,
            "Visual Question Answering": vqa_generator,
            "Visual Grounding": refcoco_generator,
            "General": general_generator,
        }

        self.transform = {
            "Image Captioning": caption_transform,
            "Visual Question Answering": vqa_transform,
            "Visual Grounding": refcoco_transform,
            "General": general_transform,
        }

        self.cfg = {
            "Image Captioning": caption_cfg,
            "Video Captioning": self.video_caption_cfg,
            "Audio Captioning": self.audio_caption_cfg,
            "Visual Question Answering": vqa_cfg,
            "Visual Grounding": refcoco_cfg,
            "General": general_cfg,
        }

        self.default_instruction = {
            "Image Captioning": "what does the image describe?",
            "Video Captioning": "what does the video describe?",
            "Audio Captioning": "what does the audio describe?",
        }

    def predict(
        self,
        input_image: Path = Input(description="Input image.", default=None),
        input_audio: Path = Input(description="Input audio.", default=None),
        input_video: Path = Input(description="Input video.", default=None),
        task_type: str = Input(
            description="Choose a task.",
            choices=[
                "Image Captioning",
                "Video Captioning",
                "Audio Captioning",
                "Visual Grounding",
                "General",
                "General Video",
            ],
            default="Image Captioning",
        ),
        instruction: str = Input(
            description="Provide question for the VQA task, region for Visual Grounding task, and instruction for General tasks. The default instruction for Captioning task is 'What does the image/video/audio describe?'",
            default=None,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        use_cuda, use_fp16 = True, True
        task_key = "General" if task_type == "General Video" else task_type

        if instruction is None:
            assert (
                "Captioning" in task_type
            ), f"Please provide instruction for {task_type} task."
            instruction = self.default_instruction[task_type]
        if task_type == "Visual Grounding":
            instruction = f'which region does the text " {instruction} " describe?'

        if "Video" in task_type:
            assert (
                input_video is not None
            ), f"Please provide input video for the {task_type} task."
            sample = construct_video_sample(
                str(input_video), self.video_caption_cfg, self.general_task
            )
        elif "Audio" in task_type:
            assert (
                input_audio is not None
            ), f"Please provide input audio for the {task_type} task."
            sample = construct_audio_sample(
                str(input_audio), self.audio_caption_cfg, self.general_task
            )
        else:
            assert (
                input_image is not None
            ), f"Please provide input image for the {task_type} task."
            transform = self.transform[task_key]
            image = Image.open(str(input_image))
            sample = construct_sample(image, instruction, transform, self.general_task)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        task = self.task[task_key]
        models = self.models[task_key]
        generator = self.generator[task_key]
        cfg = self.cfg[task_key]

        # Generate result
        with torch.no_grad():
            if task_type == "Visual Question Answering":
                result, scores = zero_shot_step(task, generator, models, sample)
                tokens = result[0]["answer"]
                bins = ""
            else:
                hypos = task.inference_step(generator, models, sample)
                tokens, bins, imgs = decode_fn(
                    hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator
                )

        if bins.strip() != "":
            w, h = image.size
            w_resize_ratio = task.cfg.patch_image_size / w
            h_resize_ratio = task.cfg.patch_image_size / h
            img = cv2.imread(str(input_image))
            coord_list = bin2coord(bins, w_resize_ratio, h_resize_ratio, cfg)
            cv2.rectangle(
                img,
                (int(coord_list[0]), int(coord_list[1])),
                (int(coord_list[2]), int(coord_list[3])),
                (0, 255, 0),
                3,
            )
            out_path = "/tmp/out.png"
            cv2.imwrite(out_path, img)

            return ModelOutput(answer=None, output=Path(out_path))
        else:
            return ModelOutput(answer=tokens, output=None)


def process_video(
    video_path,
    video_caption_cfg,
    max_num_frames=16,
    num_frames=16,
    sample_type="rand",
):
    # video
    data_path = os.path.join(video_path)

    # video process
    video_reader = VIDEO_READER_FUNCS["decord"]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    frames, frame_indices, video_duration = video_reader(
        data_path, num_frames, sample_type, max_num_frames=max_num_frames
    )

    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    patch_video_resize_transform = transforms.Compose(
        [
            transforms.CenterCrop(video_caption_cfg.task.patch_frame_size),
            type_transform,
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    patch_video = patch_video_resize_transform(frames)
    patch_video = patch_video.permute(1, 0, 2, 3)  # -> (C, T, h, w)

    return patch_video.unsqueeze(0)


def construct_video_sample(video_path, video_caption_cfg, general_task):
    pad_idx = general_task.src_dict.pad()
    patch_video = process_video(
        video_path,
        video_caption_cfg,
        max_num_frames=16,
        num_frames=video_caption_cfg.task.num_frames,
        sample_type=video_caption_cfg.task.sample_type,
    )
    patch_image = torch.zeros(
        (
            3,
            video_caption_cfg.task.patch_image_size,
            video_caption_cfg.task.patch_image_size,
        )
    )

    patch_type = torch.tensor([1])
    patch_mask = torch.tensor([True])
    src_text = encode_text(
        " what does the video describe?", general_task, append_bos=True, append_eos=True
    ).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(["42"]),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_videos": patch_video,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
            "patch_types": patch_type,
        },
    }
    return sample


def process_audio(
    audio_path, sample_rate=48000, max_audio_len=480000, audio_cfg=AUDIO_CFG
):
    # audio
    data_path = audio_path

    audio_data, orig_sr = torchaudio.load(data_path)
    audio_data = torchaudio.transforms.Resample(orig_sr, sample_rate)(audio_data[0])

    sample = {}

    sample = get_audio_features(
        sample,
        audio_data,
        max_audio_len,
        data_truncating="rand_trunc",
        data_filling="repeatpad",
        audio_cfg=audio_cfg,
    )

    waveform = sample["waveform"]
    patch_audio = waveform

    return patch_audio.unsqueeze(0)


def construct_audio_sample(audio_path, audio_caption_cfg, general_task):
    pad_idx = general_task.src_dict.pad()
    patch_audio = process_audio(
        audio_path, sample_rate=48000, max_audio_len=480000, audio_cfg=AUDIO_CFG
    )
    patch_image = torch.zeros(
        (
            3,
            audio_caption_cfg.task.patch_image_size,
            audio_caption_cfg.task.patch_image_size,
        )
    )

    patch_type = torch.tensor([2])
    patch_mask = torch.tensor([True])
    src_text = encode_text(
        " what does the audio describe?", general_task, append_bos=True, append_eos=True
    ).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(["42"]),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_audios": patch_audio,
            "patch_masks": patch_mask,
            "patch_types": patch_type,
        },
    }
    return sample


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(
        x.int().cpu(),
        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
    )
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
        if token.startswith("<bin_"):
            bin_result.append(token)
        elif token.startswith("<code_"):
            img_result.append(token)
        else:
            if bpe is not None:
                token = bpe.decode("{}".format(token))
            if tokenizer is not None:
                token = tokenizer.decode(token)
            if token.startswith(" ") or len(token_result) == 0:
                token_result.append(token.strip())
            else:
                token_result[-1] += token

    return " ".join(token_result), " ".join(bin_result), " ".join(img_result)


def bin2coord(bins, w_resize_ratio, h_resize_ratio, cfg):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [
        bin_list[0] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / w_resize_ratio
    ]
    coord_list += [
        bin_list[1] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / h_resize_ratio
    ]
    coord_list += [
        bin_list[2] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / w_resize_ratio
    ]
    coord_list += [
        bin_list[3] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / h_resize_ratio
    ]
    return coord_list


def encode_text(text, general_task, length=None, append_bos=False, append_eos=False):
    bos_item = torch.LongTensor([general_task.src_dict.bos()])
    eos_item = torch.LongTensor([general_task.src_dict.eos()])

    line = [
        general_task.bpe.encode(" {}".format(word.strip()))
        if not word.startswith("<code_") and not word.startswith("<bin_")
        else word
        for word in text.strip().split()
    ]
    line = " ".join(line)
    s = general_task.tgt_dict.encode_line(
        line=line, add_if_not_exist=False, append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


def construct_sample(image: Image, instruction: str, transform, general_task):
    patch_image = transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    pad_idx = general_task.src_dict.pad()

    instruction = encode_text(
        " {}".format(instruction.lower().strip()),
        general_task,
        append_bos=True,
        append_eos=True,
    ).unsqueeze(0)
    instruction_length = torch.LongTensor(
        [s.ne(pad_idx).long().sum() for s in instruction]
    )
    ref_dict = np.array([{"yes": 1.0}])  # just placeholder
    sample = {
        "id": np.array(["42"]),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "ref_dict": ref_dict,
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def move2gpu(models, cfg, use_fp16, use_cuda):
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)


def construct_transform(patch_image_size):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize(
                (patch_image_size, patch_image_size), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return patch_resize_transform
