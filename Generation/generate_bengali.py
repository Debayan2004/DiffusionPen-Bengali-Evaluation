
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torch.nn import DataParallel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

from unet import UNetModel
from feature_extractor import ImageEncoder
from bengali_dataset import BengaliDiffusionDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------
SAVE_PATH = r"D:\DiffusionPen\bengali_1000epoch_run"
STABLE_DIFF_PATH = r"D:\DiffusionPen\repo\stable-diffusion-v1-5"
STYLE_PATH = (
    r"D:\DiffusionPen\diffusionpen_project_files"
    r"\mixed_bengali_mobilenetv2_100.pth"
)
OUTPUT_DIR = r"D:\DiffusionPen\bengali_generated"

WORD_TO_GENERATE = "নিজস্ব"
WRITER_ID = 0

DEVICE = "cuda:0"
NUM_STEPS = 50
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Device:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Load Bengali dataset and select five style images
    # --------------------------------------------------
    print("Loading Bengali dataset...")
    dataset = BengaliDiffusionDataset(split="train")

    if not 0 <= WRITER_ID < len(dataset.writer_to_id):
        raise ValueError(
            f"WRITER_ID must be between "
            f"0 and {len(dataset.writer_to_id) - 1}"
        )

    writer_name = next(
        name
        for name, idx in dataset.writer_to_id.items()
        if idx == WRITER_ID
    )

    candidates = dataset.by_writer.get(writer_name, [])

    if not candidates:
        raise RuntimeError(
            f"No training images found for writer {writer_name}"
        )

    chosen = random.choices(candidates, k=5)

    style_images = torch.stack([
        dataset.load_image(dataset.df.iloc[i])
        for i in chosen
    ]).to(DEVICE)

    print("Writer:", writer_name, "| ID:", WRITER_ID)
    print("Style-reference shape:", tuple(style_images.shape))
    print("Text:", WORD_TO_GENERATE)

    # --------------------------------------------------
    # Load CANINE text encoder
    # --------------------------------------------------
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    text_encoder = CanineModel.from_pretrained(
        "google/canine-c"
    ).to(DEVICE).eval()

    text_encoder.requires_grad_(False)

    text_encoder = DataParallel(
        text_encoder,
        device_ids=[0]
    )

    # --------------------------------------------------
    # Match training-time U-Net configuration
    # --------------------------------------------------
    args = type("Args", (), {})()

    args.device = DEVICE
    args.img_size = (64, 256)
    args.channels = 4
    args.emb_dim = 320
    args.num_res_blocks = 1
    args.num_heads = 4
    args.latent = True
    args.img_feat = True
    args.interpolation = False
    args.mix_rate = None
    args.model_name = "diffusionpen"
    args.unet = "unet_latent"
    args.color = True

    unet = UNetModel(
        image_size=args.img_size,
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=len(dataset.writer_to_id),
        context_dim=args.emb_dim,
        vocab_size=81,
        text_encoder=text_encoder,
        args=args,
    )

    unet = DataParallel(unet, device_ids=[0]).to(DEVICE)

    checkpoint = os.path.join(
        SAVE_PATH, "models", "ema_ckpt.pt"
    )

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True
    )

    unet.load_state_dict(state)
    unet.eval()

    print("EMA checkpoint loaded.")

    # --------------------------------------------------
    # Load VAE
    # --------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        STABLE_DIFF_PATH,
        subfolder="vae"
    ).to(DEVICE).eval()

    vae.requires_grad_(False)
    vae = DataParallel(vae, device_ids=[0])

    # --------------------------------------------------
    # Load DDIM scheduler
    # --------------------------------------------------
    scheduler = DDIMScheduler.from_pretrained(
        STABLE_DIFF_PATH,
        subfolder="scheduler"
    )

    # --------------------------------------------------
    # Load MobileNetV2 style encoder and checkpoint
    # --------------------------------------------------
    style_encoder = ImageEncoder(
        model_name="mobilenetv2_100",
        num_classes=0,
        pretrained=False,
        trainable=False,
    )

    style_state = torch.load(
        STYLE_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    current = style_encoder.state_dict()

    compatible = {
        key: value
        for key, value in style_state.items()
        if key in current
        and current[key].shape == value.shape
    }

    current.update(compatible)
    style_encoder.load_state_dict(current)

    print(
        "Style encoder checkpoint tensors loaded:",
        len(compatible),
        "/",
        len(current)
    )

    style_encoder = DataParallel(
        style_encoder,
        device_ids=[0]
    ).to(DEVICE).eval()

    style_encoder.requires_grad_(False)

    # --------------------------------------------------
    # Inference and text-conditioning diagnostic
    # --------------------------------------------------
    with torch.no_grad():

        style_features = style_encoder(style_images)

        print(
            "Style feature shape:",
            tuple(style_features.shape)
        )

        text_inputs = tokenizer(
            [WORD_TO_GENERATE],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(DEVICE)

        # Initial latent
        x = torch.randn(
            (1, 4, 8, 32),
            device=DEVICE
        )

        # Compare prompts with identical latent and style.
        test_words = ["বাংলা", "নিজস্ব"]
        test_timesteps = [999, 750, 500, 250, 50]
        fixed_x = x.clone()

        print("\nText-conditioning diagnostic:")

        for timestep in test_timesteps:

            fixed_t = torch.tensor(
                [timestep],
                device=DEVICE,
                dtype=torch.long
            )

            predictions = []

            for word in test_words:

                test_inputs = tokenizer(
                    [word],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=40,
                ).to(DEVICE)

                pred = unet(
                    fixed_x,
                    timesteps=fixed_t,
                    context=test_inputs,
                    y=torch.tensor(
                        [WRITER_ID],
                        device=DEVICE
                    ),
                    style_extractor=style_features,
                )

                predictions.append(pred)

            difference = (
                predictions[0] - predictions[1]
            ).abs().mean().item()

            print(
                f"t={timestep}: "
                f"mean absolute prediction difference="
                f"{difference:.8f}"
            )

        print()

        # --------------------------------------------------
        # DDIM sampling
        # --------------------------------------------------
        scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

        for step, time in enumerate(
            scheduler.timesteps,
            start=1
        ):

            t = torch.full(
                (1,),
                int(time.item()),
                device=DEVICE,
                dtype=torch.long
            )

            predicted_noise = unet(
                x,
                timesteps=t,
                context=text_inputs,
                y=torch.tensor(
                    [WRITER_ID],
                    device=DEVICE
                ),
                style_extractor=style_features,
            )

            x = scheduler.step(
                predicted_noise,
                time,
                x
            ).prev_sample

            if step % 10 == 0:
                print(
                    f"Sampling step {step}/{NUM_STEPS}"
                )

        # --------------------------------------------------
        # Decode latent into an image
        # --------------------------------------------------
        decoded = vae.module.decode(
            x / 0.18215
        ).sample

        decoded = (
            (decoded / 2 + 0.5)
            .clamp(0, 1)[0]
            .cpu()
        )

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------
    output = transforms.ToPILImage()(decoded)

    output_path = os.path.join(
         OUTPUT_DIR,
         f"bengali_{WORD_TO_GENERATE}_{WRITER_ID}.png"
    )

    output.save(output_path)

    print("\nSaved generated image:", output_path)


if __name__ == "__main__":
    main()
