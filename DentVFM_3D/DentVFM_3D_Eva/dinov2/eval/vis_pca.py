# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from sklearn.decomposition import PCA
from dinov2.eval.setup import setup_and_build_model_3d
import argparse
from monai.transforms import (
    Compose, LoadImage, ScaleIntensityRangePercentiles, Lambda, CropForeground, Resize, Identity
)
from monai.inferers import sliding_window_inference
import nibabel


# Set random seed for reproducibility
torch.manual_seed(42)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--image-path",
        default="",
        type=str,
        help="Path to image to visualize",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory (not needed)",
        default=None
    )
    parser.add_argument(
        "--vis-type",
        help="Type of visualization to perform",
        type=str,
        default='pca'
    )
    parser.add_argument(
        "--input-type",
        help="Type of input to use for visualization",
        type=str,
        default='full_image'
    )

    return parser


def get_pca_feat_vector(tensor_input=None, feature_size=1024, img_shape=(512, 512, 512), patch_size=(16, 16, 16)):
    patch_num = tuple(img_shape[i] // patch_size[i] for i in range(3))
    print("Patch Number:", patch_num)
    if tensor_input is None:
        tensor_input = torch.randn(*patch_num, feature_size)
    # Reshape the tensor
    reshaped_input = tensor_input.reshape(-1, feature_size)
    # Perform PCA
    pca = PCA(n_components=10)
    pca_feat_vector = pca.fit_transform(reshaped_input)
    # Reshape PCA features back to the original patch structure
    pca_feat_vector = pca_feat_vector.reshape(*patch_num, 10)
    print("PCA Feature Vector Shape:", pca_feat_vector.shape)
    # normalize vector
    pca_feat_vector = (pca_feat_vector - pca_feat_vector.min()) / (pca_feat_vector.max() - pca_feat_vector.min())
    print(pca_feat_vector.max(), pca_feat_vector.min())

    return pca_feat_vector, patch_num


@torch.inference_mode()
def visualize(args):
    # get pretrained model
    model, autocast_dtype = setup_and_build_model_3d(args)

    def random_select_time(x):
        # if time axis exists, select random time slice
        if x.shape[0] > 1:
            t = torch.randint(0, x.shape[0] - 1).item()
            x = x[t:t + 1]
        return x

    if args.input_type == 'resize':
        resize_transform = Resize((96, 96, 96))
    else:
        resize_transform = Identity()

    data_transform = Compose(
        [
            LoadImage(ensure_channel_first=True),
            Lambda(func=random_select_time),
            Lambda(func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())),  # replace NaNs with mean
            ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
            CropForeground(select_fn=lambda x: x > -1, k_divisible=16),
            resize_transform,
        ]
    )

    # load image
    img = data_transform(args.image_path)

    def model_forward(x):
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            if args.vis_type == 'pca':
                output = model.get_intermediate_layers(x, reshape=True)[0]
            elif args.vis_type == 'mhsa':
                output = model.get_self_attention(x, reshape=True)
            else:
                raise ValueError(f'Unknown vis_type {args.vis_type}')
        return output

    # get volume of mhsa or pca, resize to original image size with interpolation
    img = img.cuda()
    if args.input_type == 'full_image' or args.input_type == 'resize':
        output = model_forward(img.unsqueeze(0))
    elif args.input_type == 'sliding_window':
        output = sliding_window_inference(img.unsqueeze(0), (96, 96, 96), 4, model_forward, overlap=0.)
    else:
        raise ValueError(f'Unknown input_type {args.input_type}')

    # get PCA
    if args.vis_type == 'pca':
        tokens = output[0].cpu().numpy()
        tokens = np.transpose(tokens, (1, 2, 3, 0))
        pca_feat_vector, patch_num = get_pca_feat_vector(
           tokens, feature_size=1024, img_shape=img.shape[1:], patch_size=(16, 16, 16)
        )
        vis_volume = torch.nn.functional.interpolate(
           torch.tensor(pca_feat_vector).permute(3, 0, 1, 2).unsqueeze(0),
           size=img.shape[1:],
           mode="nearest",
        ).squeeze(0).numpy()

    # get MHSA
    elif args.vis_type == 'mhsa':
        vis_volume = torch.nn.functional.interpolate(
            output.cpu(),
            size=img.shape[1:],
            mode="nearest",
        ).squeeze(0).numpy()

    else:
        raise ValueError(f'Unknown vis_type {args.vis_type}')

    # visualize all heads/pca components
    for h in range(vis_volume.shape[0]):
        vol = vis_volume[h]
        nifti = nibabel.Nifti1Image(vol, affine=img.meta['affine'])
        nibabel.save(nifti, f'{args.output_dir}/nifti_{h}.nii.gz')

    orig_img = nibabel.Nifti1Image(img.cpu().numpy()[0], affine=img.meta['affine'])
    nibabel.save(orig_img, f'{args.output_dir}/orig.nii.gz')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    visualize(args)
