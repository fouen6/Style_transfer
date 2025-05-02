import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from models import VGG
from utils import make_transform, load_image, save_image, calculate_style_loss, calculate_content_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Style Transfer=')
    parser.add_argument('--content_image', type=str, default='./data/content1.jpg', help='Path to the content image')
    parser.add_argument('--style_image', type=str, default='./data/style2.jpg', help='Path to the style image')
    parser.add_argument('--output_dir', type=str, default='./output/iterative_style_transfer', help='Output directory')
    parser.add_argument('--image_size', type=int, nargs=2, default=[300, 450], help='Image size (height, width)')
    parser.add_argument('--content_weight', type=float, default=1, help='Content weight')
    parser.add_argument('--style_weight', type=float, default=15, help='Style weight')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of steps per epoch')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='Learning rate')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 检查文件路径
    assert os.path.exists(args.content_image), f"content image is not exist: {args.content_image}"
    assert os.path.exists(args.style_image), f"style image is not exist: {args.style_image}"

    # 内容特征层及loss加权系数
    content_layers = {'5': 0.5, '10': 0.5}
    # 风格特征层及loss加权系数
    style_layers = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}

    # ----------------训练即推理过程----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform(args.image_size, normalize=True)

    content_img = load_image(args.content_image, transform).to(device)
    style_img = load_image(args.style_image, transform).to(device)

    # 用小随机噪声初始化图像（避免标准正态分布过大震荡）
    generated_img = torch.rand_like(content_img).mul(0.1).requires_grad_().to(device)
    save_image(generated_img, args.output_dir, 'noise_init.jpg')

    vgg_model = VGG(content_layers, style_layers).to(device).eval()

    # 关闭梯度追踪，节省显存
    with torch.no_grad():
        content_features, _ = vgg_model(content_img)
        _, style_features = vgg_model(style_img)

    optimizer = optim.Adam([generated_img], lr=args.learning_rate)

    for epoch in range(args.epochs):
        p_bar = tqdm(range(args.steps_per_epoch), desc=f'Epoch {epoch + 1}/{args.epochs}')
        for step in p_bar:
            generated_content, generated_style = vgg_model(generated_img)

            content_loss = sum(
                args.content_weight * content_layers[name] * calculate_content_loss(content_features[name], gen_content)
                for name, gen_content in generated_content.items()
            )
            style_loss = sum(
                args.style_weight * style_layers[name] * calculate_style_loss(style_features[name], gen_style)
                for name, gen_style in generated_style.items()
            )

            total_loss = style_loss + content_loss

            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪（可选但稳健）
            torch.nn.utils.clip_grad_norm_([generated_img], max_norm=1.0)

            optimizer.step()

            p_bar.set_postfix(style_loss=style_loss.item(), content_loss=content_loss.item(), total_loss=total_loss.item())

        # 保存中间结果
        save_image(generated_img, args.output_dir, f'generated_epoch_{epoch + 1}.jpg')
