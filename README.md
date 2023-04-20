- Stable Diffusion Intro:
  - Text-to-image latent diffusion model
  - Created by CompVis, Stability AI, and LAION
  - Trained on 512x512 images from [LAION-5B](https://laion.ai/blog/laion-5b/) database subset
  - Uses frozen CLIP ViT-L/14 text encoder
    - part of the [CLIP](https://huggingface.co/sentence-transformers/clip-ViT-L-14)
  - 860M UNet and 123M text encoder
  - Lightweight, runs on consumer GPUs
<br>
<br>
- Try Stable Diffusion from Huggingface
  - install some packages
    ```bash
    pip install -r requirements.txt
  
    # if you are in China
    pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
    ```

  - Pipeline
    - end2end
    - pretrained model on huggingface:
      - CompVis/stable-diffusion-v1-4。 512x512
      - runwayml/stable-diffusion-v1-5. 512x512
      - stabilityai/stable-diffusion-2-1-base. 512x512
      - stabilityai/stable-diffusion-2-1. 768x768
    - for a faster inference and lower memory usage, use `fp16`, also pass `torch_dtype = torch.float16`








  