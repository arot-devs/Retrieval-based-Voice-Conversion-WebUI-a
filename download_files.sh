#!/bin/bash

mkdir -p pretrained_v2
mkdir -p uvr5_weights

# v2
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth -O ./pretrained_v2/D32k.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -O ./pretrained_v2/D40k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth -O ./pretrained_v2/D48k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth -O ./pretrained_v2/G32k.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -O ./pretrained_v2/G40k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth -O ./pretrained_v2/G48k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth -O ./pretrained_v2/f0D32k.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -O ./pretrained_v2/f0D40k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth -O ./pretrained_v2/f0D48k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth -O ./pretrained_v2/f0G32k.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -O ./pretrained_v2/f0G40k.pth
# wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth -O ./pretrained_v2/f0G48k.pth


# @title #下载人声分离模型
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -O .//uvr5_weights/HP2-人声vocals+非人声instrumentals.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -O .//uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

# @title #下载hubert_base
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O ./hubert_base.pt

# @title #下载rmvpe模型
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -O ./rmvpe.pt
