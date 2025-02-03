import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
import torch
import numpy as np
import gradio as gr
import logging
import traceback
from i18n.i18n import I18nAuto
from configs.config import Config
from infer.modules.vc.modules import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# === Minimal setup for model inference ===
config = Config()
vc = VC(config)
i18n = I18nAuto()

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)

index_paths = []


def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)

gpus = "-".join([])  # (Optional) adapt this if you want GPU info or usage

# === Only the Model Inference tab code ===
with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>"
            "如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        # ==========  只保留【模型推理】Tab内容  ==========
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("刷新音色列表和索引路径"), variant="primary"
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )

                def clean():
                    return {"value": "", "__type__": "update"}

                clean_button.click(fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")

            with gr.TabItem(i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                value=0,
                            )
                            input_audio0 = gr.Textbox(
                                label=i18n("输入待处理音频文件路径(默认是正确格式示例)"),
                                placeholder="C:\\Users\\Desktop\\audio_example.wav",
                            )
                            file_index1 = gr.Textbox(
                                label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method0 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,"
                                    "crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                                ),
                                choices=["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"],
                                value="rmvpe",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，"
                                    "调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，"
                                    "使用可以削弱哑音"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"),
                                visible=False,
                            )

                            def change_choices():
                                """Refresh choices for voice and index."""
                                names_new = []
                                for name in os.listdir(weight_root):
                                    if name.endswith(".pth"):
                                        names_new.append(name)
                                index_paths_new = []
                                def lookup_indices_for_refresh(index_root):
                                    for root, dirs, files in os.walk(index_root, topdown=False):
                                        for name in files:
                                            if name.endswith(".index") and "trained" not in name:
                                                index_paths_new.append("%s/%s" % (root, name))
                                lookup_indices_for_refresh(index_root)
                                lookup_indices_for_refresh(outside_index_root)
                                return {
                                    "choices": sorted(names_new),
                                    "__type__": "update",
                                }, {
                                    "choices": sorted(index_paths_new),
                                    "__type__": "update",
                                }

                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                                api_name="infer_refresh",
                            )

                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                            vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )

            with gr.TabItem(i18n("批量推理")):
                gr.Markdown(
                    value=i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,"
                                "crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，"
                                "调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.2,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，"
                                "使用可以削弱哑音"
                            ),
                            value=4,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=2,
                            label=i18n("检索特征占比"),
                            value=0.95,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("输出信息"))

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )

            # def get_vc_data(sid0_val, p0, p1):
            #     """Simulate the function originally hooking sid0 change."""
            #     # Just returns some placeholders that the original code sets.
            #     return 0, p0, p1, "", ""

            sid0.change(
                fn=vc.get_vc,
                inputs=[sid0, protect0, protect1],
                outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                api_name="infer_change_voice",
            )
            # ==========  仅保留【模型推理】Tab内容结束  ==========

# === Launch app ===
if config.iscolab:
    app.queue().launch(share=True)
else:
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=10492,
        quiet=True,
    )