import gradio as gr
from netramodules import gdinoeye, yoloeye


DESCRIPTION = '''# <div align="center">EYE iris detection/identification Inference Demo. by Yuri D.Kobyzev(c) v.0.0.3 </div>
<div align="center">
</div>

'''

if __name__ == '__main__':
    title = 'EYE grounding dino prompt iris/pupil detection'
    with gr.Blocks(analytics_enabled=False, title=title, ) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Yolo'):
                yoloeye.EYE()
            with gr.TabItem('Gdino'):
                gdinoeye.EYE()
                

if __name__ == "__main__":
#    demo.queue().launch(auth=("gdino","gdino"), debug=True, root_path="/gdino",server_port=7861)
    demo.queue(default_concurrency_limit=5).launch(debug=True,share=True)
